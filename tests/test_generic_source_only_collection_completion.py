"""Real portable capture checks at the normal dispatch boundary; no hardware."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.runners import generic_quality_accounting as accounting
from onnx_splitpoint_tool.workflow import execution_binding as binding
from tests.test_generic_quality_accounting import real_capture


def dispatch(real_capture, tmp_path, monkeypatch, *, fault=None, service_status="failed"):
    fixture, request_path, report, report_path = real_capture
    cfg = fixture["config"]
    config_path = tmp_path / "quality_runtime/b039.json"
    config_path.parent.mkdir()
    config_path.write_bytes(fixture["path"].read_bytes())
    source = accounting._file(config_path)
    row = {"id": "hailo10_to_tensorrt", "type": "matrix", "stage1": "hailo10",
           "stage2": "tensorrt", "hardware_setup_id": cfg["identity"]["setup_id"],
           "quality_source_only": True, "variants": ["composed"], "case_ids": ["b039"],
           "generic_quality_completion_configs": {"b039": "quality_runtime/b039.json"},
           "generic_quality_completion_config_sha256": {"b039": source["sha256"]},
           "generic_quality_completion_sources": {"b039": source}}
    profile = {"run_profiles": [deepcopy(row)], "quality_gate": {
        "statistics": {"execution_location": "central_management"}}, "energy": {"enabled": False}}
    plan = {"runs": [row]}
    if fault == "mixed_performance":
        plan["runs"].append({"id": "full_fps", "type": "matrix", "variants": ["full"]})
    if fault == "full":
        profile["run_profiles"] = []
        plan = {"runs": [{"id": "full_fps", "type": "matrix", "variants": ["full"]}]}
    (tmp_path / "benchmark_plan.json").write_text(json.dumps(plan))
    benchmark_set = tmp_path / "benchmark_set.json"
    benchmark_set.write_text("{}")
    request_copy = request_path.with_name("composed_request.json")
    request_copy.write_bytes(request_path.read_bytes())
    copied = [{"kind": "central_quality_input", "destination": str(request_copy)},
              {"kind": "case_validation_report", "destination": str(report_path)}]
    if fault == "candidate_missing":
        candidate = json.loads(request_copy.read_text())["candidate"]["path"]
        (request_copy.parent / candidate).unlink()
    elif fault == "request_missing":
        copied.pop(0)
    elif fault == "child_mismatch":
        report["task_quality_input_requests_by_variant"]["composed"]["request"]["sha256"] = "a" * 64
        report_path.write_text(json.dumps(report))
    remote = {"ok": service_status == "ok", "status": service_status,
              "local_run_dir": str(tmp_path), "remote_rc": 0,
              "remote_dispatched": True, "remote_dispatch_failed": False,
              "terminal_remote_failure": False, "failure_kind": "", "primary_failure": {},
              "error": "Remote result collection failed (rc=1); remote run retained",
              "remote_cleanup": {"reason": "result_collection_not_successful",
                  "policy": {"results_downloaded": True, "result_collection_rc": 1}}}
    if fault == "execution_failed":
        remote["remote_rc"] = 1
    elif fault == "cancelled":
        remote["status"] = "cancelled"
    elif fault == "not_downloaded":
        remote["remote_cleanup"]["policy"]["results_downloaded"] = False
    elif fault == "other_failure":
        remote["error"] = "Actual model execution failure"
    class Service:
        def run(self, **kwargs):
            return deepcopy(remote)
    monkeypatch.setattr("onnx_splitpoint_tool.benchmark.services.RemoteBenchmarkService", Service)
    monkeypatch.setattr(binding, "_copy_remote_result_files", lambda *a, **k: copied)
    output = tmp_path / "dispatch"
    output.mkdir()
    result = binding._run_remote_dispatch_once(
        run_root=tmp_path.parent / "synthetic", model_id="yolo26s",
        options=SimpleNamespace(remote_working_dir=str(tmp_path / "downloads")),
        profile_payload=profile, suite_dir=tmp_path, benchmark_set_json=benchmark_set,
        result_dir=output, gates={}, log=None, target_id=cfg["identity"]["setup_id"],
        runtime_override={"enabled": True, "host": "192.0.2.10", "user": "test"})
    status = json.loads((output / ("remote_benchmark_status_" + cfg["identity"]["setup_id"] + ".json")).read_text())
    return result, status


def test_exact_portable_source_capture_reaches_pending_central_quality(real_capture, tmp_path, monkeypatch):
    result, status = dispatch(real_capture, tmp_path, monkeypatch)
    assert result.status == "ok", status
    assert status["generic_quality_source_capture"]["status"] == "verified_exact"
    assert status["capture_status"] == "capture_complete"
    assert status["central_quality_status"] == "pending_central_quality"
    assert status["canonical_nonempty_row_count"] == 0
    assert status["capture_admitted_after_collection_error"] is True
    assert status["remote_output"]["status"] == "failed"
    assert status["remote_output"]["remote_cleanup"]["policy"]["result_collection_rc"] == 1


@pytest.mark.parametrize("fault", ["request_missing", "candidate_missing", "child_mismatch",
                                  "full", "mixed_performance", "execution_failed",
                                  "cancelled", "not_downloaded", "other_failure"])
def test_missing_or_unrelated_evidence_never_gets_the_capture_exception(real_capture, tmp_path, monkeypatch, fault):
    result, status = dispatch(real_capture, tmp_path, monkeypatch, fault=fault)
    assert result.status != "ok", status
    assert status.get("capture_admitted_after_collection_error") is not True


def test_original_successful_source_capture_path_is_preserved(real_capture, tmp_path, monkeypatch):
    result, status = dispatch(real_capture, tmp_path, monkeypatch, service_status="ok")
    assert result.status == "ok", status
    assert status["capture_status"] == "capture_complete"
    assert status["capture_admitted_after_collection_error"] is False
