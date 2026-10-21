"""Actual completion proofs and explicit profile authority, without inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.runners import generic_quality_accounting as accounting
from onnx_splitpoint_tool.runners import generic_quality_completion as completion
from onnx_splitpoint_tool.workflow.runner import _performance_plan_v263, _selected_run_completeness_v60r
from tests.test_generic_quality_completion import make_config, envelope
from tests.test_generic_quality_profile_transport import profile


def projected():
    source = profile()
    source["quality_gate"]["statistics"] = {"execution_location": "central_management"}
    run = deepcopy(source["run_profiles"][0]); run["type"] = "matrix"
    return source, {"runs": [run]}


def test_only_explicit_profile_authority_removes_fps_and_reprojection_restores_it():
    source, plan = projected()
    result = _performance_plan_v263(plan, source)
    assert result["runs"] == []
    assert result["performance_matrix_applicable"] is False
    reconciled = _selected_run_completeness_v60r(benchmark_plan=result, source_records=[])
    assert reconciled["expected_run_count"] == 0
    assert reconciled["performance_matrix_applicable"] is False
    assert reconciled["matrix_complete"] is None
    assert reconciled["generic_quality_only_runs"]
    for unauthorized in ({}, {**source, "run_profiles": []}):
        restored = _performance_plan_v263(result, unauthorized)
        assert len(restored["runs"]) == 1
        assert restored["performance_matrix_applicable"] is True
        assert not accounting.projected_runs(restored)


@pytest.mark.parametrize("fault", ["cases", "source", "hash", "flag", "variants", "setup"])
def test_selected_plan_cannot_change_profile_authority(fault):
    source, plan = projected()
    row = plan["runs"][0]
    if fault == "cases": row["case_ids"] = ["b040"]
    elif fault == "source": row["generic_quality_completion_sources"]["b039"]["path"] = "/other/config.json"
    elif fault == "hash": row["generic_quality_completion_config_sha256"]["b039"] = "c"*64
    elif fault == "flag": row["quality_source_only"] = False
    elif fault == "variants": row["variants"] = ["full"]
    else: row["hardware_setup_id"] = "other"
    with pytest.raises(ValueError): _performance_plan_v263(plan, source)


@pytest.fixture
def real_capture(tmp_path):
    fixture = make_config(tmp_path)
    request, _, request_path, candidate_path = envelope(fixture, tmp_path)
    candidate = json.loads(candidate_path.read_text())
    producer = request["producer_identity"]
    producer.update(eval_run_id="synthetic", source_run_id="hailo10_to_tensorrt", performance_claims_emitted=False)
    producer.pop("producer_identity_sha256")
    producer["producer_identity_sha256"] = completion._hash(producer)
    proof = completion.completed_quality_export_fields(request[completion.FIELD], candidate["records"], producer)
    candidate.update(producer_identity=producer, **proof)
    candidate_path.write_text(json.dumps(candidate))
    request.update(**fixture["config"]["identity"], eval_run_id="synthetic", source_run_id="hailo10_to_tensorrt",
                   performance_claims_emitted=False, producer_identity=producer, **proof)
    request["candidate"].update(accounting._file(candidate_path), path=candidate_path.name)
    request_path.write_text(json.dumps(request))
    descriptor = accounting._file(request_path)
    report = {"quality_source_only": True, "performance_measurement_executed": False, "measured_variants": [],
              "variant_status": {"composed": "ok"},
              "task_quality_input_requests_by_variant": {"composed": {**request, "request": descriptor}},
              "timing_ms": {"composed": {"runs_ms": [], "mean_ms": None, "measurement_endpoint": ""}}}
    report_path = tmp_path/"validation_report.json"; report_path.write_text(json.dumps(report))
    return fixture, request_path, report, report_path


def test_actual_factory_proof_is_collected_separately_from_fps(real_capture):
    fixture, request_path, report, report_path = real_capture
    result = accounting.collect_capture(report_path, fixture["config"], config_path=fixture["path"],
                                        run_id="hailo10_to_tensorrt", eval_run_id="synthetic")
    assert result["image_count"] == 2
    assert result["performance_measurement_executed"] is False
    assert result["central_quality_decision"] == "pending"
    assert result["measured_variants"] == []


@pytest.mark.parametrize("fault", ["selected_config", "request_bytes", "candidate_bytes", "identity", "report_request", "measurement", "count", "fps", "fps_string", "fps_bool"])
def test_capture_and_child_contradictions_reject(real_capture, fault):
    fixture, request_path, report, report_path = real_capture
    if fault == "selected_config": fixture["path"].write_text(fixture["path"].read_text()+" ")
    elif fault == "request_bytes": request_path.write_text('{}')
    elif fault == "candidate_bytes": request_path.with_name('candidate.json').write_text('{}')
    elif fault == "identity": fixture["config"]["identity"]["setup_id"] = "wrong"
    elif fault == "report_request": report["task_quality_input_requests_by_variant"]["composed"]["request"]["sha256"] = 'a'*64
    elif fault == "measurement": report["performance_measurement_executed"] = True
    elif fault == "count": report["timing_ms"]["composed"]["runs_ms"] = [1.0]
    elif fault == "fps": report["fps"] = 123.0
    elif fault == "fps_string": report["fps"] = "123"
    else: report["fps"] = True
    report_path.write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        accounting.collect_capture(report_path, fixture["config"], config_path=fixture["path"],
                                   run_id="hailo10_to_tensorrt", eval_run_id="synthetic")


def test_distinct_generic_normalized_contract_requires_exact_projection_and_counts():
    source, plan = projected()
    perf = _performance_plan_v263(plan, source)
    payload = {"results": [], "result_count": 0, "status": "quality_evidence_only_complete",
        "performance_matrix_applicable": False, "quality_evidence_only_complete": True,
        "matrix_complete": True, "quality_evidence_count": 1,
        "expected_full_quality_count": 0, "expected_generic_quality_count": 1,
        "generic_quality_only_projection": perf["generic_quality_only_projection"]}
    assert accounting.normalized_contract(payload)
    for key, value in (("expected_generic_quality_count", 2), ("quality_evidence_count", 0),
                       ("expected_full_quality_count", 1), ("generic_quality_only_projection", {}),
                       ("performance_matrix_applicable", True), ("matrix_complete", False),
                       ("results", [{"fps": 3.0}])):
        assert not accounting.normalized_contract({**payload, key:value})


@pytest.mark.parametrize("backend,stage", [("hailo10", {"type":"hailo","hw_arch":"hailo10"}),
                                        ("deepx_m1", {"type":"deepx","backend":"deepx_m1"})])
def test_actual_legacy_nested_stage_descriptor_matches_authorized_profile(backend, stage):
    source, plan = projected()
    source["run_profiles"][0]["stage1"] = backend
    plan["runs"][0]["stage1"] = stage
    plan["runs"][0]["stage2"] = {"type":"onnxruntime","provider":"tensorrt"}
    assert _performance_plan_v263(plan, source)["runs"] == []


def test_controller_verifies_portable_capture_without_reopening_dut_artifacts(real_capture, tmp_path):
    fixture, request_path, report, report_path = real_capture
    # These bytes existed for genuine construction/capture, then are absent
    # from the controller's view, as a DUT-only installed artifact would be.
    Path(fixture["config"]["expected_runtime_artifacts"]["part1_runtime"]["path"]).unlink()
    relative = tmp_path/"quality_runtime/b039.json";relative.parent.mkdir()
    relative.write_bytes(fixture["path"].read_bytes())
    source = accounting._file(relative)
    row = {"id":"hailo10_to_tensorrt", "hardware_setup_id":fixture["config"]["identity"]["setup_id"],
           "case_ids":["b039"], "generic_quality_completion_configs":{"b039":"quality_runtime/b039.json"},
           "generic_quality_completion_config_sha256":{"b039":source["sha256"]}}
    copied_request = request_path.with_name("composed_request.json")
    copied_request.write_bytes(request_path.read_bytes())
    copied = [{"kind":"central_quality_input","destination":str(copied_request)},
              {"kind":"case_validation_report","destination":str(report_path)}]
    summary = accounting.copied_capture_summary(copied,[row],suite_dir=tmp_path,model_id="yolo26s",
        setup_id=fixture["config"]["identity"]["setup_id"],eval_run_id="synthetic")
    assert summary["status"] == "verified_exact", summary
    Path(fixture["config"]["runtime_source_files"][-1]["path"]).stat() # source code remains checked
