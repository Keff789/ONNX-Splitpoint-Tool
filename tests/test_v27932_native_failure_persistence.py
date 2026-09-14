from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
from onnx_splitpoint_tool.native_job_identity import failed_native_result, native_identity_key
from onnx_splitpoint_tool.workflow import runner
from scripts import native_producer_final_report as report


def jobs(producer="deepx"):
    common = dict(model="resnet50", setup_id=f"offline_{producer}_setup", comparison_backend=producer,
                  precision="float32_layout_fp16")
    return [dict(common, backend=producer+"_to_trt", case="b012", execution_mode="native_split"),
            dict(common, backend="native_full_"+producer, case="full", execution_mode="native_full_baseline"),
            dict(common, backend="native_full_tensorrt", case="full", execution_mode="native_full_baseline")]


def negative(job, reason="remote_authentication_failed"):
    return failed_native_result(job, failure_stage="native_transfer_or_dispatch", failure_reason=reason,
                                upstream_evidence_path="reports/native_producer_stage.json")


def persist(tmp_path, rows, planned, artifacts=None, roots=None):
    artifacts = artifacts if artifacts is not None else {}
    roots = roots if roots is not None else []
    runner._persist_native_blocked_rows(tmp_path, rows, expected_native_rows=planned,
        artifact_paths=artifacts, collected_roots=roots, repetition_count_requested=3)
    return tmp_path/"native_producers/_blocked", artifacts, roots


@pytest.mark.parametrize("producer", ["deepx", "hailo8", "hailo10h"])
def test_t32_f1_01_to_03_roles_report_and_matrix(tmp_path, producer):
    planned=jobs(producer)
    root, artifacts, roots=persist(tmp_path, [negative(j) for j in planned], planned)
    full=json.loads((root/"analysis_tables/native_full_baseline_eval.json").read_text())
    split=json.loads((root/f"analysis_tables/native_{producer}_producer_e2e_eval__prerequisites.json").read_text())
    assert full["schema_version"]==5 and full["row_count"]==2
    assert len(split["rows"])==1 and split["rows"][0]["execution_mode"]=="native_split"
    reader={"deepx":report._rows_from_deepx,"hailo8":report._rows_from_native_fifo_runner,"hailo10h":report._rows_from_hailo10}[producer]
    rows=report._aggregate_repetitions(report._rows_from_native_full(root)+reader(root))
    matrix=runner._native_expected_matrix_status_v60y(planned, rows, [])
    assert [matrix[k] for k in ("expected_row_count","present_expected_row_count","failed_expected_row_count","missing_expected_row_count")]==[3,3,3,0]
    assert len(artifacts)==2 and roots==[root]
    for row in rows:
        assert row["primary_failure_reason"]=="remote_authentication_failed"
        assert row["failure_stage"]=="native_transfer_or_dispatch"
        assert row["upstream_evidence_path"]=="reports/native_producer_stage.json"
        assert row["repetition_count_attempted"]==0 and row["repetition_count_valid"]==0
        assert row["repetition_count_requested"]==3 and row["repetition_records"]==[]
        assert row["fps_makespan"] is None


def test_t32_f1_04_nine_distinct_jobs(tmp_path):
    planned=sum((jobs(p) for p in ("deepx","hailo8","hailo10h")),[])
    root, artifacts, _=persist(tmp_path,[negative(j) for j in planned],planned)
    rows=json.loads((root/"analysis_tables/native_full_baseline_eval.json").read_text())["rows"]
    assert len(rows)==6 and len({native_identity_key(row) for row in rows})==6
    assert len(artifacts)==4


def test_t32_f1_05_06_idempotent_extension_preserves_specific_error(tmp_path):
    planned=jobs(); first=negative(planned[0],"exact_deterministic_build_blocker")
    root, artifacts, roots=persist(tmp_path,[first],planned)
    rows=[negative(j) for j in planned]
    persist(tmp_path,rows,planned,artifacts,roots)
    before={p.name:p.read_bytes() for p in (root/"analysis_tables").iterdir()}
    persist(tmp_path,rows,planned,artifacts,roots)
    assert before=={p.name:p.read_bytes() for p in (root/"analysis_tables").iterdir()}
    split=json.loads((root/"analysis_tables/native_deepx_producer_e2e_eval__prerequisites.json").read_text())
    assert split["rows"][0]["primary_failure_reason"]=="exact_deterministic_build_blocker"
    assert roots==[root] and len(artifacts)==2


@pytest.mark.parametrize("mutation",["missing_setup","wrong_setup","wrong_role","positive_collision"])
def test_t32_f1_11_invalid_identity_or_measurement_fails_closed(tmp_path, mutation):
    planned=jobs(); row=negative(planned[1])
    if mutation=="missing_setup":
        row["setup_id"]="";row["planned_native_identity"]["setup_id"]=""
    elif mutation=="wrong_setup": row["setup_id"]="other_setup"
    elif mutation=="wrong_role": row["execution_mode"]="native_split"
    else: row.update(ok=True,result_ok=True,fps_makespan=10)
    with pytest.raises(ValueError,match="native_blocked_"):
        persist(tmp_path,[row],planned)
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("backend",["native_full_deepx","native_full_hailo8","native_full_hailo10h","native_full_tensorrt"])
def test_t32_f1_10_12_full_origin_zero_starts_and_concise(tmp_path, backend):
    planned=dict(jobs()[1],backend=backend)
    row=negative(planned)
    row.update(execution_mode="native_full_baseline",runtime_success=False,repetition_count_requested=3,
               repetition_records=[],fps_makespan=None,execution_precision="",full_runtime_precision="",
               runtime_precision_source="unavailable",performance_claim_eligible=False)
    root=tmp_path/"native_producers/_blocked"; (root/"analysis_tables").mkdir(parents=True)
    companion=dict(row, quality_evidence_only=True,performance_claims_emitted=False,ok=True,fps_makespan=123)
    (root/"analysis_tables/native_full_baseline_eval.json").write_text(json.dumps({"rows":[row,companion]}))
    rows=report._aggregate_repetitions(report._rows_from_native_full(root))
    assert len(rows)==1
    assert rows[0]["primary_failure_reason"]=="remote_authentication_failed"
    assert rows[0]["failure_stage"]=="native_transfer_or_dispatch"
    assert rows[0]["repetition_count_attempted"]==0 and rows[0]["repetition_records"]==[]
    assert rows[0]["execution_precision"]=="" and rows[0]["returncode"] is None
    reports=tmp_path/"reports";reports.mkdir()
    (reports/"native_producer_combined_summary.json").write_text(json.dumps({"rows":rows}))
    _, concise=runner._native_concise_summary_v60w(reports)
    assert concise[0]["failure_reason"]=="remote_authentication_failed"
    assert concise[0]["primary_failure_reason"]=="remote_authentication_failed"


def test_t32_f1_07_independent_existing_measurement_is_not_rewritten(tmp_path):
    planned=jobs()
    measurement=tmp_path/"native_producers/hailo8/analysis_tables/native_full_baseline_eval.json"
    measurement.parent.mkdir(parents=True)
    fixture=Path(__file__).parent/"fixtures/v27931_complete_set/original_identity_projection.json"
    historical=json.loads(fixture.read_text())
    successful=next(row for row in historical["rows"] if row["ok"] and row["backend"]=="native_full_hailo8")
    measurement.write_text(json.dumps({"rows":[successful]}))
    before=measurement.read_bytes()
    persist(tmp_path,[negative(j) for j in planned],planned)
    assert measurement.read_bytes()==before


def test_t32_f1_11_existing_measurement_collision_does_not_overwrite(tmp_path):
    planned=jobs()
    root, _, _=persist(tmp_path,[negative(planned[1])],planned)
    path=root/"analysis_tables/native_full_baseline_eval.json"
    payload=json.loads(path.read_text())
    payload["rows"][0].update(ok=True,result_ok=True,runtime_success=True,
                              repetition_count_attempted=3,fps_makespan=123)
    path.write_text(json.dumps(payload))
    before=path.read_bytes()
    with pytest.raises(ValueError,match="native_blocked_measurement_collision"):
        persist(tmp_path,[negative(planned[1])],planned)
    assert path.read_bytes()==before


@pytest.mark.parametrize("backend", ["deepx", "hailo8", "hailo10h", "tensorrt"])
def test_t32_f1_started_full_exception_is_not_a_zero_start(backend):
    from subprocess import CompletedProcess
    # This is the actual Full.main stdout contract: exceptions are handled,
    # so rc=3 does not come with a traceback mentioning the runner filename.
    stdout="\n".join(f"[native-full] resnet50/{backend} repetition={i}/3" for i in range(1,4))
    result=CompletedProcess(["ssh", "fixture"],3,stdout+"\n{\"output\":\"native_full_baseline_eval.json\"}","")
    assert runner._native_remote_launch_observed(result)


@pytest.mark.parametrize("stdout", [
    "[native-full] engine_build_python=/fixture/venv/bin/python",
    "[native-full] resnet50/deepx preparing exact Full input",
    "[native-full] resnet50/deepx repetition=0/3",
    "Permission denied (publickey).",
])
def test_t32_f1_full_preparation_and_auth_failure_are_not_starts(stdout):
    from subprocess import CompletedProcess
    assert not runner._native_remote_launch_observed(CompletedProcess(["ssh","fixture"],3,stdout,""))
