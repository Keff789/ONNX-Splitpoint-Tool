from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.runners.request_latency import (
    RequestLatency, aggregate_latency, latency_fields, validate_latency,
)


def recorder(n, *, task_complete=True):
    return RequestLatency(n, task_complete=task_complete, start_anchor="test:admission", end_anchor="test:task_completion")


def sample(values, **kwargs):
    r = recorder(len(values), **kwargs)
    for i, value in enumerate(values):
        r.start(i, i * 3_000_000)
        r.complete(i, i * 3_000_000 + round(value * 1_000_000))
    return r.report()


def test_serial_request_pairs_and_known_skew_quantiles():
    r = sample([1, 2, 3, 4, 90])
    assert r["status"] == "complete"
    assert r["count"] == r["expected_count"] == 5
    assert r["mean_ms"] == 20
    assert r["p50_ms"] == 3
    assert r["p95_ms"] == pytest.approx(72.8)
    assert (r["min_ms"], r["max_ms"]) == (1, 90)


def test_overlapping_identical_inputs_out_of_order_are_distinct_requests():
    same_bytes = np.zeros(3, dtype=np.uint8)
    r = recorder(3)
    for i in range(3):
        assert same_bytes.tobytes() == b"\0\0\0"
        r.start(i, i * 3_000_000)
    for i in [2, 0, 1]:
        r.complete(i, i * 3_000_000 + 12_000_000)
    assert r.report()["count"] == 3
    assert r.report()["mean_ms"] == 12


@pytest.mark.parametrize("fault", ["duplicate_start", "duplicate_end", "missing", "missing_start", "unknown_id", "negative", "clock", "timestamp_type"])
def test_invalid_pairing_never_filters_into_success(fault):
    r = recorder(2)
    r.start(0, 1_000_000); r.complete(0, 3_000_000)
    if fault != "missing_start": r.start(1, 4_000_000)
    if fault != "missing": r.complete(1, 5_000_000 if fault != "negative" else 3_000_000)
    if fault == "duplicate_start": r.start(0, 1_000_000)
    if fault == "duplicate_end": r.complete(0, 3_000_000)
    if fault == "unknown_id": r.complete(2, 8_000_000)
    if fault == "clock": r.complete(1, 8_000_000, clock_domain="another_target:clock")
    if fault == "timestamp_type": r.start(1, 1.5)
    result = r.report()
    assert result["status"] == "invalid"
    assert result["mean_ms"] is None
    assert result["expected_count"] == 2
    assert any(result["errors"].values())


@pytest.mark.parametrize("stop", ["cancel", "timeout", "last_request_incomplete"])
def test_interrupted_work_keeps_uncompleted_request_in_denominator(stop):
    r = recorder(4)
    r.start(0, 0); r.complete(0, 100)
    r.start(1, 200)
    failure = r.failure(stop)
    result = failure.request_latency
    assert result["status"] == "invalid"
    assert (result["count"], result["expected_count"]) == (1, 4)
    assert result["errors"]["missing_completions"] == 3


def test_duration_mode_has_no_wallclock_sized_new_buffer():
    r = RequestLatency(2, task_complete=True, start_anchor="a", end_anchor="b", enabled=False)
    for i in range(1000): r.start(i); r.complete(i)
    assert r.starts == r.ends == []
    assert r.report()["status"] == "unavailable"


def test_pooled_quantiles_use_all_requests_and_keep_repeats():
    records = [{"repetition_id": "a", "request_latency": sample([1, 1])},
               {"repetition_id": "b", "request_latency": sample([100] * 8)}]
    result = aggregate_latency(records, requested=2)
    assert result["mean_ms"] == pytest.approx(80.2)
    assert result["p50_ms"] == result["p95_ms"] == 100
    assert len(result["repetitions"]) == 2
    assert "ci95_low" not in result
    records[1]["request_latency"]["pairs"][-1][2] = None
    result = aggregate_latency(records, requested=2)
    assert result["status"] == "invalid" and result["p95_ms"] is None


def test_mixed_semantics_duplicate_replicates_and_missing_repeat_rejected():
    rows = [{"repetition_id": "a", "request_latency": sample([12])},
            {"repetition_id": "b", "request_latency": sample([12], task_complete=False)}]
    assert aggregate_latency(rows)["reason"] == "repetition_latency_semantics_mismatch"
    rows[1] = dict(rows[0])
    assert aggregate_latency(rows)["reason"] == "repetition_identity_missing_or_duplicate"
    assert aggregate_latency(rows[:1], requested=2)["status"] == "invalid"


def test_historical_values_preserved_without_invented_request_latency():
    old = {"task": "classification", "completed_frames": 100, "makespan_ms": 309,
           "repetition_count_requested": 1, "repetition_count_valid": 1,
           "fps_makespan": 100 / .309, "measurement_boundary": "workers_ready_to_last_completed_trt_frame",
           "latency_mean_ms": 7.25, "latency_semantics": "old_real_serial_host_time"}
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields
    assert rate_endpoint_fields(old)["completed_task_fps"] == pytest.approx(100/.309)
    assert rate_endpoint_fields(old)["request_latency_mean_ms"] is None
    assert old["latency_mean_ms"] == 7.25
    host = latency_fields({"request_latency": sample([7.25], task_complete=False)})
    assert host["host_output_latency_mean_ms"] == 7.25
    assert host["request_latency_mean_ms"] is None
    assert host["request_latency_unavailable_reason"] == "task_postprocess_not_in_measured_path"


def test_wrong_clock_domain_and_backwards_raw_pair_are_rejected():
    raw = sample([12])
    raw["clock"] = "wall_clock"
    assert validate_latency(raw)["status"] == "invalid"
    raw = sample([12]); raw["pairs"][0][2] = -1
    assert validate_latency(raw)["status"] == "invalid"


@pytest.mark.parametrize("backend", ["deepx", "hailo10"])
def test_real_python_fifo_scheduling_counts_warmup_and_delayed_completion(backend):
    from test_v272_detection_hotloop_integration import (
        _FakeCompletionRuntime, _FakeTRT, _FakeDeepX, _FakeHailo10Session,
    )
    from scripts import native_deepx_trt_e2e_from_benchmarkset as dx
    from scripts import native_hailo10_trt_e2e_from_benchmarkset as h10
    completion = _FakeCompletionRuntime(tail_s=.003)
    warmup = _FakeCompletionRuntime(tail_s=0)
    options = dict(frames=12, warmup=2, queue_depth=1, task="detection",
                   completion_runtime=completion, warmup_completion_runtime=warmup)
    if backend == "deepx":
        result = dx._deepx_fifo_run(_FakeDeepX(), np.zeros((1,4)), _FakeTRT(), **options)
    else:
        result = h10._hailo10_async_fifo_run(_FakeHailo10Session(), {"input": np.zeros((1,4))}, _FakeTRT(), inflight=3, **options)
    evidence = result["request_latency"]
    assert result["completed_work_units"] == completion.completed_count == evidence["count"] == 12
    assert warmup.completed_count == 2
    assert evidence["status"] == "complete"
    assert evidence["min_ms"] >= 3
    # Backlog and a postprocessing tail belong to requests, not just P2 time.
    assert evidence["mean_ms"] > result["p2_run_ms"] + 3
    assert evidence["warmup_included"] is False


def test_async_full_ids_survive_slot_reuse_and_callbacks_include_postprocess():
    from test_v269_native_runner_p0_completion import _completion_session
    session, _ = _completion_session()
    def finish(_outputs): time.sleep(.001)
    result = session.benchmark_throughput({"input": np.zeros(1)}, frames=5, inflight=2, warmup_frames=3,
                                          postprocess_callback=finish)
    raw = result["request_latency"]
    assert raw["status"] == "complete"
    assert raw["count"] == 5 and raw["min_ms"] >= 1
    assert [x[0] for x in raw["pairs"]] == list(range(5))
    assert result["warmup_completed_frames"] == 3


def test_twelve_ms_requests_three_ms_cadence_through_raw_summary_gui_csv(tmp_path):
    from scripts import native_producer_final_report as report
    from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
    from onnx_splitpoint_tool.workflow.runner import _native_concise_summary_v60w
    from onnx_splitpoint_tool.workflow.scientific_reporting import _write_reports
    from onnx_splitpoint_tool.native_rate_endpoints import format_rate_endpoints
    raw = {"task": "detection", "case": "b001", "model": "counterexample",
           "completed_frames": 100, "completed_work_units": 100, "makespan_ms": 309,
           "repetition_count_requested": 1, "repetition_count_valid": 1,
           "fps_makespan": 100/.309, "measurement_endpoint": "completed_task",
           "measurement_boundary": "workers_ready_to_last_completed_task_frame",
           "postprocess_completion_verified": True, "completed_task_endpoint_attested": True,
           "completed_task_endpoint_contract_hash": "c" * 64, "ok": True,
           "request_latency": sample([12] * 100)}
    path = tmp_path / "raw.json"; path.write_text(json.dumps(raw))
    tables = tmp_path / "analysis_tables"; tables.mkdir()
    (tables / "native_fifo_eval_runner_r9b.json").write_text(json.dumps({"rows": [
        {"model": "counterexample", "case": "b001", "report": str(path), "ok": True}]}))
    row, = report._rows_from_native_fifo_runner(tmp_path, include_direct_fallback=False)
    reports = tmp_path / "reports"; reports.mkdir()
    (reports / "native_producer_combined_summary.json").write_text(json.dumps({"rows": [row]}))
    matrix = collect_native_performance_matrix(tmp_path)
    normalized, = matrix["observations"]
    assert normalized["completed_task_fps"] == pytest.approx(100/.309)
    assert normalized["request_latency_mean_ms"] == normalized["request_latency_p95_ms"] == 12
    _, concise = _native_concise_summary_v60w(reports)
    assert concise[0]["request_latency_mean_ms"] == 12
    text = format_rate_endpoints(normalized)
    assert "323.625 FPS" in text and "Mean 12.000 / P50 12.000 / P95 12.000 ms" in text
    assert "n=100/100" in text and "ab vorbereitetem Input" in text
    _write_reports(reports / "scientific", {"created_at":"2026-09-16", "profile_id":"r9b", "rows":[], "summary":{},
                                          "native_performance_matrix":matrix})
    import csv
    with (reports / "scientific/native_performance_observations.csv").open() as stream:
        exported, = list(csv.DictReader(stream))
    assert float(exported["request_latency_mean_ms"]) == 12
    assert float(exported["completed_task_fps"]) == pytest.approx(100/.309)


@pytest.mark.parametrize("field,value", [("admission_wait_included",False),("warmup_included",True),
    ("start_anchor",""),("end_anchor",None),("expected_count","100")])
def test_invalid_semantic_flags_do_not_crash_or_claim_complete(field,value):
    raw=sample([12]);raw[field]=value
    assert latency_fields({"request_latency":raw})["request_latency_status"]=="invalid"
    assert latency_fields({"request_latency":{"schema":"broken"}})["request_latency_status"]=="invalid"


def test_pooled_concise_only_replay_revalidates_original_pairs():
    rows=[{"repetition_id":"a","request_latency":sample([1,2,90])},
          {"repetition_id":"b","request_latency":sample([3,4])}]
    first=latency_fields({"repetition_records":rows,"repetition_count_requested":2})
    second=latency_fields(first)
    assert second==first
    assert second["request_latency_p95_ms"]==pytest.approx(72.8)
    second["request_latency"]["repetitions"][0]["request_latency"]["pairs"][0][2]=None
    assert latency_fields(second)["request_latency_status"]=="invalid"


def test_h8_actual_fifo_pairs_and_failure_preserve_missing_last_request():
    from test_v272_hailo8_completed_detection_hotloop import _run,_Completion
    backend,trt,result=_run(completion=_Completion(),warmup_completion=_Completion(),frames=8,warmup=2)
    assert result["request_latency"]["count"]==8
    assert result["request_latency"]["expected_count"]==8
    with pytest.raises(RuntimeError,match="completion failed") as caught:
        _run(completion=_Completion(fail_on=3),frames=8)
    raw=caught.value.request_latency
    assert raw["status"]=="invalid" and raw["expected_count"]==8 and raw["count"]==2
    assert raw["errors"]["missing_completions"]>0


def test_h10_full_callback_failure_preserves_partial_pairs():
    from test_v269_native_runner_p0_completion import _completion_session
    session,_=_completion_session()
    calls=0
    def complete(outputs):
        nonlocal calls
        calls+=1
        if calls==3:raise RuntimeError("test callback failed")
    with pytest.raises(RuntimeError) as caught:
        session.benchmark_throughput({"images":np.zeros((1,2),dtype=np.float32)},frames=5,
            inflight=2,warmup_frames=0,postprocess_callback=complete)
    raw=caught.value.request_latency
    assert raw["expected_count"]==5 and raw["status"]=="invalid" and raw["count"]==3
    assert raw["pairs"][2][2] is None # failed callback; an already admitted later request completes


def test_identical_completion_counts_in_two_local_replacement_processes(tmp_path):
    import os,subprocess,sys,textwrap
    code=textwrap.dedent("""
        import json,sys
        from concurrent.futures import ThreadPoolExecutor
        from onnx_splitpoint_tool.runners.request_latency import RequestLatency
        active=sys.argv[1]=='1'
        r=RequestLatency(32,task_complete=True,start_anchor='local_process:before_executor_admission',end_anchor='local_process:after_completion',enabled=active)
        def work(i):
            result=sum([7]*1000)
            r.complete(i)
            return (i,result)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[]
            for i in range(32):
                r.start(i);futures.append(pool.submit(work,i))
            rows=[f.result() for f in futures]
        print(json.dumps({'rows':rows,'latency':r.report()}))
    """)
    results=[]
    for active in ('0','1'):
        result=subprocess.run([sys.executable,'-B','-c',code,active],capture_output=True,text=True,check=True,timeout=30)
        results.append(json.loads(result.stdout))
    assert results[0]['rows']==results[1]['rows'] and len(results[1]['rows'])==32
    assert results[0]['latency']['status']=='unavailable'
    assert results[1]['latency']['count']==32


def test_recorder_hotpath_has_no_io_hashes_or_device_synchronization():
    import ast,inspect
    import onnx_splitpoint_tool.runners.request_latency as module
    tree=ast.parse(inspect.getsource(module))
    forbidden={'open','write','write_text','dump','dumps','sha256','synchronize','wait','sleep'}
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef) and node.name in {'_record','start','complete'}:
            calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
                   for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))]
            assert not forbidden.intersection(calls)


def test_saved_pair_errors_cannot_be_erased_by_repairing_only_raw_pairs():
    raw=sample([12]);raw['errors']['missing_completions']=1
    assert validate_latency(raw)['status']=='invalid'


def test_historical_repeats_without_pairs_remain_unavailable():
    value=latency_fields({'repetition_records':[{'repetition_id':'a'},{'repetition_id':'b'}]})
    assert value['request_latency_status']=='unavailable'
    assert value['request_latency_unavailable_reason']=='request_timestamps_missing'


@pytest.mark.parametrize('endpoint',['completed_task','raw_model_outputs'])
def test_single_repetition_preserves_rate_but_does_not_publish_degenerate_ci(endpoint):
    from onnx_splitpoint_tool.native_rate_endpoints import project_rate_endpoint
    fps=3/.018
    raw={'task':'classification','measurement_endpoint':endpoint,
         'measurement_boundary':'workers_ready_to_last_completed_trt_frame',
         'completed_frames':3,'makespan_ms':18,'fps_makespan':fps,
         'fps_ci95_low':fps,'fps_ci95_high':fps,'fps_ci95_method':'legacy_bootstrap_n1'}
    result=project_rate_endpoint(raw,endpoint)
    assert result['status']=='available' and result['fps']==fps
    assert result['ci95_low'] is result['ci95_high'] is None
    assert result['ci95_method']=='not_estimated_single_repetition'
    assert raw['fps_ci95_low']==raw['fps_ci95_high']==fps
