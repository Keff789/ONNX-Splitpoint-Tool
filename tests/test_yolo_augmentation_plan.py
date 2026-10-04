"""Graph-only bounded augmentation; no compiler, device or quality execution."""
import copy
import hashlib
import json
from pathlib import Path

import onnx
import pytest

from onnx_splitpoint_tool import yolo_augmentation_plan as plan
from onnx_splitpoint_tool.generic_completion_plan import MODEL_COUNTS, SETUPS


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _candidate(b, *, eligible=True):
    return {"boundary": b, "case_id": f"b{b:03d}", "selection_eligible": eligible, "exclusions": [],
            "cut_tensors": [f"t{b}"], "part2_external_inputs": [f"t{b}"],
            "inputs": [{"name": f"t{b}", "shape": [1, 4, 8, 8], "elemtype": 1}]}


def test_graph_coverage_ignores_results_and_has_deterministic_tie_break():
    candidates = [_candidate(b) for b in [8, 67, 120, 118]]
    candidates[0].update(fps=100000, accuracy_class="reference_close")
    candidates[2].update(fps=1, accuracy_class="accuracy_loss")
    first = plan.choose_boundaries(candidates, [3, 62], 1)
    assert [r["boundary"] for r in first] == [120]
    assert first[0]["minimum_topological_distance"] == 58
    assert plan.choose_boundaries(candidates, [3, 62, 120], 1)[0]["boundary"] == 8
    assert plan.choose_boundaries([_candidate(365, eligible=False), _candidate(38)], [3, 21], 1)[0]["boundary"] == 38


def test_actual_analyzer_rejects_single_cut_with_second_original_p2_input(tmp_path):
    model_id = "yolo11l"
    root = tmp_path / "models" / model_id
    path = root / "benchmark_set/legacy_suite/models" / f"{model_id}.onnx"
    path.parent.mkdir(parents=True)
    vi = lambda name: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 4])
    nodes = [onnx.helper.make_node("Identity", ["x"], ["a"]),
             onnx.helper.make_node("Relu", ["a"], ["b"]),
             onnx.helper.make_node("Add", ["b", "z"], ["c"]),
             onnx.helper.make_node("Identity", ["c"], ["out"])]
    model = onnx.helper.make_model(onnx.helper.make_graph(nodes, "two_inputs", [vi("x"), vi("z")], [vi("out")]))
    onnx.save(model, path)
    _write(root / "model_manifest.json", {"observed_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    observed = plan.graph_candidates(tmp_path, model_id)
    assert [r["boundary"] for r in observed["candidates"]] == [2]
    assert observed["candidates"][0]["part2_external_inputs"] == ["c"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="graph identity"):
        plan.graph_candidates(tmp_path, model_id)


def _result(identity):
    config = {"identity": identity, "original_run": "run", "task": "detection",
              "input_binding": {"source": "prepared"}, "completion_contract": {"task": "detection"},
              "measurement_boundary": "prepared_input_before_p1_to_last_completed_task",
              "expected_runtime_artifacts": {"part1": "hef", "part2": "engine"},
              "functional_reference": {"source": "original"}, "queue_depth": 2,
              "repetitions": 3, "warmup_count": 100, "task_count": 1000}
    result = {**config, "schema_version": 1, "measurement_mode": "generic_completed_task", "execution_id": "finished", "status": "ok"}
    result["repetitions"] = [{"producer": "generic_split", "status": "ok", "repetition_index": i,
        "execution_id": "finished", "runtime_instance_id": f"process-{i}", "pid": i,
        "requested_warmup_count": 100, "warmup_completed_count": 100, "requested_task_count": 1000,
        "completed_task_count": 1000, "processor_count_delta": 1100,
        "warmup_end_ns": 1, "measurement_start_ns": 2, "measurement_end_ns": 1000000002,
        "measured_makespan_seconds": 1.0, "completed_task_fps": 1000,
        "measurement_boundary": config["measurement_boundary"], "clock": "perf_counter_ns",
        "worker_count": 2, "queue_depth": 2, "fill_and_drain_included": True,
        "handoff_timed": True, "postprocess_timed": True, "output_materialization_timed": True,
        "preprocessing_timed": False, "runtime_artifacts": config["expected_runtime_artifacts"],
        "functional_control": {"status": "passed", "checked_outside_timer": True,
                               "reference_binding": config["functional_reference"]}} for i in [1, 2, 3]]
    return config, result


@pytest.fixture
def base(tmp_path, monkeypatch):
    run, completion = tmp_path / "run", tmp_path / "completion"
    rows = []
    for model, count in MODEL_COUNTS.items():
        cases = plan.BASE_CASES.get(model, [f"b{i:03d}" for i in range(1, count + 1)])
        for setup, backend in SETUPS.items():
            for case in cases:
                direction = {"deepx": "deepx_to_trt", "hailo8": "hailo8_to_trt", "hailo10h": "hailo10h_to_trt"}[backend]
                identity = {"model_id": model, "case_id": case, "setup_id": setup,
                    "direction": direction, "precision": "fp16", "comparison_backend": backend, "output_endpoint_id": model + ":endpoint"}
                config, result = _result(identity)
                result_path = completion / "results" / model / setup / case / "result.json"
                _write(result_path, result)
                rows.append({**identity, "identity": identity, "case_key": "|".join(identity.values()),
                    "model_sha256": model + "-sha", "runtime_config": config, "result_path": str(result_path),
                    "original_manifest": str(run / model / case / "split_manifest.json"),
                    "native_source": str(run / model / case / "native.json"), "generic_original_quality_request": "original-request",
                    "input_binding": config["input_binding"], "completion_contract": config["completion_contract"],
                    "measurement_boundary": config["measurement_boundary"]})
    _write(completion / "plan/execution-plan.json", {"original_run": str(run), "rows": rows})
    _write(completion / "terminal-status.json", {"status": "complete", "exitcode": 0,
        "summary": {"valid_repetitions": 576, "completed_task_count": 576000, "status_counts": {"completed": 192}}})
    pools = {"yolo11l": [2, 3, 8, 62, 67, 118, 120], "yolo26s": [3, 8, 21, 26, 38, 364, 365], "yolov7_paper": [2, 9, 30, 44, 64, 66]}
    monkeypatch.setattr(plan, "graph_candidates", lambda _, m: {"model_id": m, "model_sha256": m + "-sha", "source_graph": str(run / (m + ".onnx")),
        "candidates": [_candidate(b, eligible=b not in [364, 365]) for b in pools[m]]})
    monkeypatch.setattr(plan, "_negative_records", lambda: [])
    monkeypatch.setattr(plan, "_paper_anchor", lambda graph: {"status": "same_original_graph_and_boundary"})
    return run, completion


def test_bounded_primary_and_standby_plan_preserves_original_files(base, tmp_path):
    run, completion = base
    before = {str(p): p.read_bytes() for p in completion.rglob("*.json")}
    actual = plan.build_augmentation_plan(run, completion, output_root=tmp_path / "derived")
    assert (actual["primary_count"], actual["reserve_count"]) == (12, 9)
    assert actual["models"]["yolo11l"]["primary_cases"] == ["b120"]
    assert actual["models"]["yolo26s"]["primary_cases"] == ["b038"]
    assert actual["models"]["yolov7_paper"]["primary_cases"] == ["b044", "b066"]
    assert actual["models"]["yolov7_paper"]["reserve_cases"] == ["b030"]
    assert actual["models"]["yolo26m"]["primary_cases"] == []
    assert len(actual["start_case_keys"]) == 3
    starters = [r for r in actual["rows"] if r["case_key"] in actual["start_case_keys"]]
    assert {r["setup_id"] for r in starters} == set(SETUPS)
    assert all(r["selection_role"] == "primary" for r in starters)
    assert not any(r["activated"] for r in actual["rows"] if r["selection_role"] == "reserve")
    assert before == {str(p): p.read_bytes() for p in completion.rglob("*.json")}
    assert actual["frozen"] is False


@pytest.mark.parametrize("damage", ["foreign_run", "changed_base", "incomplete_result", "foreign_graph", "paper"])
def test_rejects_foreign_or_incomplete_authority(base, monkeypatch, damage):
    run, completion = base
    path = completion / "plan/execution-plan.json"; data = json.loads(path.read_text())
    row = next(r for r in data["rows"] if r["model_id"] == "yolo11l")
    if damage == "foreign_run": data["original_run"] = str(run / "foreign")
    elif damage == "changed_base": row["case_id"] = "b061"
    elif damage == "foreign_graph": row["model_sha256"] = "foreign"
    elif damage == "incomplete_result":
        p = Path(row["result_path"]); r = json.loads(p.read_text()); r["repetitions"].pop(); _write(p, r)
    else: monkeypatch.setattr(plan, "_paper_anchor", lambda _: {"status": "historical_identity_mismatch"})
    _write(path, data)
    with pytest.raises(ValueError): plan.build_augmentation_plan(run, completion)


def test_original_output_tree_is_never_a_plan_destination(base):
    run, completion = base
    with pytest.raises(ValueError, match="outside preserved bases"):
        plan.build_augmentation_plan(run, completion, output_root=completion / "plan/new")


@pytest.fixture
def exact_terminal_case(tmp_path):
    from onnx_splitpoint_tool.native_job_identity import known_build_exclusion
    source = json.loads((Path(__file__).parent / "fixtures/v281_status/night_blocked_native_rows.json").read_text())
    terminal = next(r for r in source if known_build_exclusion(r))
    observation = terminal["upstream_build_observation"]
    context = observation["build_evidence"]["context"]
    manifest = copy.deepcopy(context["split_manifest"])
    # Historical contexts may predate the descriptive exact-input field; the
    # current exporter supplies it and the existing endpoint key is unchanged.
    manifest["part2_external_inputs"] = list(manifest["cut_tensors"])
    row = {"model_id": terminal["model"], "case_id": terminal["case"],
        "setup_id": terminal["setup_id"], "backend": terminal["comparison_backend"],
        "comparison_backend": terminal["comparison_backend"], "direction": terminal["backend"],
        "precision": terminal["precision"], "selection_role": "primary", "case_key": "exact-primary",
        "model_sha256": context["full_source_onnx_sha256"],
        "cut_tensors": manifest["cut_tensors"], "part2_external_inputs": manifest["part2_external_inputs"]}
    root = tmp_path / "augmentation"
    stage_path = root / "models" / row["model_id"] / "stages/build_backend_artifacts/stage_result.json"
    stage = {"details": {"deferred_build_readiness": {"blocked_jobs": [observation]}}}
    _write(stage_path, stage)
    _write(root / "models" / row["model_id"] / "benchmark_set/legacy_suite" / row["case_id"] / "split_manifest.json", manifest)
    reserve = {**row, "case_id": "b008", "case_key": "fixed-reserve", "selection_role": "reserve", "activated": False}
    cohort = {"augmentation_run": str(root), "rows": [row, reserve]}
    return cohort, root, stage_path, stage


def test_exact_normal_terminal_is_read_only_and_reserve_stays_unactivated(exact_terminal_case):
    cohort, root, path, _ = exact_terminal_case
    before, plan_before = path.read_bytes(), copy.deepcopy(cohort)
    rows = plan.collect_technical_terminals(cohort, root)
    assert len(rows) == 1 and rows[0]["status"] == "excluded_known_build"
    assert rows[0]["case_key"] == "exact-primary" and rows[0]["measurement_started"] is False
    proposed = plan.propose_reserve_activation(cohort, rows)
    assert proposed["executable_reserve_budget"] == 0
    assert len(proposed["proposals"]) == 1
    assert not proposed["proposals"][0]["activated"] and not proposed["proposals"][0]["dispatch_allowed"]
    assert path.read_bytes() == before and cohort == plan_before


@pytest.mark.parametrize("change", ["text_only", "runtime_started", "compiler_started", "wrong_full", "wrong_setup", "wrong_boundary", "duplicate", "wrong_current_cut", "wrong_record"])
def test_no_terminal_from_unbound_or_conflicting_build_evidence(exact_terminal_case, change):
    cohort, root, path, stage = exact_terminal_case
    observation = stage["details"]["deferred_build_readiness"]["blocked_jobs"][0]
    if change == "text_only": observation.update(status="failed", readiness="blocked", build_evidence={})
    elif change == "runtime_started": observation["runtime_started"] = True
    elif change == "compiler_started": observation["compiler_dispatched"] = True
    elif change == "wrong_full": cohort["rows"][0]["model_sha256"] = "e" * 64
    elif change == "wrong_setup": observation["build_evidence"]["context"]["setup_id"] = "foreign"
    elif change == "wrong_boundary": observation["boundary"] = "b999"
    elif change == "duplicate": stage["details"]["deferred_build_readiness"]["blocked_jobs"].append(copy.deepcopy(observation))
    elif change == "wrong_current_cut": cohort["rows"][0]["cut_tensors"] = ["wrong"]
    elif change == "wrong_record": observation["build_evidence"]["record"]["record_sha256"] = "0" * 64
    _write(path, stage)
    assert plan.collect_technical_terminals(cohort, root) == []
    assert plan.propose_reserve_activation(cohort, [{"case_key": "exact-primary", "status": "accuracy_loss"}])["proposals"] == []


def test_terminal_reader_rejects_other_run_and_duplicate_proposals(exact_terminal_case, tmp_path):
    cohort, root, _, _ = exact_terminal_case
    with pytest.raises(ValueError, match="run mismatch"):
        plan.collect_technical_terminals(cohort, tmp_path / "other")
    rows = plan.collect_technical_terminals(cohort, root)
    with pytest.raises(ValueError, match="duplicate technical terminal"):
        plan.propose_reserve_activation(cohort, rows + rows)


@pytest.fixture
def cold_terminal_case(exact_terminal_case, monkeypatch, tmp_path):
    from types import SimpleNamespace
    from onnx_splitpoint_tool.build_evidence import canonical_build_key_from_hailo_v3_payload
    from onnx_splitpoint_tool.build_evidence_store import BuildEvidenceStore
    from onnx_splitpoint_tool.hailo_attempt_receipts import begin_hailo_attempt, finalize_hailo_attempt
    from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService
    cohort, root, stage_path, stage = exact_terminal_case
    row = cohort['rows'][0]
    obs = stage['details']['deferred_build_readiness']['blocked_jobs'][0]
    prior = obs['build_evidence']
    case = root/'models'/row['model_id']/'benchmark_set/legacy_suite'/row['case_id']
    arch = 'hailo10' if row['backend'] == 'hailo10h' else row['backend']
    output = case/'hailo'/arch/'part1'; output.mkdir(parents=True)
    source = case/'part1.onnx'; source.write_bytes(b'fixture exact current P1')
    context = copy.deepcopy(prior['context'])
    key = canonical_build_key_from_hailo_v3_payload(prior['cache_payload_v3'],
        builder_source_onnx_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        full_source_onnx_sha256=context['full_source_onnx_sha256'],
        boundary_endpoint_contract_sha256=prior['key']['boundary_endpoint_contract_sha256'],
        expected_cache_key=prior['cache_key_v3'])
    monkeypatch.setenv('ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT',str(tmp_path/'fixture-evidence'))
    record = BuildEvidenceStore().record(key,'COMPILE_INFEASIBLE',evidence_origin={
        'source':'live_hailo_build','phase':'mapping','outdir':str(output),'net_name':key['recipe']['net_name']})
    info = {**prior,'status':'MISS','negative_evidence_hit':False,'compiler_dispatch_allowed':True,
        'key':key,'recorded':True,'recorded_state':'COMPILE_INFEASIBLE','record_sha256':record['record_sha256']}
    info.pop('record',None)
    args = {'outdir':str(output),'net_name':key['recipe']['net_name'],'hw_arch':row['backend'],
        'build_evidence_context':context}
    result = SimpleNamespace(ok=False, skipped=False,timed_out=False,cache_hit=False,
        details={'build_evidence':info,'compiler_dispatch_count':1},calib_info={},
        last_stage='mapping',error='Mapping failed',failure_kind='allocator_mapping_failed',returncode=1)
    summary = BenchmarkGenerationService().compact_hailo_build_summary(result)
    summary['source_onnx_path']=str(source)
    manifest_path=case/'split_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest.setdefault('hailo',{}).setdefault('hefs',{}).setdefault(arch,{})['part1_build']=summary
    _write(manifest_path,manifest)
    request_path=output/'deferred_hailo_build.json'
    _write(request_path,{'source_onnx':str(source),'kwargs':args,'status':'failed','build_observation':{'compiler_dispatch_count':1}})
    obs.update(status='failed',readiness='blocked',compiler_dispatch_count=1,compiler_dispatched=True,
        build_evidence={},request=str(request_path),upstream_evidence_path=str(request_path))
    _write(stage_path,stage)
    attempt=begin_hailo_attempt(outdir=output,bound={**args,'onnx_path':str(source)})
    finalize_hailo_attempt(attempt=attempt,result=result)
    return cohort,root,stage_path,stage,manifest_path,manifest


def test_new_cold_compile_failure_preserves_real_dispatch_and_failure(cold_terminal_case):
    cohort,root,path,stage,manifest_path,_=cold_terminal_case
    before=(path.read_bytes(),manifest_path.read_bytes())
    rows=plan.collect_technical_terminals(cohort,root)
    assert len(rows)==1
    row=rows[0]
    assert row['status']=='failed' and row['technical_terminal_kind']=='actual_compile_failed'
    assert row['compiler_dispatched'] is True and row['compiler_dispatch_count']==1
    assert row['runtime_started'] is False and row['measurement_started'] is False
    assert row['repetition_count_attempted']==row['repetition_count_valid']==0
    assert 'build_exclusion' not in row
    assert row['actual_compile_failure']['original_recorded_build_evidence']['compiler_dispatch_allowed'] is True
    assert row['upstream_build_observation']==stage['details']['deferred_build_readiness']['blocked_jobs'][0]
    assert (path.read_bytes(),manifest_path.read_bytes())==before


@pytest.mark.parametrize('change',['model','setup','key','record','phase','count','timeout','source','attempt','transient'])
def test_cold_terminal_refuses_binding_and_terminal_evidence_drift(cold_terminal_case,change):
    cohort,root,path,stage,manifest_path,manifest=cold_terminal_case
    obs=stage['details']['deferred_build_readiness']['blocked_jobs'][0]
    hefs=manifest['hailo']['hefs'];summary=next(v['part1_build'] for v in hefs.values())
    info=summary['build_evidence']
    if change=='model':info['context']['model_id']='foreign'
    elif change=='setup':info['context']['setup_id']='foreign'
    elif change=='key':info['key']['full_source_onnx_sha256']='e'*64
    elif change=='record':info['record_sha256']='e'*64
    elif change=='phase':
        terminal=Path(obs['request']).parent/'hailo_attempt_receipts/terminal_attempt.json'
        d=json.loads(terminal.read_text());d['compiler_phase']='cache_lookup';_write(terminal,d)
    elif change=='count':obs['compiler_dispatch_count']=0
    elif change=='timeout':summary['timed_out']=True
    elif change=='source':Path(summary['source_onnx_path']).write_bytes(b'different P1')
    elif change=='attempt':
        terminal=Path(obs['request']).parent/'hailo_attempt_receipts/terminal_attempt.json'
        d=json.loads(terminal.read_text());d['terminal']=False;_write(terminal,d)
    elif change=='transient':info['recorded_state']='TRANSIENT_INFRASTRUCTURE'
    _write(path,stage);_write(manifest_path,manifest)
    assert plan.collect_technical_terminals(cohort,root)==[]
