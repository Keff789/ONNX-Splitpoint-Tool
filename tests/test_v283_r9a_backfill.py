import copy
import json
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
from onnx import helper, TensorProto

from onnx_splitpoint_tool.backend_backfill import DEFAULT_BACKFILL, BackendBackfill, bind_plan_cases, backfill_policy
from onnx_splitpoint_tool.benchmark.services import (BenchmarkGenerationRuntime, BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionCallbacks, BenchmarkGenerationExecutionService)
from onnx_splitpoint_tool.core_analysis import analyze_model
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.runner import expected_profile_measurements_v60r


def negative():
    return {'status': 'HIT', 'state': 'COMPILE_INFEASIBLE', 'reusable': True,
            'negative_evidence_hit': True, 'reason': 'exact_deterministic_outcome'}


@pytest.mark.parametrize('accelerator_scope', [True, False])
def test_real_split_generator_backend_selection_and_matrix(tmp_path, accelerator_scope):
    source = tmp_path/'model.onnx'
    nodes = [helper.make_node('Identity', ['x' if i == 0 else f'v{i}'], [f'v{i+1}']) for i in range(6)]
    graph = helper.make_graph(nodes, 'chain', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 4, 4])],
                              [helper.make_tensor_value_info('v6', TensorProto.FLOAT, [1, 3, 4, 4])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)]), source)
    analysis = analyze_model(str(source)); suite = tmp_path/'suite'; suite.mkdir()
    runs = [{'id': f'{b}_to_tensorrt', 'stage1': {'provider':b}, 'stage2': {'provider':'tensorrt'}} for b in ('hailo8', 'hailo10', 'deepx_m1')]
    runtime = BenchmarkGenerationRuntime(suite, suite/'log', suite/'generation_state.json', 1, [2], [2, 3, 4], 'model', str(source), 'end')
    calls = []
    def physical_leaf(path, **kw):
        b = kw['build_evidence_context']['boundary']; arch = kw['hw_arch']
        assert kw['cache_only'] is True  # This test supplies warm/negative leaves, never compiler PASS.
        calls.append((b, arch))
        if b == 2 and arch == 'hailo8':
            return SimpleNamespace(ok=False, skipped=True, timed_out=False, hef_path=None, failure_kind='known_negative_build_evidence',
                unsupported_reason='COMPILE_INFEASIBLE', error='exact negative', elapsed_s=0,
                details={'build_evidence': negative()}, calib_info={})
        artifact = Path(kw['outdir'])/'part1.hef'; artifact.parent.mkdir(parents=True, exist_ok=True); artifact.write_bytes(b'controlled cached artifact')
        return SimpleNamespace(ok=True, skipped=False, timed_out=False, hef_path=artifact, failure_kind='', error='', elapsed_s=0,
                               details={'cache_hit': True}, calib_info={})
    if not accelerator_scope: runs = []
    cfg = BenchmarkGenerationExecutionConfig(runtime=runtime, target_cases=1, gap=0, ranked_candidates=[2], candidate_search_pool=[2, 3, 4],
        out_dir=suite, base='model', pad=3, strict_boundary=False, model=analysis['model'], nodes=analysis['nodes'], order=analysis['order'],
        analysis_payload=analysis, full_model_src=str(source), require_single_part2_input=True, hef_targets=['hailo8', 'hailo10h'] if accelerator_scope else [],
        hef_part1=True, bench_plan_runs=runs, hailo_build_hef_fn=physical_leaf, backend_backfill_policy=DEFAULT_BACKFILL)
    cb = BenchmarkGenerationExecutionCallbacks(log=lambda *a, **k: None, queue_put=lambda *a: None, persist_state=runtime.persist,
        publish_hailo_diagnostics=lambda *a, **k: None, predicted_metrics_for_boundary=lambda *a: {}, hailo_parse_entry_for_boundary=lambda *a: None,
        hailo_parse_scalar_fields=lambda *a: {})
    selected = BenchmarkGenerationExecutionService().execute_case_build_loop(cfg, cb)
    if not accelerator_scope:
        assert selected == [2] and len(runtime.cases) == 1
        assert not runtime.generation_state.get('backend_backfill')
        assert not calls
        return
    assert selected == [2, 3]
    state = json.loads(runtime.state_path.read_text())['backend_backfill']
    assert {c['backend']: c['selected_case_ids'] for c in state['contracts']} == {'hailo8': ['b003'], 'hailo10h': ['b002'], 'deepx': ['b002']}
    assert (3, 'hailo10h') not in calls and not any(b == 4 for b, _ in calls)
    assert state['cold_builds_started'] == 0
    assert state['audit_case_count'] == 1
    plan = {'runs': runs + [{'id': 'hailo8_full'}, {'id': 'ort_tensorrt', 'variants': ['full']}]}
    bind_plan_cases(plan, state)
    matrix = expected_profile_measurements_v60r(model_id='model', benchmark_plan=plan, benchmark_set_contract={'cases': runtime.cases})
    assert len([r for r in matrix if r['variant'] == 'split']) == 3
    assert len([r for r in matrix if r['variant'] == 'full']) == 2


def controller(policy=None):
    state = {}; persisted = []
    kwargs = dict(state=state, policy=policy or DEFAULT_BACKFILL, pool=[2,3,4], initial=[2], quota=1,
                  contracts=[{'id':'h8', 'backend':'hailo8', 'run_id':'hailo8_to_trt', 'stage':'part1'}],
                  persist=lambda: persisted.append(copy.deepcopy(state)))
    return BackendBackfill(**kwargs), kwargs, persisted


def test_pool_resume_does_not_reset_consumption():
    ctl, kwargs, _ = controller({**DEFAULT_BACKFILL, 'max_candidates_per_backend': 2})
    for b in (2, 3): ctl.observe(b, ctl.begin(b), {'hailo8': {'part1_build': {'build_evidence': negative()}}})
    resumed = BackendBackfill(**kwargs)
    assert resumed.begin(4) == []
    assert resumed.state['contracts'][0]['status'] == 'pool_exhausted'
    with pytest.raises(ValueError, match='resume_contract_changed'):
        BackendBackfill(**{**kwargs, 'pool':[2,4,3]})


@pytest.mark.parametrize('failure', ['quality_FAIL','timeout','unknown','INCONCLUSIVE'])
def test_only_exact_build_exclusions_admit_replacement(failure):
    ctl, _, _ = controller()
    ctl.observe(2, ctl.begin(2), {'hailo8': {'part1_build': {'error': failure}}})
    assert ctl.active() == []
    assert ctl.state['contracts'][0]['status'] == 'infrastructure_blocked'
    assert ctl.state['contracts'][0]['selected_case_ids'] == ['b002']


def test_config_defaults_disabled_and_frozen_history():
    config = default_run_modes_config()
    for mode in ['standard', 'final']:
        resolved, _ = apply_run_mode({'name':'r9a'}, mode_id=mode, config=config)
        assert backfill_policy(resolved) == DEFAULT_BACKFILL
        disabled, _ = apply_run_mode({'name':'r9a', 'selection_policy': {'backend_backfill': {'enabled':False}}}, mode_id=mode, config=config)
        assert backfill_policy(disabled) == {}
    old, _ = apply_run_mode({'name':'old', 'execution_preset': {'id':'standard', 'follow_tool_config':False,
                                                            'snapshot': config['modes']['standard']}}, config=config)
    assert backfill_policy(old) == {}


def test_next_miss_is_dispatched_before_later_hit_and_consumed_on_failure():
    ctl, kwargs, persisted = controller({**DEFAULT_BACKFILL, 'max_cold_builds': 1, 'max_hailo_part1_builds': 1})
    calls = []
    def compiler_leaf(source, **options):
        calls.append((str(source), options['cache_only']))
        if options['cache_only']:
            return SimpleNamespace(ok=False, skipped=True, failure_kind='cache_miss', error='cache miss', details={})
        assert persisted[-1]['cold_builds_started'] == 1
        return SimpleNamespace(ok=False, skipped=False, failure_kind='timeout', error='timeout', details={})
    dispatch = ctl.bind_builder(compiler_leaf)
    options = {'cache_only': False, 'hw_arch':'hailo8', 'build_evidence_context': {'stage':'part1', 'boundary':3}}
    result = dispatch('next_missing', **options)
    assert result.failure_kind == 'timeout'
    assert calls == [('next_missing', True), ('next_missing', False), ('next_missing', True)]
    resumed = BackendBackfill(**kwargs)
    result = resumed.bind_builder(compiler_leaf)('next_missing', **options)
    assert result.failure_kind == 'build_budget_exhausted'
    assert sum(not cache_only for _, cache_only in calls) == 1


def test_persistence_failure_prevents_cold_dispatch():
    ctl, _, _ = controller()
    calls = []
    def compiler_leaf(source, **options):
        calls.append(options['cache_only'])
        return SimpleNamespace(ok=False, skipped=True, failure_kind='cache_miss', error='cache miss', details={})
    def cannot_persist(): raise OSError('checkpoint unavailable')
    ctl.persist = cannot_persist
    with pytest.raises(OSError):
        ctl.bind_builder(compiler_leaf)('next', cache_only=False, hw_arch='hailo8', build_evidence_context={'stage':'part1'})
    assert calls == [True]


def test_trt_actual_process_start_budget_survives_failure_and_new_call(tmp_path):
    import sys
    from scripts.native_trt_from_benchmarkset import _run
    checkpoint = tmp_path/'native_build_state.json'
    kwargs = dict(cwd=tmp_path, log_path=tmp_path/'builder.log', dry_run=False,
                  build_state=str(checkpoint), max_build_starts=1, timeout_s=2, artifact_role='part2', case_id='b003')
    first = _run([sys.executable, '-c', 'raise SystemExit(17)'], **kwargs)
    assert first['returncode'] == 17
    second = _run([sys.executable, '-c', 'raise SystemExit(99)'], **kwargs)
    assert second['status'] == 'build_budget_exhausted'
    assert second['compiler_dispatched'] is False
    assert len(json.loads(checkpoint.read_text())['starts']) == 1


def test_trt_inner_process_timeout_is_bounded_and_consumes_start(tmp_path):
    import sys
    from scripts.native_trt_from_benchmarkset import _run
    checkpoint = tmp_path/'native_build_state.json'
    result = _run([sys.executable, '-c', 'import time;time.sleep(30)'], cwd=tmp_path,
                  log_path=tmp_path/'log', dry_run=False, build_state=str(checkpoint), max_build_starts=1,
                  timeout_s=1, artifact_role='part2', case_id='b003')
    assert result['returncode'] == 124
    assert result['elapsed_s'] < 5
    assert len(json.loads(checkpoint.read_text())['starts']) == 1


def test_interrupted_candidate_is_replayed_with_original_case_padding():
    ctl, kwargs, _ = controller()
    assert ctl.begin(2)
    resumed = BackendBackfill(**kwargs)
    active = resumed.begin(2)
    assert active and active[0]['considered'] == [2]
    resumed.observe(2, active, {'hailo8': {'part1_build': {'build_evidence': negative()}}}, case_id='b0002')
    assert BackendBackfill(**kwargs).begin(2) == []


def test_scoped_matrix_preserves_setup_cases_and_full_references():
    from onnx_splitpoint_tool.backend_backfill import selection_contracts, selected_cases_for_backend
    descriptors = [{'model_id':'model', 'run_id':'hailo8_to_tensorrt', 'backend':'hailo8_to_tensorrt',
                    'variant':'split', 'expected_setup_id': setup, 'measurement_endpoint':'completed_task'}
                   for setup in ('a','b')]
    runs = [{'id':'hailo8_to_tensorrt', 'stage1':'hailo8', 'stage2':'tensorrt', 'physical_identity_descriptors':descriptors}]
    definitions = selection_contracts(runs)
    assert len(definitions) == 2
    state = {'enabled':True, 'policy':DEFAULT_BACKFILL, 'cold_builds_started':0,
             'contracts':[dict(c, selected_case_ids=[case]) for c,case in zip(definitions,['b002','b003'])]}
    plan = {'runs':runs}; bind_plan_cases(plan,state)
    matrix = expected_profile_measurements_v60r(model_id='model', benchmark_plan=plan,
                                               benchmark_set_contract={'cases':[{'case_id':'b002'},{'case_id':'b003'}]})
    assert {(r['setup_id'],r['case_id']) for r in matrix} == {('a','b002'),('b','b003')}
    assert selected_cases_for_backend(state,'hailo8',['b003'],setup_id='a') == []
    assert sum(plan['backend_backfill_build_budget']['trt_starts_by_setup'].values()) <= DEFAULT_BACKFILL['max_cold_builds']


def test_deepx_physical_dispatch_shares_saved_split_budget(tmp_path):
    from onnx_splitpoint_tool.backend_backfill import call_with_build_budget, reserve_active_build
    ctl, _, _ = controller({**DEFAULT_BACKFILL,'max_cold_builds':1})
    path = tmp_path/'generation_state.json'; path.write_text(json.dumps({'backend_backfill':ctl.state}))
    dispatched=[]
    def leaf():
        reserve_active_build(['controlled-dx-com'])
        dispatched.append(1)
    call_with_build_budget(leaf,state_path=path,stage='part1')
    with pytest.raises(RuntimeError,match='build_budget_exhausted:deepx'):
        call_with_build_budget(leaf,state_path=path,stage='part1')
    assert dispatched == [1]
    assert json.loads(path.read_text())['backend_backfill']['cold_builds_started'] == 1


def test_remote_suite_replacement_preserves_and_collects_build_consumption(tmp_path):
    import hashlib, subprocess, tarfile
    from onnx_splitpoint_tool.benchmark.remote_run import _verified_uncached_suite_extract_command, _remote_result_collect_script
    suite=tmp_path/'suite';suite.mkdir()
    state={'max_build_starts':1,'starts':[{'index':1}]}
    (suite/'native_trt_build_state.json').write_text(json.dumps(state))
    bundle=tmp_path/'suite.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        incoming=tmp_path/'benchmark_plan.json';incoming.write_text('{}')
        archive.add(incoming,arcname=incoming.name)
    command=_verified_uncached_suite_extract_command(remote_bundle=str(bundle), remote_suite_dir=str(suite),
                                                     bundle_hash=hashlib.sha256(bundle.read_bytes()).hexdigest())
    subprocess.run(['bash','-c',command],check=True)
    assert json.loads((suite/'native_trt_build_state.json').read_text()) == state
    assert not bundle.exists()


def test_execution_filters_respect_setup_and_direction():
    from onnx_splitpoint_tool.backend_backfill import selected_cases_for_backend
    contracts=[{'backend':'hailo8','setup_id':setup,'run_id':direction,'selected_case_ids':[case]}
               for setup,direction,case in [('h8_a','hailo8_to_tensorrt','b002'),
                 ('h8_b','hailo8_to_tensorrt','b003'),('h8_a','trt_to_hailo8','b004')]]
    state={'enabled':True,'contracts':contracts}
    assert selected_cases_for_backend(state,'hailo8',['b002','b003','b004'],setup_id='h8_a',run_id='hailo8_to_trt')==['b002']
    # Execute the actual exported template helper, with its normal ID resolver.
    import ast
    from typing import List, Dict, Any
    source=(Path(__file__).resolve().parents[1]/'onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt').read_text()
    tree=ast.parse(source)
    names={'_cases_for_run','_case_identifiers','_canonical_plan_run_id','_normalize_variants'}
    module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    import typing
    ns=dict(vars(typing))
    exec(compile(module,'actual_benchmark_suite_template','exec'),ns)
    cases=[{'folder':b,'case_dir':b} for b in ['b002','b003','b004']]
    run={'id':'hailo8_to_trt','case_ids':['b002','b003','b004'],'backend_selection_contracts':contracts}
    assert ns['_cases_for_run'](cases,run,setup_id='h8_a')==cases[:1]
    assert ns['_cases_for_run'](cases,run,setup_id='unknown')==[]
    with pytest.raises(ValueError,match='setup_identity_missing'):ns['_cases_for_run'](cases,run)


def test_trt_model_limit_is_shared_across_setup_copies():
    state={'enabled':True,'policy':{**DEFAULT_BACKFILL,'max_cold_builds':8,'max_trt_part2_builds':1},
           'cold_builds_started':0,'contracts':[{'stage':'part1','setup_id':setup,'selected_case_ids':['b002'],'run_id':'hailo8_to_trt'} for setup in ['a','b','c']]}
    plan={'runs':[]};bind_plan_cases(plan,state)
    assert plan['backend_backfill_build_budget']['trt_starts_by_setup']=={'a':1,'b':0,'c':0}


def test_real_workflow_resume_is_distinct_from_remote_reuse_permission():
    from onnx_splitpoint_tool.workflow.execution_binding import _remote_args_from_options
    first=_remote_args_from_options(SimpleNamespace(resume=False),{})
    resumed=_remote_args_from_options(SimpleNamespace(resume=True),{})
    assert first.resume is True and first.native_build_budget_resume is False
    assert resumed.resume is True and resumed.native_build_budget_resume is True


def test_normal_mode_preserves_separate_diagnostic_inference_and_transport_limits():
    config=default_run_modes_config()
    source={'name':'r9a_limits','execution_preset':{'id':'standard','follow_tool_config':False,'snapshot':config['modes']['standard']},
            'benchmark_execution':{'extra_args':['--timeout','120']},'remote_execution':{'timeout_s':900}}
    source['execution_preset']['snapshot']['runtime']['benchmark']['timeout_s']=120
    resolved,_=apply_run_mode(source,config=config)
    from onnx_splitpoint_tool.workflow.execution_binding import _remote_args_from_options
    args=_remote_args_from_options(SimpleNamespace(benchmark_extra_args=resolved['benchmark_execution']['extra_args']),resolved)
    assert resolved['benchmark_execution']['timeout_s']==120
    assert args.timeout_s==900 and args.add_args=='--timeout 120'
    assert not resolved['remote_execution'].get('host')


@pytest.mark.parametrize('resume', [False,True])
def test_real_trt_builder_first_start_and_missing_resume_checkpoint(tmp_path,monkeypatch,resume):
    import os,sys
    from onnx_splitpoint_tool.runners import native_split_quality_runtime as runtime
    from test_v27536_quality_first_real_builder_reuse import _write_source_part2
    suite=tmp_path/'suite';case=suite/'b038';part1=case/'hailo/hailo8/part1/compiled.hef'
    part1.parent.mkdir(parents=True);part1.write_bytes(b'controlled Hailo metadata fixture')
    _write_source_part2(case/'yolo26s_part2_native.onnx')
    monkeypatch.setattr(runtime,'_hailo_metadata',lambda **kw:{'name':'cut','runtime_name':'cut/hailort','shape':[2,2,2],
       'canonical_part2_shape':[1,2,2,2],'dtype':'uint8','quantization':{'source':'hailort_hef_output_vstream_info','scale':0.03125,'zero_point':17.0}})
    compiler=tmp_path/'bin/trtexec';compiler.parent.mkdir()
    log=tmp_path/'compiler_starts.txt'
    compiler.write_text(f'#!{sys.executable}\nimport sys\nfrom pathlib import Path\nif "--help" in sys.argv:\n print("--memPoolSize");sys.exit(0)\nif any(a.startswith("--saveEngine=") for a in sys.argv):\n Path({str(log)!r}).open("a").write("start\\n")\nsys.exit(17)\n')
    compiler.chmod(0o755);monkeypatch.setenv('PATH',str(compiler.parent)+os.pathsep+os.environ['PATH'])
    monkeypatch.delenv('ONNX_SPLITPOINT_ARTIFACT_POLICY',raising=False)
    policy={**DEFAULT_BACKFILL,'remaining_cold_builds':1,'max_trt_part2_builds':1,'trt_build_timeout_s':30,'trt_starts_by_setup':{'h8':1}}
    with pytest.raises(RuntimeError):
        runtime.prepare_native_split_quality_binding(benchmark_set=suite,case_id='b038',model_id='yolo26s',setup_id='h8',backend='hailo8_to_trt',
          eval_run_id='r9a',source_run_id='hailo8_to_trt',cache_root=tmp_path/'cache',output_path=tmp_path/'binding.json',workspace_mb=64,timeout_s=30,
          build_budget=policy,resume=resume)
    checkpoint=suite/'native_trt_build_state.json'
    if resume:
        assert not log.exists() and not checkpoint.exists()
    else:
        assert log.read_text().splitlines()==['start']
        assert len(json.loads(checkpoint.read_text())['starts'])==1


@pytest.mark.parametrize('dispatch_count,expected', [(0,'HIT'),(1,'MISS'),(None,'MISS'),(False,'MISS')])
def test_cache_bilanz_uses_physical_dispatch_count_with_real_receipt(tmp_path,dispatch_count,expected):
    _check_cache_bilanz_receipt(tmp_path,dispatch_count,expected)


@pytest.mark.parametrize('dispatch_count,skipped,corrupt_receipt,expected', [
    (1,True,False,'MISS'), (-1,False,False,'MISS'), (0.0,False,False,'MISS'),
    ('0',False,False,'MISS'), (0,False,True,'HIT'), (1,True,True,'MISS'),
])
def test_cache_bilanz_counter_conflict_and_receipt_gate(tmp_path,dispatch_count,skipped,corrupt_receipt,expected):
    _check_cache_bilanz_receipt(tmp_path,dispatch_count,expected,skipped,corrupt_receipt)


def _check_cache_bilanz_receipt(tmp_path,dispatch_count,expected,contradictory_skip=False,corrupt_receipt=False):
    from onnx_splitpoint_tool import hailo_backend as backend
    from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract,preprocessing_contract_sha256
    from onnx_splitpoint_tool.workflow.artifact_cache_preflight import _hailo_observations,resolve_artifact_cache_preflight_policy
    source=tmp_path/'source.onnx';source.write_bytes(b'controlled cached source')
    contract=canonical_image_preprocessing_contract('classification',(224,224))
    key,payload=backend._hailo_cache_key(model_path=source,activation_part1=None,hw_arch='hailo8',opt_level=1,
        calib_dir=None,calib_count=1,calib_batch_size=1,extra_model_script='',start_nodes=None,end_nodes=None,
        preprocessing_contract=contract,effective_calib_count=1,calibration_storage='memory',calibration_memory_cap_bytes=64*1024*1024,
        net_name='model_full',net_input_shapes={'images':[1,3,224,224]},disable_rt_metadata_extraction=True)
    hef=tmp_path/'cache/compiled.hef';hef.parent.mkdir();hef.write_bytes(b'controlled already cached HEF')
    backend._write_hailo_receipt(hef_path=hef,source_onnx=source,compiler_onnx=source,hw_arch='hailo8',net_name='model_full',
        preprocessing_contract=contract,preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=key,cache_payload=payload,calibration_identity=payload['calibration_identity'],calibration_count=1)
    assert backend._load_valid_hailo_receipt(hef) is not None
    if corrupt_receipt:
        hef.write_bytes(b'tampered artifact')
    attempt={'ok':True,'skipped':contradictory_skip}
    if dispatch_count is not None:attempt['compiler_dispatch_count']=dispatch_count
    path=tmp_path/'models/model/benchmark_set/hailo_artifact_service_plan.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'full_baseline_requests':[{'backend':'hailo8','requested':True,'status':'ready_existing_hef',
       'hef_path':str(hef),'build_attempt':attempt}]}))
    rows,_=_hailo_observations(run_root=tmp_path,model_id='model',policy=resolve_artifact_cache_preflight_policy({}))
    assert rows[0].status == ('MISS' if corrupt_receipt else expected)
    assert rows[0].evidence['current_build'] is (expected=='MISS')
