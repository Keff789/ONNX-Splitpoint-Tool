import copy
import json
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
from onnx import helper as h, TensorProto as T

from onnx_splitpoint_tool.native_split_quality import detection_score_graph_contract, probability_quantization_suitability
from onnx_splitpoint_tool.backend_backfill import BackendBackfill, DEFAULT_BACKFILL, backfill_policy, bind_plan_cases


def graphs(classes=7, positions=11, prefix='renamed', axis=1):
    shape = [1, classes + 4, positions] if axis == 1 else [1, positions, classes + 4]
    scores = [1, classes, positions] if axis == 1 else [1, positions, classes]
    boxes = [1, 4, positions] if axis == 1 else [1, positions, 4]
    def vi(name, dims): return h.make_tensor_value_info(name, T.FLOAT, dims)
    p1 = h.make_model(h.make_graph([
        h.make_node('Sigmoid', ['logits'], ['prob']), h.make_node('Concat', ['boxes', 'prob'], [prefix], axis=axis)
    ], 'producer', [vi('boxes', boxes), vi('logits', scores)], [vi(prefix, shape)]), opset_imports=[h.make_opsetid('', 13)])
    p2 = h.make_model(h.make_graph([
        h.make_node('Split', [prefix, 'widths'], ['coords', 'scores'], axis=axis),
        h.make_node('ReduceMax', ['scores'], ['maximum'], axes=[axis]),
        h.make_node('TopK', ['maximum', 'k'], ['ranked', 'indices'], axis=2 if axis == 1 else 1),
        h.make_node('TopK', ['scores', 'k'], ['out', 'classes'], axis=axis),
    ], 'consumer', [vi(prefix, shape)], [vi('out', [1, 1, positions] if axis == 1 else [1, positions, 1])],
       [h.make_tensor('widths', T.INT64, [2], [4, classes]), h.make_tensor('k', T.INT64, [1], [1])]), opset_imports=[h.make_opsetid('', 13)])
    return p1, p2


def contract(classes=7, positions=11, axis=1, scale=3.2, zp=35):
    p1, p2 = graphs(classes, positions, axis=axis)
    sem = detection_score_graph_contract(p1, p2, task='detection')
    assert sem['status'] == 'PROVEN'
    tensor = dict(canonical_part2_shape=sem['shape'], hef_native_storage_dtype='uint8', dtype='uint8',
        quantization={'source': 'hailort_hef_output_vstream_info', 'scale': scale, 'zero_point': zp, 'rounding': 'nearest_unspecified_ties'})
    return sem, tensor


@pytest.mark.parametrize('classes,positions,axis', [(1, 3, 1), (7, 11, 2), (80, 8400, 1), (13, 5, 1)])
def test_names_sizes_layout_and_archived_contracts(classes, positions, axis):
    rows = json.loads((Path(__file__).parent/'fixtures/v283_r9f_contracts.json').read_text())['contracts']
    for row in rows:
        q = row['native_output']['quant_info']
        sem, tensor = contract(classes, positions, axis, q['qp_scale'], q['qp_zp'])
        assert probability_quantization_suitability(sem, tensor)['status'] == 'INCOMPATIBLE'
        tensor['dtype'] = 'float32'
        assert probability_quantization_suitability(sem, tensor)['status'] == 'INCOMPATIBLE'


@pytest.mark.parametrize('scale,zp,rounding,status', [
    (3.2,35,'nearest_unspecified_ties','INCOMPATIBLE'), (3.2,35,'unknown','UNKNOWN'),
    (2,34,'nearest_unspecified_ties','UNKNOWN'), (2,34,'nearest_even','INCOMPATIBLE'),
    (2,35,'nearest_unspecified_ties','UNKNOWN'),
    (2,35,'nearest_even','NOT_COLLAPSED'), (1.9999,35,'nearest_unspecified_ties','NOT_COLLAPSED'),
    (.01,0,'unknown','NOT_COLLAPSED'), (3.2,35,'ceil','NOT_COLLAPSED'),
    (0,35,'nearest_even','UNKNOWN'), (-1,35,'nearest_even','UNKNOWN'),
    (float('nan'),35,'nearest_even','UNKNOWN'), (float('inf'),35,'nearest_even','UNKNOWN'),
    (3.2,-1,'nearest_even','UNKNOWN'), (3.2,256,'nearest_even','UNKNOWN'), (3.2,35.5,'nearest_even','UNKNOWN'),
])
def test_range_rounding_and_invalid_quantinfo(scale,zp,rounding,status):
    sem,tensor=contract(scale=scale,zp=zp);tensor['quantization']['rounding']=rounding
    assert probability_quantization_suitability(sem,tensor)['status']==status


@pytest.mark.parametrize('mutation', ['semantics','native_dtype','source','shape','channels','scale_missing'])
def test_unknown_metadata_never_rejects(mutation):
    sem,tensor=contract()
    if mutation=='semantics':sem['score_semantics']='logits'
    if mutation=='native_dtype':tensor.pop('hef_native_storage_dtype')
    if mutation=='source':tensor['quantization']['source']='unbound_json'
    if mutation=='shape':tensor['canonical_part2_shape']=[1,12,11]
    if mutation=='channels':sem['score_channels']=[100]
    if mutation=='scale_missing':tensor['quantization'].pop('scale')
    assert probability_quantization_suitability(sem,tensor)['status']=='UNKNOWN'


def test_per_channel_only_scores_count_and_axis_is_required():
    sem,tensor=contract(classes=3)
    q=tensor['quantization'];q.update(scale=[.001]*4+[3.2]*3,zero_point=[35]*7,canonical_channel_axis=1)
    assert probability_quantization_suitability(sem,tensor)['status']=='INCOMPATIBLE'
    q['scale'][5]=.1
    assert probability_quantization_suitability(sem,tensor)['status']=='NOT_COLLAPSED'
    q['canonical_channel_axis']=2
    assert probability_quantization_suitability(sem,tensor)['status']=='UNKNOWN'
    q['canonical_channel_axis']=1;q['scale'].pop()
    assert probability_quantization_suitability(sem,tensor)['status']=='UNKNOWN'


@pytest.mark.parametrize('change',['featuremap','logits','wrong_split','wrong_reduce','classification','index_only','custom_domain','unused_ranking'])
def test_graph_semantic_negatives(change):
    p1,p2=graphs();task='detection'
    if change=='featuremap':p2.graph.node[0].op_type='Conv'
    if change=='logits':p1.graph.node[0].op_type='Identity'
    if change=='wrong_split':p2.graph.initializer[0].int64_data[:]=[5,6]
    if change=='wrong_reduce':p2.graph.node[1].attribute[0].ints[:]=[2]
    if change=='classification':task='classification'
    if change=='index_only':p2.graph.output[0].name='indices'
    if change=='custom_domain':p1.graph.node[0].domain='custom.probability'
    if change=='unused_ranking':p2.graph.node[-1].CopyFrom(h.make_node('Identity',['scores'],['out']))
    assert detection_score_graph_contract(p1,p2,task=task)['status']=='UNKNOWN'


def controller(policy=None):
    state={};persisted=[]
    contracts=[dict(id=b,backend=b,stage='part1',run_id=b+'_to_trt',setup_id=b) for b in ['hailo10h','hailo8','deepx']]
    ctl=BackendBackfill(state=state,policy=policy or DEFAULT_BACKFILL,pool=[2,3,4,5,6],initial=[2],quota=1,
                       contracts=contracts,persist=lambda:persisted.append(copy.deepcopy(state)))
    return ctl,persisted


def evidence():
    sem,tensor=contract();proof=probability_quantization_suitability(sem,tensor)
    return {**proof,'binding_verified':True,'artifact':'exact.hef','semantic':sem,'native_tensor':tensor}


def hefs(proof):
    return {'hailo10h':{'part1':'exact.hef','part1_output_artifact':'exact.hef','part1_build':{'cache_hit':True},'part1_output_suitability':proof},
            'hailo8':{'part1':'h8.hef'},'deepx':{'part1':'model.dxnn'}}


def test_backend_local_exclusion_preserves_build_and_plan_binding():
    ctl,_=controller();rows=hefs(evidence());ctl.observe(2,ctl.begin(2),rows)
    assert [c['backend'] for c in ctl.active()]==['hailo10h']
    ctl.observe(3,ctl.begin(3),hefs({'status':'NOT_COLLAPSED'}));ctl.finish()
    assert {c['backend']:c['selected_case_ids'] for c in ctl.state['contracts']}=={'hailo10h':['b003'],'hailo8':['b002'],'deepx':['b002']}
    row=ctl.state['audit_cases'][0]
    assert row['cache_hit'] and row['build_status']=='BUILT' and not row['build_evidence']
    plan={'runs':[{'id':c['run_id']} for c in ctl.state['contracts']]+[{'id':'full','variants':['full']}]}
    bind_plan_cases(plan,ctl.state)
    assert plan['runs'][0]['case_ids']==['b003'] and 'case_ids' not in plan['runs'][-1]


@pytest.mark.parametrize('reason',['AP_FAIL','INCONCLUSIVE','timeout','gpu_error','transport_error','empty_detections','all_zero_image','foreign_receipt'])
def test_nonproof_never_admits_backfill(reason):
    ctl,_=controller();proof=evidence();proof.update(status='UNKNOWN',reason=reason,binding_verified=False)
    ctl.observe(2,ctl.begin(2),hefs(proof))
    assert not ctl.active() and ctl.state['contracts'][0]['status']=='output_contract_unknown'
    assert not ctl.state['contracts'][0]['selected_case_ids']


def test_miss_stops_at_next_ranked_candidate_and_budget_never_wraps():
    ctl,_=controller({**DEFAULT_BACKFILL,'max_candidates_per_backend':4,'max_cold_builds':0})
    ctl.observe(2,ctl.begin(2),hefs(evidence()))
    ctl.observe(3,ctl.begin(3),{'hailo10h':{'part1_build':{'failure_kind':'build_budget_exhausted','error':'MISS/build_required'}}})
    assert ctl.begin(4)==[] and ctl.state['contracts'][0]['considered']==[2,3]
    assert ctl.state['contracts'][0]['status']=='build_budget_exhausted'
    ctl,_=controller({**DEFAULT_BACKFILL,'max_candidates_per_backend':4})
    for b in [2,3,4,5]:ctl.observe(b,ctl.begin(b),hefs(evidence()))
    assert ctl.begin(6)==[] and ctl.state['contracts'][0]['status']=='pool_exhausted'


def test_explicit_historical_policy_does_not_acquire_new_rule():
    old={k:v for k,v in DEFAULT_BACKFILL.items() if k!='technical_output_contract_version'}
    assert backfill_policy({'selection_policy':{'backend_backfill':old}})==old
    ctl,_=controller(old);ctl.observe(2,ctl.begin(2),hefs(evidence()))
    assert ctl.state['contracts'][0]['selected_case_ids']==['b002']
    BackendBackfill(state=ctl.state,policy=backfill_policy({'selection_policy':{'backend_backfill':old}}),
        pool=ctl.state['candidate_order'],initial=ctl.state['initial_selection'],quota=1,
        contracts=ctl.state['contract_definitions'],persist=lambda:None)


def test_built_hef_does_not_inherit_other_candidates_compile_failure():
    ctl,_=controller();rows=hefs({'status':'NOT_COLLAPSED'})
    rows['hailo10h']['part1_build']['build_evidence']={'status':'HIT','state':'COMPILE_INFEASIBLE','reusable':True,'negative_evidence_hit':True,'reason':'exact_deterministic_outcome'}
    ctl.observe(2,ctl.begin(2),rows)
    assert ctl.state['contracts'][0]['selected_case_ids']==['b002']
    assert not ctl.state['audit_cases'][0]['build_evidence']


def test_normal_config_migration_changes_only_policy():
    from onnx_splitpoint_tool.run_modes import default_run_modes_config,validate_run_modes_config,apply_run_mode
    cfg=default_run_modes_config();cfg['schema_version']=14;cfg['backend_backfill'].pop('technical_output_contract_version')
    migrated=validate_run_modes_config(cfg)
    assert migrated['schema_version']==15 and migrated['backend_backfill']['technical_output_contract_version']==1
    assert cfg['modes']==migrated['modes']
    resolved,_=apply_run_mode({'name':'normal','execution_preset':{'id':'standard','follow_tool_config':True}},config=migrated)
    assert backfill_policy(resolved)['technical_output_contract_version']==1


def test_resolver_missing_reader_or_foreign_receipt_is_unknown(tmp_path,monkeypatch):
    from onnx_splitpoint_tool.runners import native_split_quality_runtime as rt
    from onnx_splitpoint_tool import hailo_backend as hb
    p1,p2=graphs();one=tmp_path/'p1.onnx';two=tmp_path/'p2.onnx';onnx.save(p1,one);onnx.save(p2,two)
    monkeypatch.setattr(hb,'_load_valid_hailo_receipt',lambda *a,**k:None)
    result=rt.resolve_hailo_output_suitability(part1=tmp_path/'old.hef',source_part1=one,source_part2=two,task='detection')
    assert result['status']=='UNKNOWN' and result['required'] and result['reason']=='exact_hef_source_receipt_unavailable'


@pytest.mark.parametrize('case',['valid','reader_missing','foreign_hef','foreign_size','foreign_p2','invalid_binding','unknown_sdk','explicit_ceil'])
def test_resolver_bound_metadata_and_reader_negative_matrix(tmp_path,monkeypatch,case):
    from onnx_splitpoint_tool.runners import native_split_quality_runtime as rt
    from onnx_splitpoint_tool import hailo_backend as hb,native_split_quality as nsq
    p1,p2=graphs();one=tmp_path/'p1.onnx';two=tmp_path/'p2.onnx';onnx.save(p1,one);onnx.save(p2,two)
    receipt={'hw_arch':'hailo10h','hef_sha256':'a'*64,'hef_size_bytes':123,'source_onnx_sha256':rt._sha256_file(one),
             'hailo_sdk_version':'hailo-dataflow-compiler:5.3.0'}
    def receipt_reader(path,**kw):
        assert kw['source_onnx_sha256']==rt._sha256_file(one)
        return receipt
    monkeypatch.setattr(hb,'_load_valid_hailo_receipt',receipt_reader)
    _,tensor=contract();tensor['quantization'].pop('rounding')
    bound={'artifacts':{'part1_runtime':{'sha256':'a'*64,'size_bytes':123},'source_part2_onnx':{'sha256':rt._sha256_file(two)}},
           'boundary_metadata_payload':{'boundary_tensor':tensor}}
    # File/crosslink validation remains covered by the existing binding suite;
    # this leaf double isolates the resolver's independent artifact comparison.
    def validate(value,**kw):
        assert kw['verification_mode']=='portable'
        return (None,'corrupt') if case=='invalid_binding' else (value,'')
    monkeypatch.setattr(nsq,'validate_native_split_quality_binding',validate)
    if case=='foreign_hef':bound['artifacts']['part1_runtime']['sha256']='b'*64
    if case=='foreign_size':bound['artifacts']['part1_runtime']['size_bytes']=124
    if case=='foreign_p2':bound['artifacts']['source_part2_onnx']['sha256']='c'*64
    if case=='unknown_sdk':receipt['hailo_sdk_version']='unknown'
    if case=='explicit_ceil':tensor['quantization']['rounding']='ceil'
    if case=='reader_missing':
        def missing(**kw):raise RuntimeError('no HailoRT reader')
        monkeypatch.setattr(rt,'_hailo_metadata',missing);bound=None
    result=rt.resolve_hailo_output_suitability(part1=tmp_path/'part1.hef',source_part1=one,source_part2=two,task='detection',binding=bound)
    assert result['status']==('INCOMPATIBLE' if case=='valid' else 'NOT_COLLAPSED' if case=='explicit_ceil' else 'UNKNOWN')
    assert result['required'] is True


@pytest.mark.parametrize('unknown',[False,True])
def test_real_generator_backendwise_admission_before_runtime(tmp_path,monkeypatch,unknown):
    from onnx_splitpoint_tool.benchmark.services import (BenchmarkGenerationRuntime, BenchmarkGenerationExecutionConfig,
        BenchmarkGenerationExecutionCallbacks, BenchmarkGenerationExecutionService)
    from onnx_splitpoint_tool.core_analysis import analyze_model
    from onnx_splitpoint_tool.runners import native_split_quality_runtime as rt
    source=tmp_path/'model.onnx'
    nodes=[h.make_node('Identity',['x' if i==0 else f'v{i}'],[f'v{i+1}']) for i in range(6)]
    graph=h.make_graph(nodes,'chain',[h.make_tensor_value_info('x',T.FLOAT,[1,3,4,4])],[h.make_tensor_value_info('v6',T.FLOAT,[1,3,4,4])])
    onnx.save(h.make_model(graph,opset_imports=[h.make_opsetid('',13)]),source)
    analysis=analyze_model(str(source));suite=tmp_path/'suite';suite.mkdir()
    runtime=BenchmarkGenerationRuntime(suite,suite/'log',suite/'generation_state.json',1,[2],[2,3,4],'model',str(source),'end')
    calls=[]
    def leaf(path,**kw):
        assert kw['cache_only'] is True
        calls.append((kw['build_evidence_context']['boundary'],kw['hw_arch']))
        artifact=Path(kw['outdir'])/'part1.hef';artifact.parent.mkdir(parents=True,exist_ok=True);artifact.write_bytes(b'controlled cached artifact')
        return SimpleNamespace(ok=True,skipped=False,timed_out=False,hef_path=artifact,failure_kind='',error='',elapsed_s=0,details={'cache_hit':True},calib_info={})
    probes=[]
    def resolver(**kw):
        probes.append(str(kw['source_part1']))
        assert kw['source_part1'].is_file() and kw['source_part2'].is_file() and kw['part1'].is_file()
        if 'b2.' not in str(kw['source_part1']):return {'status':'NOT_COLLAPSED'}
        proof=evidence();proof['artifact']=str(kw['part1'])
        if unknown:proof.update(status='UNKNOWN',reason='missing native metadata',binding_verified=False)
        return proof
    monkeypatch.setattr(rt,'resolve_hailo_output_suitability',resolver)
    runs=[{'id':b+'_to_tensorrt','stage1':b,'stage2':'tensorrt'} for b in ('hailo8','hailo10h','deepx')]
    cfg=BenchmarkGenerationExecutionConfig(runtime=runtime,target_cases=1,gap=0,ranked_candidates=[2],candidate_search_pool=[2,3,4],
        out_dir=suite,base='model',pad=3,strict_boundary=False,model=analysis['model'],nodes=analysis['nodes'],order=analysis['order'],
        analysis_payload=analysis,full_model_src=str(source),require_single_part2_input=True,hef_targets=['hailo8','hailo10h'],
        hef_part1=True,bench_plan_runs=runs,hailo_build_hef_fn=leaf,backend_backfill_policy=DEFAULT_BACKFILL)
    cb=BenchmarkGenerationExecutionCallbacks(log=lambda *a,**k:None,queue_put=lambda *a:None,persist_state=runtime.persist,
        publish_hailo_diagnostics=lambda *a,**k:None,predicted_metrics_for_boundary=lambda *a:{},hailo_parse_entry_for_boundary=lambda *a:None,hailo_parse_scalar_fields=lambda *a:{})
    chosen=BenchmarkGenerationExecutionService().execute_case_build_loop(cfg,cb)
    assert chosen==([2] if unknown else [2,3])
    state=runtime.generation_state['backend_backfill'];matrix={c['backend']:c['selected_case_ids'] for c in state['contracts']}
    assert matrix=={'hailo8':['b002'],'hailo10h':[] if unknown else ['b003'],'deepx':['b002']}
    assert len(probes)==(1 if unknown else 2) and not any(b==4 or b==3 and a=='hailo8' for b,a in calls)
    assert all(case in {c['folder'] for c in runtime.cases} for contract in state['contracts'] for case in contract['selected_case_ids'])
    assert state['cold_builds_started']==0
