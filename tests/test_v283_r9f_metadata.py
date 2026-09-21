"""Early normal HEF metadata: real receipts/resolver/selection, isolated SDK processes."""
import copy
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
from onnx import helper as h, TensorProto as T

from onnx_splitpoint_tool import hailo_backend as hb
from onnx_splitpoint_tool.runners import native_split_quality_runtime as rt
from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract, preprocessing_contract_sha256
from test_v283_r9f_output_contract import graphs


def seal_fixture(hef, p1, *, arch='hailo10h'):
    contract = canonical_image_preprocessing_contract('detection', (32, 32))
    key, payload = hb._hailo_cache_key(model_path=p1, activation_part1=None, hw_arch=arch, opt_level=1,
        calib_dir=None, calib_count=1, calib_batch_size=1, extra_model_script='', start_nodes=None, end_nodes=None,
        preprocessing_contract=contract, effective_calib_count=1, calibration_storage='memory', calibration_memory_cap_bytes=67108864)
    receipt = hb._write_hailo_receipt(hef_path=hef, source_onnx=p1, compiler_onnx=p1, hw_arch=arch,
        net_name='arbitrary_graph', preprocessing_contract=contract, preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=key, cache_payload=payload, calibration_identity=payload['calibration_identity'], calibration_count=1)
    # Fixed compiler identity belongs only to test fixtures, never product conditions.
    receipt['hailo_sdk_version'] = receipt['cache_payload']['hailo_sdk_version'] = 'hailo-dataflow-compiler:5.3.0'
    receipt['cache_key'] = hashlib.sha256(json.dumps(receipt['cache_payload'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    hb._hailo_receipt_path(hef).write_text(json.dumps(receipt))
    assert hb._load_valid_hailo_receipt(hef, source_onnx_sha256=rt._sha256_file(p1),allow_legacy_v2=True)
    return receipt


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    folder=tmp_path/'sdk';folder.mkdir()
    (folder/'hailo_platform.py').write_text('''
import json
from pathlib import Path
from types import SimpleNamespace as N
class HEF:
    def __init__(self,path):
        self.row=json.loads(Path(path).read_text())
        with (Path(__file__).parent/'reads').open('a') as f:f.write(path+'\\n')
    def _info(self,native=False):
        r=self.row
        q=N(qp_scale=r.get('scale',3.2),qp_zp=r.get('zp',35))
        if 'rounding' in r:q.rounding=r['rounding']
        return N(name=r.get('name','other/output'),shape=r.get('shape',[1,11,11]),quant_info=q,
            format=N(type=r.get('native_dtype','UINT8') if native else r.get('host_dtype','UINT8'),order='FCR'))
    def get_output_vstream_infos(self):return [self._info()]
    def get_network_group_names(self):return ['different_network']
    def get_output_stream_infos(self,group):return [self._info(True)]
    def get_stream_names_from_vstream_name(self,name,group):return [self._info(True).name]
def VDevice(*a,**k):raise AssertionError('device opening forbidden')
''')
    bootstrap=folder/'bootstrap.py'
    bootstrap.write_text("import sys\nsys.path.insert(0,"+repr(str(folder))+")\na=sys.argv[1:]\nif a[0]=='-B':a=a[1:]\nassert a[0]=='-c'\ncode=a[1]\nsys.argv=['-c',*a[2:]]\nexec(compile(code,'<product-reader>','exec'))\n")
    reader=folder/'python';reader.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' -B '+shlex.quote(str(bootstrap))+' "$@"\n');reader.chmod(0o755)
    monkeypatch.setenv('HAILO_PY',str(reader))
    return folder,reader


def fixture_contract(tmp_path,row=None,prefix='renamed',classes=7):
    p1,p2=graphs(classes=classes,prefix=prefix)
    one=tmp_path/'source.onnx';two=tmp_path/'tail.onnx';onnx.save(p1,one);onnx.save(p2,two)
    hef=tmp_path/'compiled.hef';hef.write_text(json.dumps(row or {'shape':[1,11,classes+4]}))
    receipt=seal_fixture(hef,one)
    return dict(part1=hef,source_part1=one,source_part2=two,task='detection'),receipt


@pytest.mark.parametrize('row,status',[
    ({'scale':3.2},'INCOMPATIBLE'),({'scale':.01},'NOT_COLLAPSED'),
    ({'scale':3.2,'host_dtype':'FLOAT32'},'INCOMPATIBLE'),
    ({'native_dtype':'FLOAT32'},'UNKNOWN'),({'scale':3.2,'rounding':'ceil'},'NOT_COLLAPSED'),
    ({'scale':2,'zp':35,'rounding':'nearest_even'},'NOT_COLLAPSED'),
    ({'scale':2,'zp':34,'rounding':'nearest_even'},'INCOMPATIBLE'),
    ({'scale':[.01,3.2]},'UNKNOWN'),({'shape':[1,12,11]},'UNKNOWN'),
])
def test_real_isolated_reader_without_quality_or_engine(tmp_path,sdk,row,status):
    args,_=fixture_contract(tmp_path,row)
    result=rt.resolve_hailo_output_suitability(**args,metadata_context={})
    assert result['status']==status,result
    assert result['required']
    assert not (tmp_path/'native_split_quality_binding.json').exists()
    assert not list(tmp_path.glob('*.engine'))
    if status!='UNKNOWN':
        assert result['metadata_source']['kind']=='local_hef_reader'
        assert result['output_metadata']['device_opened'] is False
        assert result['native_tensor']['hef_format_order']=='FCR'


def test_reuse_in_generation_does_not_repeat_process_or_hash(tmp_path,sdk,monkeypatch):
    args,_=fixture_contract(tmp_path);context={'observations':{}}
    first=rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert first['status']=='INCOMPATIBLE'
    def forbidden(*a,**k):raise AssertionError('repeated hash')
    monkeypatch.setattr(rt,'_sha256_file',forbidden)
    assert rt.resolve_hailo_output_suitability(**args,metadata_context=context)==first
    assert len((sdk[0]/'reads').read_text().splitlines())==1
    args['part1'].write_text('{"scale":.01}')
    assert rt.resolve_hailo_output_suitability(**args,metadata_context=context)['status']=='UNKNOWN'


@pytest.mark.parametrize('change',['bytes','size','p1','p2_binding','reader_missing','json','truncated','duplicate'])
def test_normal_metadata_negatives_never_reject(tmp_path,sdk,monkeypatch,change):
    args,receipt=fixture_contract(tmp_path)
    if change=='bytes':args['part1'].write_bytes(b'x'*args['part1'].stat().st_size)
    if change=='size':args['part1'].write_bytes(b'longer foreign HEF')
    if change=='p1':args['source_part1'].write_bytes(onnx.helper.make_model(onnx.load(args['source_part1']).graph,producer_name='different').SerializeToString())
    if change=='p2_binding':
        (tmp_path/'native_split_quality_binding.json').write_text(json.dumps({'artifacts':{
            'part1_runtime':{'path':'/exact/existing.hef','sha256':receipt['hef_sha256'],'size_bytes':receipt['hef_size_bytes']},
            'source_part2_onnx':{'sha256':'e'*64}}}))
    if change=='reader_missing':monkeypatch.setenv('HAILO_PY',str(tmp_path/'missing'))
    if change in ('json','truncated','duplicate'):
        content={'json':'not json','truncated':'{"outputs": [','duplicate':'{"outputs": [], "outputs": []}'}[change]
        sdk[1].write_text('#!/bin/sh\nprintf %s '+shlex.quote(content)+'\n')
    result=rt.resolve_hailo_output_suitability(**args,metadata_context={})
    assert result['status']=='UNKNOWN' and result['required'],result


@pytest.mark.parametrize('foreign',[None,'hash','size'])
def test_remote_reader_only_exact_reference_without_quality_pass(tmp_path,sdk,monkeypatch,foreign):
    from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport
    args,receipt=fixture_contract(tmp_path)
    binding={'setup_id':'configured_setup','artifacts':{'part1_runtime':{'path':str(args['part1']),
        'sha256':receipt['hef_sha256'],'size_bytes':receipt['hef_size_bytes']},
        'source_part2_onnx':{'sha256':rt._sha256_file(args['source_part2'])}}}
    if foreign=='hash':binding['artifacts']['part1_runtime']['sha256']='a'*64
    if foreign=='size':binding['artifacts']['part1_runtime']['size_bytes']+=1
    (tmp_path/'native_split_quality_binding.json').write_text(json.dumps(binding))
    activate=tmp_path/'activate';activate.write_text('export PATH='+shlex.quote(str(sdk[0]))+':"$PATH"\n')
    monkeypatch.setenv('HAILO_PY',str(tmp_path/'missing-local-reader'))
    calls=[]
    def ssh_process(self,command,timeout=None,**kw):
        assert command[0]=='ssh';calls.append(command)
        shell_command=shlex.split(command[-1])[0]
        proc=subprocess.run(['bash','-c',shell_command],capture_output=True,text=True,timeout=timeout)
        return proc.returncode,proc.stdout+proc.stderr
    monkeypatch.setattr(SSHTransport,'_run_capture',ssh_process)
    monkeypatch.setattr(SSHTransport,'_diagnostic',lambda *a,**k:'unavailable')
    target={'id':'configured_setup','accelerator':'hailo10h','remote':{'enabled':True,'host':'controlled-test-only',
        'remote_venv':str(activate)}}
    result=rt.resolve_hailo_output_suitability(**args,metadata_context={'targets':[target]})
    assert result['status']==('UNKNOWN' if foreign else 'INCOMPATIBLE'),result
    assert len(calls)==(0 if foreign else 1)
    assert not binding.get('quality_completed') and 'engine' not in binding['artifacts']


def test_reader_timeout_is_unknown_without_retry(tmp_path,sdk,monkeypatch):
    args,_=fixture_contract(tmp_path)
    sdk[1].write_text('#!'+sys.executable+'\nimport time\ntime.sleep(10)\n')
    monkeypatch.setattr(rt,'_HEF_METADATA_TIMEOUT_S',.1)
    result=rt.resolve_hailo_output_suitability(**args,metadata_context={})
    assert result['status']=='UNKNOWN' and 'TimeoutExpired' in result['reason']


@pytest.mark.parametrize('fault',['hash','size','native_quant','mapping'])
def test_child_response_must_describe_exact_native_file(tmp_path,sdk,fault):
    args,_=fixture_contract(tmp_path)
    proc=subprocess.run([str(sdk[1]),'-B','-c',rt._HAILO_HEF_METADATA_PROBE,str(args['part1']),'--identity'],capture_output=True,text=True,check=True)
    data=json.loads(proc.stdout)
    if fault=='hash':data['part1_artifact_sha256']='c'*64
    if fault=='size':data['part1_artifact_size_bytes']+=1
    if fault=='native_quant':data['outputs'][0]['native_streams'][0]['quantization']['scale']=.001
    if fault=='mapping':data['outputs'][0]['native_stream_names']=['foreign']
    sdk[1].write_text('#!/bin/sh\nprintf %s '+shlex.quote(json.dumps(data))+'\n')
    result=rt.resolve_hailo_output_suitability(**args,metadata_context={})
    assert result['status']=='UNKNOWN' and result['required'],result


def test_existing_intrinsic_metadata_requires_no_quality_pass(tmp_path,sdk,monkeypatch):
    args,receipt=fixture_contract(tmp_path)
    found=rt.resolve_hailo_output_suitability(**args,metadata_context={})
    payload={'schema':'onnx-splitpoint/native-part1-boundary-metadata','schema_version':1,
        'part1_artifact_sha256':receipt['hef_sha256'],'part1_artifact_size_bytes':receipt['hef_size_bytes'],
        'boundary_tensor_count':1,'boundary_tensor':found['native_tensor']}
    payload['metadata_sha256']=rt.canonical_json_sha256(payload)
    (tmp_path/'part1_boundary_metadata.json').write_text(json.dumps(payload))
    monkeypatch.setenv('HAILO_PY',str(tmp_path/'reader_unavailable'))
    result=rt.resolve_hailo_output_suitability(**args,metadata_context={})
    assert result['status']=='INCOMPATIBLE' and result['metadata_source']['kind']=='exact_boundary_metadata'
    assert len((sdk[0]/'reads').read_text().splitlines())==1
    payload['boundary_tensor']['quantization']['scale']=.01
    (tmp_path/'part1_boundary_metadata.json').write_text(json.dumps(payload))
    assert rt.resolve_hailo_output_suitability(**args,metadata_context={})['status']=='UNKNOWN'


@pytest.mark.parametrize('missing',[False,True])
def test_normal_generator_real_resolver_and_ranked_backfill(tmp_path,sdk,monkeypatch,missing,metadata_targets=()):
    from onnx_splitpoint_tool.benchmark.services import (BenchmarkGenerationRuntime, BenchmarkGenerationExecutionConfig,
        BenchmarkGenerationExecutionCallbacks, BenchmarkGenerationExecutionService)
    from onnx_splitpoint_tool.core_analysis import analyze_model
    from onnx_splitpoint_tool.backend_backfill import DEFAULT_BACKFILL
    p1,p2=graphs()
    prefix=[h.make_node('Split',['image','widths'],['boxes','logits'],axis=1)]
    nodes=prefix+list(p1.graph.node)+[h.make_node('Identity',['renamed'],['bridge'])]+list(p2.graph.node)
    nodes[4].input[0]='bridge'
    model=h.make_model(h.make_graph(nodes,'unrelated_model',[h.make_tensor_value_info('image',T.FLOAT,[1,11,11])],
        list(p2.graph.output)+[h.make_tensor_value_info('ranked',T.FLOAT,[1,1,1])],list(p2.graph.initializer)),opset_imports=[h.make_opsetid('',13)])
    source=tmp_path/'unrelated.onnx';onnx.save(model,source);analysis=analyze_model(str(source))
    suite=tmp_path/'suite';suite.mkdir();calls=[]
    runtime=BenchmarkGenerationRuntime(suite,suite/'log',suite/'state.json',1,[3],[3,4,5],'unrelated',str(source),'end')
    # This leaf materializes cached fixture bytes and real receipts; it never
    # replaces the metadata resolver, semantic calculation or selection.
    def cached_fixture(path,**kw):
        assert kw['cache_only'] is True
        boundary=kw['build_evidence_context']['boundary'];calls.append((boundary,kw['hw_arch']))
        hef=Path(kw['outdir'])/'compiled.hef';hef.parent.mkdir(parents=True,exist_ok=True)
        hef.write_text(json.dumps({'scale':3.2 if boundary==3 else .01}))
        seal_fixture(hef,Path(path),arch=kw['hw_arch'])
        return SimpleNamespace(ok=True,skipped=False,timed_out=False,hef_path=hef,failure_kind='',error='',elapsed_s=0,details={'cache_hit':True},calib_info={})
    if missing:monkeypatch.setenv('HAILO_PY',str(tmp_path/'missing'))
    runs=[{'id':b+'_to_tensorrt','stage1':b,'stage2':'tensorrt'} for b in ('hailo8','hailo10h','deepx')]
    cfg=BenchmarkGenerationExecutionConfig(runtime=runtime,target_cases=1,gap=0,ranked_candidates=[3],candidate_search_pool=[3,4,5],
        out_dir=suite,base='unrelated',pad=3,strict_boundary=False,model=analysis['model'],nodes=analysis['nodes'],order=analysis['order'],
        analysis_payload=analysis,full_model_src=str(source),require_single_part2_input=True,hef_targets=['hailo8','hailo10h'],
        hef_part1=True,bench_plan_runs=runs,hailo_build_hef_fn=cached_fixture,backend_backfill_policy=DEFAULT_BACKFILL,benchmark_task='detection',
        hailo_metadata_targets=metadata_targets)
    cb=BenchmarkGenerationExecutionCallbacks(log=lambda *a,**k:None,queue_put=lambda *a:None,persist_state=runtime.persist,
        publish_hailo_diagnostics=lambda *a,**k:None,predicted_metrics_for_boundary=lambda *a:{},hailo_parse_entry_for_boundary=lambda *a:None,hailo_parse_scalar_fields=lambda *a:{})
    selected=BenchmarkGenerationExecutionService().execute_case_build_loop(cfg,cb)
    state=runtime.generation_state['backend_backfill']
    matrix={c['backend']:c['selected_case_ids'] for c in state['contracts']}
    assert matrix=={'hailo8':['b003'],'hailo10h':[] if missing else ['b004'],'deepx':['b003']},state
    assert selected==([3] if missing else [3,4])
    assert state['cold_builds_started']==0 and not any(b==5 or (b==4 and a=='hailo8') for b,a in calls)
    first=json.loads((suite/'b003/split_manifest.json').read_text())
    proof=first['hailo']['hefs']['hailo10h']['part1_output_suitability']
    assert proof['status']==('UNKNOWN' if missing else 'INCOMPATIBLE')


def test_observation_rechecks_changed_cache_meta(tmp_path,sdk):
    args,receipt=fixture_contract(tmp_path);context={'observations':{}}
    assert rt.resolve_hailo_output_suitability(**args,metadata_context=context)['status']=='INCOMPATIBLE'
    hb._hailo_cache_meta_path(args['part1']).write_text('{"cache_key":"foreign"}')
    result=rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert result['status']=='UNKNOWN' and result['reason']=='exact_hef_source_receipt_unavailable'
    assert len((sdk[0]/'reads').read_text().splitlines())==1


@pytest.mark.parametrize('conflict',[False,True])
def test_explicit_native_rounding_is_not_replaced_by_compiler_default(tmp_path,sdk,conflict):
    args,_=fixture_contract(tmp_path)
    proc=subprocess.run([str(sdk[1]),'-B','-c',rt._HAILO_HEF_METADATA_PROBE,str(args['part1']),'--identity'],capture_output=True,text=True,check=True)
    payload=json.loads(proc.stdout);row=payload['outputs'][0]
    row['native_streams'][0]['quantization']['rounding']='ceil'
    if conflict:row['quantization']['rounding']='nearest_even'
    sdk[1].write_text('#!/bin/sh\nprintf %s '+shlex.quote(json.dumps(payload))+'\n')
    result=rt.resolve_hailo_output_suitability(**args,metadata_context={})
    assert result['status']==('UNKNOWN' if conflict else 'NOT_COLLAPSED'),result


def test_normal_exact_registry_reference_without_adjacent_quality_binding(tmp_path,sdk,monkeypatch):
    from onnx_splitpoint_tool.artifact_store import ArtifactStore
    args,receipt=fixture_contract(tmp_path)
    tensor=rt.resolve_hailo_output_suitability(**args,metadata_context={})['native_tensor']
    cache_meta=hb._hailo_cache_meta_path(args['part1']);cache_meta.write_text(json.dumps(hb._hailo_cache_meta_from_receipt(receipt)))
    store=ArtifactStore(tmp_path/'artifact_store')
    monkeypatch.setenv('HAILO_PY',str(tmp_path/'unavailable'))
    context={'artifact_store_root':str(store.root),'observations':{}}
    assert rt.resolve_hailo_output_suitability(**args,metadata_context=context)['status']=='UNKNOWN'
    record=store.register_hailo_bundle(source_path=args['part1'],receipt_path=hb._hailo_receipt_path(args['part1']),
        cache_meta_path=cache_meta,contract={'kind':'hailo_hef','test':'exact source'},metadata={'legacy_cache_key':receipt['cache_key']})
    metadata={'schema':'onnx-splitpoint/native-part1-boundary-metadata','schema_version':1,
        'part1_artifact_sha256':receipt['hef_sha256'],'part1_artifact_size_bytes':receipt['hef_size_bytes'],
        'boundary_tensor_count':1,'boundary_tensor':tensor}
    metadata['metadata_sha256']=rt.canonical_json_sha256(metadata)
    existing=Path(record.object_path).parent/'part1_boundary_metadata.json';existing.write_text(json.dumps(metadata))
    result=rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert result['status']=='INCOMPATIBLE' and result['metadata_source']['path']==str(existing),result
    assert not (args['part1'].parent/'native_split_quality_binding.json').exists()
    assert rt.resolve_hailo_output_suitability(**args,metadata_context=context)==result
    existing.write_text('{}')
    result=rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert result['status']=='UNKNOWN',result


def test_broken_optional_registry_does_not_block_local_reader(tmp_path,sdk):
    args,_=fixture_contract(tmp_path)
    store=tmp_path/'broken_registry';store.mkdir();(store/'registry.sqlite3').write_bytes(b'broken')
    result=rt.resolve_hailo_output_suitability(**args,metadata_context={'artifact_store_root':str(store),'observations':{}})
    assert result['status']=='INCOMPATIBLE' and result['metadata_source']['kind']=='local_hef_reader',result
