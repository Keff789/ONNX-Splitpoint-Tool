"""R9C: current-output top-k completion in the existing prepared-input paths."""
from __future__ import annotations
import ast
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from onnx_splitpoint_tool.runners.harness.classification import classification_topk, ClassificationCompletion
from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int8, np.uint8, np.int32])
@pytest.mark.parametrize('values,expected', [([2,7,7,1,7,5,0], [1,2,4,5,0]), ([3,3,3], [0,1,2]), ([8], [0])])
def test_reference_dtype_ties_small_k(dtype, values, expected):
    logits=np.asarray(values,dtype=dtype)
    assert classification_topk({'logits':logits}).tolist()==[expected]
    # The original quality Top-k ranks float32 logits with stable ties.
    template=(ROOT/'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt').read_text()
    tree=ast.parse(template)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'_classification_topk','_softmax1d'}]
    ns={'np':np,'Tuple':tuple,'Optional':__import__('typing').Optional,'List':list}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<quality-reference>','exec'),ns)
    top1,top5,_=ns['_classification_topk'](logits)
    assert top1==expected[0] and top5==expected

@pytest.mark.parametrize('shape', [(2,3),(2,1,1,3),(2,3,1,1)])
def test_batch_axis_and_offset_do_not_merge_classes(shape):
    values=np.asarray([[2,3,1],[5,0,8]],dtype=np.float32).reshape(shape)
    assert classification_topk({'aux':np.ones((1,99)),'predictions':values},label_offset=1).tolist()==[[2,1,3],[3,1,2]]

@pytest.mark.parametrize('output', [{}, {'x':np.zeros((2,2,2))}, {'x':np.zeros((1,0))},
    {'x':np.array([np.nan,2])}, {'x':np.array([np.inf,2])}, {'x':np.array([-np.inf,2])}, {'x':np.array(['one'])}])
def test_invalid_outputs_never_count_as_completed(output):
    processor=ClassificationCompletion()
    with pytest.raises(ValueError):processor.process(output)
    assert processor.completed_count==0 and not processor.last_result

@pytest.mark.parametrize('backend',['h8_python','h10','deepx'])
def test_actual_fifo_completes_current_outputs_once_with_warmup_separate(monkeypatch,backend):
    from test_v272_detection_hotloop_integration import _FakeTRT,_FakeDeepX,_FakeHailo10Session
    from scripts import native_hailo10_trt_e2e_from_benchmarkset as h10, native_deepx_trt_e2e_from_benchmarkset as dx, native_hailo_trt_fifo_from_benchmarkset as h8
    import onnx_splitpoint_tool.runners.harness.classification as completion
    calls=[];real=completion.classification_topk
    def topk(outputs,**kw):
        result=real(outputs,**kw);calls.append(int(result[0,0]));time.sleep(.001);return result
    monkeypatch.setattr(completion,'classification_topk',topk)
    class TRT(_FakeTRT):
        def __init__(self):super().__init__();self.n=0
        def run_prepared(self):
            self.n+=1;v=np.zeros((1,7),np.float32);v[0,self.n%7]=10;return {'logits':v}
    trt=TRT();opts=dict(frames=12,warmup=2,queue_depth=1)
    if backend=='deepx': result=dx._deepx_fifo_run(_FakeDeepX(),np.zeros((1,4)),trt,**opts)
    elif backend=='h10':result=h10._hailo10_async_fifo_run(_FakeHailo10Session(),{'input':np.zeros((1,4))},trt,inflight=3,**opts)
    else:
        class Hailo:
            def run(self,*args):return SimpleNamespace(outputs={'boundary':np.zeros((1,4),np.float32)})
        result=h8._hailo8_python_fifo_run(Hailo(),None,{'input':np.zeros((1,4))},trt,
            duration_s=0,completion_runtime=ClassificationCompletion(),warmup_completion_runtime=ClassificationCompletion(),**opts)
    assert len(calls)==14 and calls==[n%7 for n in range(1,15)]
    assert result['postprocess_completed_frames']==result['completed_frames']==12
    raw=result['request_latency'];assert raw['task_complete'] and raw['status']=='complete'
    assert raw['count']==12 and raw['min_ms']>=1 and raw['warmup_included'] is False
    assert rate_endpoint_fields(result)['completed_task_fps']==pytest.approx(result['fps_makespan'])


def test_actual_hailo_full_callback_topk_before_completion_and_slot_reuse(monkeypatch):
    from test_v269_native_runner_p0_completion import _completion_session
    session,slots=_completion_session();processor=ClassificationCompletion();calls=[]
    def finish(outputs):
        answer=processor.process(outputs);calls.append(answer['top1'].tolist());time.sleep(.001)
    result=session.benchmark_throughput({'input':np.zeros(1)},frames=12,inflight=3,warmup_frames=2,postprocess_callback=finish)
    assert len(calls)==14 and calls==[[2]]*14
    assert result['postprocess_completed_frames']==12
    assert result['request_latency']['count']==12 and result['request_latency']['min_ms']>=1


def test_h10_unmeasured_semantic_capture_has_no_task_completion_dependency(monkeypatch):
    from scripts import native_hailo10_trt_e2e_from_benchmarkset as h10
    from test_v272_detection_hotloop_integration import _FakeHailo10Session
    def forbidden():
        raise AssertionError('Raw semantic capture must not construct a task completion hook')
    monkeypatch.setattr(h10,'ClassificationCompletion',forbidden)
    session=_FakeHailo10Session()
    session.describe_io=lambda: {'test':'raw-slot'}
    inputs={'input':np.zeros((1,4),np.float32)}
    for diagnostic in [None,{}]:
        result=h10._capture_raw_hailo10_sample(session,inputs,diagnostic_capture=diagnostic)
        np.testing.assert_array_equal(result['trt_input'],np.arange(4,dtype=np.float32).reshape(1,4))
        if diagnostic is not None:
            assert diagnostic['runtime_io']=={'test':'raw-slot'}
            assert diagnostic['raw_outputs']['boundary'].shape==(1,4)


def test_cpp_scalar_parity_and_half_conversion(tmp_path):
    from scripts.native_hailo_trt_fifo_from_benchmarkset import CPP_SOURCE
    start=CPP_SOURCE.index('// Classification ranking')
    end=CPP_SOURCE.index('// End classification scalar helpers.')
    code='#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <limits>\n#include <numeric>\n#include <stdexcept>\n#include <vector>\n#include <iostream>\n'+CPP_SOURCE[start:end]
    code+='''int main() {
      for (const auto &v : std::vector<std::vector<float>>{{2,7,7,1,7,5,0},{3,3,3},{8}}) {
        for(auto i:classification_top5(v)) std::cout << i << " "; std::cout << "\\n";
      }
      if(classification_half(0x3c00)!=1 || classification_half(0xc000)!=-2 || classification_half(1)!=std::ldexp(1.f,-24)) return 2;
      for(float v:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        try {classification_top5({1,v});return 3;}catch(const std::runtime_error&){}
      }
    }'''
    source=tmp_path/'topk.cpp';source.write_text(code);binary=tmp_path/'topk'
    subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(binary)],check=True,timeout=30)
    result=subprocess.run([str(binary)],capture_output=True,text=True,check=True,timeout=10)
    assert result.stdout.splitlines()==['1 2 4 5 0 ','0 1 2 ','0 ']


def test_trt_full_uses_existing_hotloop_and_preserves_trtexec_series(tmp_path,monkeypatch):
    from scripts import native_trt_full_completed_hotloop as hotloop,native_full_baseline_eval_runner as full
    from test_v270i_p0_completed_endpoint import _write_runtime_input
    manifest,_,_= _write_runtime_input(tmp_path);engine=tmp_path/'existing.engine';engine.write_bytes(b'existing test engine')
    instances=[]
    class TRT:
        inputs=['images'];shapes={'images':(1,3,4,4)};dtypes={'images':np.dtype(np.float32)}
        def __init__(self,path):assert path==engine;self.runs=0;self.prepares=0;self.closed=False;instances.append(self)
        def prepare_inputs(self,feed):self.prepares+=1
        def run_prepared(self):
            self.runs+=1;v=np.zeros((1,7),np.float32);v[0,self.runs%7]=9;return {'logits':v}
        def close(self):self.closed=True
    monkeypatch.setattr(hotloop,'NativeTRT',TRT)
    commands=[]
    def run(command,**kw):
        commands.append(command);monkeypatch.setattr(sys,'argv',command[1:]);return {'rc':hotloop.main(),'timed_out':False}
    monkeypatch.setattr(full,'_run',run)
    row=full._attach_trt_completed_task_hotloop({'ok':True,'task':'classification','contract_family':'classification_logits',
        'input_manifest':str(manifest),'quality_first_producer_identity':{'engine':{'path':str(engine),'sha256':full._sha256_file(engine)}},
        'fps_makespan':999,'latency_mean_ms':3.25,'fps_source':'trtexec'},tmp_path,'mobilenet',
        SimpleNamespace(frames=12,warmup=2,duration_s=0,timeout=30))
    assert row['ok'],row
    assert len(commands)==1 and '--classification' in commands[0]
    assert row['accelerator_only_diagnostic']['fps_makespan']==999 and row['accelerator_only_diagnostic']['latency_mean_ms']==3.25
    assert row['postprocess_completed_frames']==12 and row['request_latency']['count']==12
    assert row['classification_topk']['top1']==[15%7]
    assert instances[0].runs==15 and instances[0].prepares==1 and instances[0].closed
    projection=rate_endpoint_fields(full._aggregate_full_repetitions([row],requested=1))
    assert projection['completed_task_fps']==row['fps_makespan']
    assert projection['request_latency_status']=='complete'
    assert row['completed_task_endpoint_attested'] is False # no fabricated Detection/Quality attestation


def test_deepx_full_real_prepared_loop_and_semantic_merge(tmp_path,monkeypatch):
    from test_v27930_native_full_semantic_merge import prepare_runner_case,runner
    values=np.asarray([[1,2,3,7,7,0]],np.float32)
    case=prepare_runner_case(tmp_path,monkeypatch,model='mobilenet_v3_large',task='classification',output=values)
    semantic,blocked=runner._deepx_full_series_preflight(case.root,case.model,case.ns)
    assert blocked is None,blocked
    row=runner._generic_full_via_suite(case.root,case.model,'native_full_deepx','deepx_m1_full',case.ns,prepared_input_manifest=Path(semantic['input_manifest']))
    assert row['ok'],row
    assert row['postprocess_completed_frames']==3 and row['classification_topk']['top1']==[3]
    assert row['e2e_scope']=='full_task_pipeline'
    assert row['comparison_endpoint_stratum']=='classification_top1_top5'
    assert row['postprocess_location']=='host'
    merged=runner._attach_semantic_dump(row,case.root,case.model,'native_full_deepx','deepx_m1_full',case.ns,precomputed_result=semantic)
    assert merged['ok'],merged
    assert merged['request_latency']['task_complete'] and merged['postprocess_completed_frames']==3
    assert rate_endpoint_fields(runner._aggregate_full_repetitions([merged],requested=1))['completed_task_fps']==row['fps_makespan']


def test_classification_rate_rejects_completion_count_conflict():
    from test_v283_r9b_request_latency import sample
    row={'task':'classification','task_complete':True,'completed_task_stage':'classification_top1_top5','contract_family':'classification_logits',
        'postprocess_completion_verified':True,'postprocess_completed_frames':99,'completed_frames':100,'makespan_ms':100,'fps_makespan':1000,
        'measurement_boundary':'workers_ready_to_last_completed_task_frame','request_latency':sample([1]*100)}
    assert rate_endpoint_fields(row)['completed_task_fps'] is None


def test_h10_sync_existing_hotloop_topk_and_warmup():
    import queue,threading,textwrap
    from scripts import native_hailo10_trt_e2e_from_benchmarkset as h10
    from onnx_splitpoint_tool.runners.request_latency import RequestLatency
    from test_v272_detection_hotloop_integration import _FakeTRT
    source=(ROOT/'scripts/native_hailo10_trt_e2e_from_benchmarkset.py').read_text()
    start=source.index('            # Legacy synchronous producer path')
    end=source.index("        if bool(getattr(ns, 'dump_outputs'",start)
    body=textwrap.dedent(source[start:end])
    class TRT(_FakeTRT):
        def __init__(self):super().__init__();self.n=0
        def run_prepared(self):self.n+=1;return {'logits':np.array([[0.,self.n,0.]])}
    class Hailo:
        def run(self,*args):return SimpleNamespace(outputs={'boundary':np.zeros((1,4),np.float32)})
    trt=TRT()
    ns={'np':np,'time':time,'queue':queue,'threading':threading,'RequestLatency':RequestLatency,
        'ClassificationCompletion':ClassificationCompletion,'task_effective':'classification',
        'completion_execution_contract':None,'_open_fresh_runtime':lambda:(Hailo(),None,trt,'test-sync'),
        'inputs':{'input':np.zeros((1,4))},'_pick_hailo_output':h10._pick_hailo_output,'hailo_output_format':'float32','report':{},
        'ns':SimpleNamespace(repetitions=1,warmup=2,frames=12,duration_s=0,queue_depth=1,inflight=3)}
    exec(compile('def actual_sync():\n'+textwrap.indent(body,'    ')+'\n    return report\n','<existing-sync-hotloop>','exec'),ns)
    result=ns['actual_sync']()
    assert trt.n==14 and result['postprocess_completed_frames']==12
    assert result['classification_topk']['top1']==[1]
    assert result['request_latency']['status']=='complete' and result['request_latency']['task_complete']


def build_classification_report_fixture(tmp_path,endpoint):
    from test_v283_r9b_request_latency import sample
    from scripts import native_producer_final_report as report
    from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
    from onnx_splitpoint_tool.workflow.scientific_reporting import _write_reports
    raw={'task':'classification','case':'b135','model':'mobilenet','contract_family':'classification_logits','ok':True,
         'completed_frames':100,'completed_work_units':100,'makespan_ms':400,'fps_makespan':250,
         'measurement_boundary':'workers_ready_to_last_completed_trt_frame','measurement_endpoint':'model_outputs',
         'task_quality_status':'FAIL','repetition_count_requested':1,'repetition_count_valid':1}
    if endpoint=='topk':
        raw.update(ClassificationCompletion().report(100));raw['request_latency']=sample([4]*100)
    elif endpoint=='host':raw['request_latency']=sample([2]*100,task_complete=False)
    path=tmp_path/'raw.json';path.write_text(json.dumps(raw))
    tables=tmp_path/'analysis_tables';tables.mkdir()
    (tables/'native_fifo_eval_runner_r9c.json').write_text(json.dumps({'rows':[{'model':'mobilenet','case':'b135','report':str(path),'ok':True}]}))
    row,=report._rows_from_native_fifo_runner(tmp_path,include_direct_fallback=False)
    reports=tmp_path/'reports';reports.mkdir()
    (reports/'native_producer_combined_summary.json').write_text(json.dumps({'rows':[row]}))
    matrix=collect_native_performance_matrix(tmp_path)
    _write_reports(reports/'scientific',{'created_at':'2026-09-17','profile_id':'r9c','rows':[],'summary':{},'native_performance_matrix':matrix})
    import csv
    with (reports/'scientific/native_performance_observations.csv').open() as stream: exported,=list(csv.DictReader(stream))
    if endpoint=='topk':
        assert float(exported['request_latency_mean_ms'])==4 and float(exported['completed_task_fps'])==250
    else:
        assert not exported['request_latency_mean_ms'] and not exported['completed_task_fps']
        assert float(exported['host_output_fps'])==250
    return matrix

@pytest.mark.parametrize('endpoint',['topk','host','historical'])
def test_reporter_to_csv_preserves_endpoint_and_quality_axes(tmp_path,endpoint):
    matrix=build_classification_report_fixture(tmp_path,endpoint)
    row,=matrix['observations']
    assert row.get('task_quality_status')!='PASS'
