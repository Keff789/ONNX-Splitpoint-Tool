"""Reproduce the real 2026-09-06 H8 YOLO11l b003 postflight validation gap."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import canonical_json_sha256, persist_detection_completion_execution_artifacts
from onnx_splitpoint_tool.native_three_stage import FastDetectionCompletionRuntime, NativeThreeStageError, verify_fast_completion_attestation
from onnx_splitpoint_tool.validation.host_postprocess import resolve_host_postprocess_evidence
from scripts.native_hailo_trt_fifo_from_benchmarkset import _hailo8_completion_fields
from scripts.native_producer_validate_visualize import (
    _copy_completed_v2_projection, _verified_completed_v2_contract,
    _completed_v2_self_reference_detection, _verified_execution_completed_result_artifact,
)
from test_v279_native_three_stage import _decoded_completion_contract


def _archived():
    raw = json.loads((Path(__file__).parent/'fixtures/v27925_h8_fast_oracle_yolo11l_b003.json').read_text())
    row = {}
    _copy_completed_v2_projection(row, raw)
    row['task'] = 'detection'
    return row


def test_real_v24_fast_oracle_seal_counts_and_physical_endpoint_recover():
    row = _archived()
    assert row['endpoint_contract_hash'] == row['completion_execution_contract']['source_endpoint']['endpoint_contract_hash']
    att = verify_fast_completion_attestation(row['completed_task_endpoint_attestation'], execution_contract=row['completion_execution_contract'])
    assert att['completed_work_units'] == 3000
    assert att['fast_detection_count'] == 7
    assert att['observation_relation'] == 'postflight_oracle_sentinel'
    assert resolve_host_postprocess_evidence(row)['available'] is True
    assert _verified_completed_v2_contract(row)[0] == 'native_three_stage_fast_oracle_outside_timing'


def test_real_v24_artifact_still_requires_exact_persisted_bytes(tmp_path):
    row = _archived()
    att = row['completed_task_endpoint_attestation']
    target = tmp_path/'result.json'
    persist_detection_completion_execution_artifacts(row, output_path=target)
    result = _verified_execution_completed_result_artifact(att['last_result'], att, row)
    assert len(result) == 7
    target.write_text('{}')
    with pytest.raises(Exception, match='persistence_invalid'):
        _verified_execution_completed_result_artifact(att['last_result'], att, row)


@pytest.mark.parametrize('mutation', ['seal','count','fast_hash','artifact','schema','location','detections'])
def test_real_v24_resealed_inconsistent_oracle_is_rejected(mutation):
    row = _archived();att=copy.deepcopy(row['completed_task_endpoint_attestation'])
    if mutation=='seal': att['status']='failed'
    elif mutation=='count': att['completion_count']=2999
    elif mutation=='fast_hash': att['fast_content_sha256']='0'*64
    elif mutation=='artifact': att['artifact']['execution_contract_sha256']='0'*64
    elif mutation=='schema': att['schema_sha256']='0'*64
    elif mutation=='location': att['observation_relation']='same_hotloop_sentinel'
    elif mutation=='detections': att['last_result']['detections'][0]['score']=0.5
    if mutation!='seal':
        att.pop('attestation_sha256');att['attestation_sha256']=canonical_json_sha256(att)
    with pytest.raises(Exception):
        verify_fast_completion_attestation(att,execution_contract=row['completion_execution_contract'])


@pytest.mark.parametrize('field,value', [
    ('endpoint_contract_hash','0'*64),('output_endpoint_id','wrong'),
    ('postprocess_completed_frames',2999),('postprocess_included',False),
    ('completion_observation_relation','same_hotloop_sentinel'),
    ('host_tail_available',False),('completed_task_comparison_endpoint_contract_hash','0'*64),
])
def test_projection_conflicts_are_not_silently_repaired(field,value):
    row = _archived();row[field]=value
    _copy_completed_v2_projection(row,dict(row))
    assert row[field]==value
    assert resolve_host_postprocess_evidence(row)['available'] is False
    with pytest.raises(Exception): _verified_completed_v2_contract(row)


def test_new_producer_to_persisted_semantics_and_changed_raw_dump(tmp_path):
    outputs,contract=_decoded_completion_contract()
    runtime=FastDetectionCompletionRuntime(contract)
    runtime.process(outputs);runtime.process(outputs)
    row=_hailo8_completion_fields(runtime,completed_work_units=2)
    row['task']='detection'
    persist_detection_completion_execution_artifacts(row, output_path=tmp_path/'sentinel.json')
    assert resolve_host_postprocess_evidence(row)['available'] is True
    result=_completed_v2_self_reference_detection(outputs,outputs,row)
    assert result['available'] is True, result
    assert result['semantic_result_binding_status']=='verified_postflight_oracle_matches_fast_sentinel'
    assert result['native_detections']==result['reference_detections']
    changed={k:np.asarray(v).copy() for k,v in outputs.items()}
    next(iter(changed.values())).flat[0]+=1
    result=_completed_v2_self_reference_detection(outputs,changed,row)
    assert result['available'] is False
    assert 'dumped_sentinel_mismatch' in result['reason']


def test_actual_runner_dumps_last_measured_outputs_without_extra_inference(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import native_hailo_trt_fifo_from_benchmarkset as runner
    outputs,contract=_decoded_completion_contract()
    events=[]
    class Backend:
        def run(self, *_):
            events.append('probe_hailo_infer')
            return SimpleNamespace(outputs={'boundary':np.ones((1,4),dtype=np.float32)})
    class Trt:
        def run(self, *_):
            events.append('probe_trt_infer')
            return outputs
    monkeypatch.setattr(runner,'_open_hailo8_python_runtime',lambda **kw:(Backend(),SimpleNamespace(input_names=['images'], handle=SimpleNamespace(runtime_input_shapes={'images':(2,2,3)})),Trt()))
    monkeypatch.setattr(runner,'_close_hailo8_python_runtime',lambda *_:events.append('close'))
    monkeypatch.setattr(runner,'_hailo8_python_input',lambda *a,**kw:{'images':np.ones((2,2,3),dtype=np.uint8)})
    monkeypatch.setattr(runner,'_hailo8_python_boundary',lambda *a,**kw:('boundary',np.ones((1,4),dtype=np.float32),{}))
    monkeypatch.setattr(runner,'_hailo8_detection_completion_contract',lambda **kw:contract)
    monkeypatch.setattr(runner,'_annotate_output_contract',lambda *a:None)
    monkeypatch.setattr(runner,'_seal_manifest_payload_files',lambda *a:None)
    measured=[]
    def fifo(*a,**kw):
        values={k:v.copy() for k,v in outputs.items()}
        next(iter(values.values())).flat[0]+=len(measured)+1
        runtime=kw['completion_runtime']
        runtime.process(values);runtime.process(values)
        measured.append(values)
        events.append('measured_fifo_returned')
        return {**runner._hailo8_completion_fields(runtime,completed_work_units=2),'fps_makespan':10+len(measured)}
    monkeypatch.setattr(runner,'_hailo8_python_fifo_run',fifo)
    args=SimpleNamespace(model_id='yolo26s',hailo_format='uint8',preprocess_mode_effective='letterbox',letterbox_pad_value=114,
        dump_outputs=True,dump_boundary=False,output_dir='',boundary_dir='',benchmark_set=str(tmp_path),precision='fp16',
        repetitions=3,completion_runtime_mode='fast_oracle_outside_timing',frames=2,warmup=0,queue_depth=2,duration_s=0)
    payload,_,_=runner._run_hailo8_python_detection(bs=tmp_path,case='b001',hef=tmp_path/'x.hef',engine=tmp_path/'x.engine',image=tmp_path/'x.jpg',work=tmp_path,args=args,quality_binding=None)
    manifest=json.loads(Path(payload['native_fifo_output_manifest']).read_text())
    assert manifest['dump_inference_scope']=='last_measured_completion_source_outputs'
    dumped=np.fromfile(tmp_path/'native_fifo_outputs/output_000.bin',dtype=np.float32).reshape(next(iter(measured[-1].values())).shape)
    assert np.array_equal(dumped,next(iter(measured[-1].values())))
    assert not np.array_equal(dumped,next(iter(outputs.values())))
    assert events.count('probe_hailo_infer')==1
    assert events.count('probe_trt_infer')==1
    assert events.count('measured_fifo_returned')==3
    assert events.count('close')==4
    assert payload['fps_makespan']==12.0
    verify_fast_completion_attestation(payload['completed_task_endpoint_attestation'],execution_contract=contract,outputs=measured[-1])
