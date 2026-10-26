"""Normal plan/render/control integration; only storage/SDK arithmetic are fixtures."""
import ast
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import generic_only_plan
from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1
from tests.test_generic_only_plan import scope_fixture
from tests.test_generic_only_generated import prepared_case, image_value
from tests.test_v272_detection_completion_runtime import _decoded_outputs
from tests.test_v2912_generic_split_completion import _module


def descriptor(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_normal_single_feature_plan_renders_and_runs_all_bound_observer_tasks(monkeypatch,tmp_path):
    fixture=json.loads((Path(__file__).parent/'fixtures/generic_single_feature_boundary/deepx_yolov7_b009_actual_context.json').read_text())
    context=fixture['runtime_context'];runtime=context['boundary_runtime']
    boundary=dict(schema_version=1,minimum_boundary_tensors=1,**deepcopy(context['logical_boundary']),
        **{k:deepcopy(runtime[k])for k in ('part2_to_part1','passthrough_inputs','transfer_contract')})
    _,_,_,row=scope_fixture(tmp_path/'scope',monkeypatch,backend='deepx')
    row['boundary_observation']=deepcopy(boundary)
    row['generic_artifact_binding']['boundary_contract']=deepcopy(boundary)
    planned,_=generic_only_plan.runtime_configuration(row,str(tmp_path/'normal_stage'),'single-feature-rendered-test')
    assert planned['boundary_observation']==planned['generic_artifact_binding']['boundary_contract']==boundary
    # Normal existing writer copies the self-contained library and renders CLI.
    suite=tmp_path/'normal_rendered_suite';suite.mkdir()
    module=_module(monkeypatch,suite_dir=suite)
    rendered=Path(module.__file__)
    config,output,_=prepared_case(tmp_path,'deepx')
    config['boundary_observation']=deepcopy(planned['boundary_observation'])
    config['generic_artifact_binding']['boundary_contract']=deepcopy(planned['generic_artifact_binding']['boundary_contract'])
    sidecar=Path(config['mixed_input_control']['path']);payload=json.loads(sidecar.read_text())
    payload['boundary_observation']=deepcopy(config['boundary_observation'])
    payload['task_input_indices']=[(i%2)^(i>=16)for i in range(32)]
    payload['task_sequence_policy']='first16_alternate_second16_invert_v1'
    sidecar.write_text(json.dumps(payload));config['mixed_input_control']=descriptor(sidecar)
    for source in config['runtime_source_files']:
        if source['path']==str(sidecar):source['sha256']=config['mixed_input_control']['sha256']
    config['runtime_source_files'].append(descriptor(rendered))
    name=next(iter(boundary['part1_outputs']));shape=boundary['part1_outputs'][name]['shape']
    assert name=='getitem_9'and shape==[1,128,160,160]
    def values(feed):return {name:np.full(shape,image_value(feed['images']),dtype=np.float32)}
    class Engine:
        def __init__(self):self.jobs={}
        def get_input_tensors_info(self):return [{'name':'images','shape':[1,64,64,3],'dtype':'uint8'}]
        def get_output_tensors_info(self):return [{'name':name,'shape':shape,'dtype':'float32'}]
        def run_async(self,arrays,*,user_arg,output_buffer):
            out=np.ndarray(shape,dtype=np.float32,buffer=output_buffer);out.fill(image_value(arrays[0]));self.jobs[user_arg]=[out];return user_arg
        def wait(self,job_id):return self.jobs.pop(job_id)
    class Session:
        def __init__(self):self._engine=Engine()
        def _prepare_async_inputs(self,feed):return [feed['images']]
        def _materialize_outputs(self,outputs):return {name:outputs[0]}
    session=Session();adapter=DeepXAsyncStage1(session,lambda:pytest.fail('unused timer input'),inflight=4)
    def stage2(inputs,feed):
        out=_decoded_outputs();out['detections'][0,0,(0,2)]+=float(inputs[name].flat[0]);return out
    def observed():
        return dict(part1_io={'canonical_outputs':[name],'synthetic_CPU_SDK':True},
            part2_io={'inputs':boundary['part2_inputs']},
            **{k:deepcopy(boundary[k])for k in ('part2_to_part1','passthrough_inputs','transfer_contract')})
    # Execute the exact emitted control-dispatch block, including its decision
    # to supply the live observer. No equivalent hand-built report/callback.
    tree=ast.parse(rendered.read_text())
    blocks=[n for n in ast.walk(tree)if isinstance(n,ast.If)
        and any(isinstance(c,ast.FunctionDef)and c.name=='mixed_control'for c in n.body)]
    assert len(blocks)==1
    ns=dict(generic_completion_config=config,out_dir=output,hailo_p1=None,
        run_stage1_map=values,completion_transfer=lambda outputs,feed:outputs,
        completion_stage2=stage2,stage1_async=adapter,streaming_impl='auto',
        canonical_full_feeds=None,_generic_boundary_runtime=observed,__file__=str(rendered))
    exec(compile(ast.Module(body=blocks,type_ignores=[]),str(rendered),'exec'),ns)
    assert ns['boundary_runtime'] is observed
    proof=ns['mixed_control'](1)
    assert proof['mixed_completed_count']==proof['serial_reference_count']==32
    assert proof['boundary_observation']['observed_boundary_tensor_count']==1
    assert proof['boundary_observation']['observed_task_count']==32
    assert proof['boundary_observation']['slot_buffer_ownership']['slots_disjoint']
    assert not adapter.pending and not session._engine.jobs
    binding=importlib.import_module('splitpoint_runners.generic_mixed_control_binding')
    repetition=dict(stage1_execution=config['stage1_execution'],handoff_payload_copy=False,mixed_input_control=proof)
    assert binding.validate_bound_mixed_result(repetition,config,result_root=output)==proof
    report_path=output/proof['report']['path'];report=json.loads(report_path.read_text())
    assert len(report['observations'])==32 and all(o['boundary_observation']['passed']for o in report['observations'])
    tensor=report['observations'][13]['boundary_observation']['mixed']['part2_before_inference'][name]
    tensor['dtype']='float16';report_path.write_text(json.dumps(report));proof['report']['sha256']=descriptor(report_path)['sha256']
    with pytest.raises(ValueError):binding.validate_bound_mixed_result(repetition,config,result_root=output)
