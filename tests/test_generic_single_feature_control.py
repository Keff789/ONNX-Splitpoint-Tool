"""Explicit 1→1 controls; actual009 metadata and synthetic CPU SDK adapters."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_mixed_control as mixed
from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from tests.test_generic_mixed_control import recipe, Session, callbacks


def single_contract():
    spec={'shape':[1,6],'dtype':'float32'}
    return dict(schema_version=1,minimum_boundary_tensors=1,part1_outputs={'cut':deepcopy(spec)},
        part2_inputs={'cut':deepcopy(spec)},part2_to_part1={'cut':'cut'},passthrough_inputs={},
        transfer_contract={'boundary_mode':'canonical_float32'})


def test_actual_009_single_feature_contract_preserves_exact_observed_names_shapes_and_types():
    value=json.loads((Path(__file__).parent/'fixtures/generic_single_feature_boundary/deepx_yolov7_b009_actual_context.json').read_text())
    ctx=value['runtime_context'];observed=ctx['boundary_runtime'];logical=ctx['logical_boundary']
    assert value['image_count']==ctx['boundary_observation_count']==5000
    contract=dict(schema_version=1,minimum_boundary_tensors=1,**deepcopy(logical),
                  **{k:deepcopy(observed[k])for k in ('part2_to_part1','passthrough_inputs','transfer_contract')})
    before=deepcopy(contract)
    assert mixed.normalize_boundary_observation(contract)==before
    assert contract['part1_outputs']==contract['part2_inputs']=={'getitem_9':{'shape':[1,128,160,160],'dtype':'float32'}}
    assert contract['passthrough_inputs']=={}


@pytest.mark.parametrize('fault',['empty_p1','empty_p2','foreign_source','foreign_target','unmapped_input',
    'missing_mapping','unexpected_passthrough','lower_multitensor_minimum','zero_minimum','bool_minimum'])
def test_single_feature_opt_in_does_not_relax_mapping_coverage_or_multitensor_contract(fault):
    value=single_contract()
    if fault=='empty_p1':value['part1_outputs'].clear()
    elif fault=='empty_p2':value['part2_inputs'].clear()
    elif fault=='foreign_source':value['part2_to_part1']['cut']='foreign'
    elif fault=='foreign_target':value['part2_to_part1']={'foreign':'cut'}
    elif fault=='unmapped_input':value['part2_inputs']['extra']=deepcopy(value['part2_inputs']['cut'])
    elif fault=='missing_mapping':value['part2_to_part1'].clear()
    elif fault=='unexpected_passthrough':value['passthrough_inputs']={'cut':'images'}
    elif fault=='lower_multitensor_minimum':
        value['part1_outputs']['other']=deepcopy(value['part1_outputs']['cut']);value['part2_inputs']['other']=deepcopy(value['part2_inputs']['cut']);value['part2_to_part1']['other']='other'
    elif fault=='zero_minimum':value['minimum_boundary_tensors']=0
    else:value['minimum_boundary_tensors']=True
    with pytest.raises(ValueError):mixed.normalize_boundary_observation(value)


@pytest.mark.parametrize('fault',[None,'wrong_dtype','wrong_shape','wrong_content','wrong_live_mapping'])
def test_all_32_single_features_use_the_existing_async_slot_and_typed_boundary_proofs(recipe,fault):
    tasks,config=deepcopy(recipe);value=single_contract();first,transfer,second,_,_=callbacks();session=Session()
    def transferred(outputs,feed):
        result=transfer(outputs,feed)
        if fault=='wrong_dtype':result['cut']=result['cut'].astype(np.float16)
        elif fault=='wrong_shape':result['cut']=result['cut'].reshape(2,3)
        return result
    def runtime():
        return dict(part1_io={'canonical_outputs':['cut']},part2_io={'inputs':value['part2_inputs']},
            part2_to_part1={'cut':'foreign'}if fault=='wrong_live_mapping'else value['part2_to_part1'],
            passthrough_inputs={},transfer_contract=value['transfer_contract'])
    def cloned(outputs):
        if fault=='wrong_content':outputs['cut']+=1
        return {k:v.copy()for k,v in outputs.items()}
    try:
        args=dict(config=config,original_sources={'synthetic_CPU_only':True},serial_stage1=first,
            transfer=transferred,stage2=second,processor_factory=ClassificationCompletion,
            stage1_async_factory=lambda prepare:DeepXAsyncStage1(session,prepare,inflight=4),
            boundary_observation=value,boundary_runtime=runtime,clone_payload=cloned if fault=='wrong_content'else None)
        if fault is not None:
            with pytest.raises(mixed.MixedControlError):mixed.run_mixed_control(tasks,**args)
            return
        report=mixed.run_mixed_control(tasks,**args)
        assert report['status']=='passed'and report['mixed_completed_count']==report['serial_reference_count']==32
        assert report['boundary_observation']['observed_boundary_tensor_count']==1
        assert report['boundary_observation']['observed_task_count']==32
        assert report['boundary_observation']['slot_buffer_ownership']['slots_disjoint']
        assert all(o['boundary_observation']['passed']for o in report['observations'])
    finally:session._engine.finish()
