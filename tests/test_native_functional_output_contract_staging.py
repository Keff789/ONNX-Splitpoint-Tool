"""Source-only output declaration staging checks; no SDK or inference runs."""
import json
import os
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract, load_manifest_outputs, runtime_output_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    build_detection_completion_execution_contract, verify_detection_completion_execution_contract,
)
from onnx_splitpoint_tool.workflow import native_functional_reference as subject
from tests.test_native_functional_reference import finite_plan


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def archive(tmp_path, monkeypatch):
    suite=tmp_path/'archive/models/yolov7_paper/benchmark_set/legacy_suite'
    case=suite/'b009';case.mkdir(parents=True)
    path=suite/'output_contracts.json'
    write(path,{'model_id':'yolov7_paper','task':'detection','contracts':[{
        'backend':'cuda_ort','variant':'full','model_id':'yolov7_paper','task':'detection',
        'contract_status':'recorded','endpoint_mode':'raw_detection_head',
        'host_tail_required':True,'postprocessing_required':True,'requires_external_postprocess':True}]})
    declaration=load_authoritative_output_contract(path,backend='tensorrt',model_id='yolov7_paper',task='detection')
    assert declaration['authoritative_output_contract'] is True
    native=tmp_path/'archive/native_producers/job/native.json';write(native,{})
    manifest=native.parent/'native_outputs/native_outputs_manifest.json'
    write(manifest,{'authoritative_output_contract_resolution':declaration})
    import onnx_splitpoint_tool.resources.remote_scripts.native_producer_final_report as reporter
    monkeypatch.setattr(reporter,'_resolve_split_semantic_manifest',lambda *_a,**_k:
                        (manifest,{'manifest_resolution_status':'manifest_verified'}))
    return {'model_id':'yolov7_paper','native_source':str(native),'original_local_case_dir':str(case)},path,manifest


def test_original_archived_declaration_is_bound_by_actual_file_and_selected_row(archive):
    row,path,manifest=archive
    result=subject._output_contract_source(row,{})
    assert result['source']==subject.bound(path)
    assert result['semantic_output_manifest']==subject.bound(manifest)
    assert result['declaration']['backend']=='cuda_ort' and result['declaration']['stage']=='raw_head'


def test_private_captured_declaration_keeps_its_exact_historical_container(archive):
    row,path,_=archive
    captured=Path(row['native_source']).parent/'benchmark_set/output_contracts.json'
    captured.parent.mkdir();captured.write_bytes(path.read_bytes())
    path.write_text('{}')
    assert subject._output_contract_source(row,{})['source']==subject.bound(captured)


@pytest.mark.parametrize('fault',[None,'changed_plan','remote_mapping','source_hash'])
def test_separate_process_summary_requires_original_normal_stage_mapping(archive,tmp_path,fault):
    row,path,manifest=archive
    declaration=json.loads(manifest.read_text())['authoritative_output_contract_resolution']
    staging=tmp_path/'original_normal_staging_plan.json'
    payload={'stage_files':[{'local':str(path),'remote':declaration['declaration_source'],'sha256':subject.sha(path)}]}
    if fault=='remote_mapping':payload['stage_files'][0]['remote']='/unrelated/output_contracts.json'
    elif fault=='source_hash':payload['stage_files'][0]['sha256']='0'*64
    write(staging,payload)
    summary=tmp_path/'original_native_summary.json'
    write(summary,{'plan':str(staging),'plan_sha256':subject.sha(staging)})
    row['native_measurement_boundary']={'process_repetition_policy':'separate_processes','source':str(summary)}
    if fault=='changed_plan':staging.write_text('{}')
    if fault:
        with pytest.raises(ValueError,match='source|staging plan'):
            subject._output_contract_source(row,{})
    else:
        result=subject._output_contract_source(row,{})
        assert result['source']==subject.bound(path)
        assert result['original_authority']=={'summary':subject.bound(summary),'staging_plan':subject.bound(staging)}


@pytest.mark.parametrize('fault',['missing','container_hash','selected_row_hash','model','stage','symlink','captured_conflict'])
def test_missing_or_detached_authority_is_rejected_before_staging(archive,fault):
    row,path,manifest=archive
    metadata=json.loads(manifest.read_text())
    if fault=='missing':path.unlink()
    elif fault=='container_hash':path.write_text('{}')
    elif fault in ('selected_row_hash','model','stage'):
        field={'selected_row_hash':'source_contract_sha256','model':'model_id','stage':'stage'}[fault]
        metadata['authoritative_output_contract_resolution'][field]='different'
        write(manifest,metadata)
    elif fault=='symlink':
        moved=path.with_name('actual.json');path.rename(moved);path.symlink_to(moved)
    else:
        captured=Path(row['native_source']).parent/'benchmark_set/output_contracts.json';write(captured,{})
    with pytest.raises(ValueError,match='declaration'):
        subject._output_contract_source(row,{})


@pytest.mark.parametrize('fault',['missing_source','source_changed','missing_remote_hash'])
def test_capture_plan_cannot_drop_output_declaration_closure(finite_plan,fault):
    if fault=='missing_source':finite_plan.pop('output_contract_source')
    elif fault=='source_changed':finite_plan['output_contract_source']['source']['sha256']='0'*64
    else:
        key=next(key for key in finite_plan['expected_remote_files'] if key.endswith('/output_contracts.json'))
        del finite_plan['expected_remote_files'][key]
    with pytest.raises(ValueError,match='declaration|closure'):
        subject.validate_capture_plan(finite_plan)


ACTUAL_ROOT=os.environ.get('ONNX_SPLITPOINT_NATIVE_REFERENCE_REVIEW_ROOT','')
LABELS=('deepx_yolov7_paper_b009','deepx_yolov7_paper_b044','deepx_yolov7_paper_b066','h10_yolo26s_b038')


@pytest.mark.skipif(not ACTUAL_ROOT,reason='Optional immutable existing Native archive review; never executes hardware')
@pytest.mark.parametrize('label',LABELS)
def test_actual_four_archived_sources_replay_normal_endpoint_builder_offline(label,tmp_path):
    plan=json.loads((Path(ACTUAL_ROOT)/label/'PREPARED.json').read_text())
    assert plan['controller_commit']=='4a0be2c3464a873c348783cf20010a526f911a7e'
    raw=json.loads(subject.checked(plan['historical_native_source']).read_text())
    binding=subject._output_contract_source(plan['source_row'],raw)
    manifest_path=subject.checked(binding['semantic_output_manifest'])
    manifest=json.loads(manifest_path.read_text())
    for tensor in manifest['outputs']:
        subject.checked({'path':str((manifest_path.parent/tensor['file']).resolve()),'sha256':tensor['sha256']})
    outputs=load_manifest_outputs(manifest_path,manifest)
    assert len(outputs)==len(manifest['outputs'])
    # Exact bytes at a new private root must retain the logical endpoint. This
    # uses genuine stored outputs, not a synthetic endpoint or inference run.
    staged=tmp_path/'private/output_contracts.json';staged.parent.mkdir()
    staged.write_bytes(subject.checked(binding['source']).read_bytes())
    declaration=load_authoritative_output_contract(staged,backend='tensorrt',
        model_id=plan['source_row']['model_id'],task='detection')
    assert declaration['authoritative_output_contract'] is True
    assert declaration['source_contracts_sha256']==binding['source']['sha256']
    endpoint=runtime_output_contract('detection',outputs,raw_fallback=False,declared_contract=declaration)
    original=raw['completion_execution_contract']
    assert endpoint['endpoint_contract_complete'] is True
    assert endpoint['endpoint_contract_hash']==original['source_endpoint']['endpoint_contract_hash']
    assert endpoint['tensor_signature']==original['source_endpoint']['tensor_signature']
    processor=original['processor_contract']
    rebuilt=build_detection_completion_execution_contract(model_id=plan['source_row']['model_id'],outputs=outputs,
        input_hw=processor['input_hw'],original_wh=processor['original_wh'],source_endpoint_contract=endpoint,
        preprocess={'mode':'letterbox','rgb':True,'pad_value':114})
    rebuilt=verify_detection_completion_execution_contract(rebuilt)
    assert rebuilt['source_endpoint']['output_endpoint_id']==original['source_endpoint']['output_endpoint_id']
    assert rebuilt['completion_mode']==original['completion_mode']
