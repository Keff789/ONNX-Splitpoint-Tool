"""Actual105 own-H8 hint with known unexecuted backend metadata; pure relation."""
from copy import deepcopy
import hashlib,json
from pathlib import Path

import pytest
from onnx_splitpoint_tool.generic_split_manifest import validate_generic_split_manifest_staging

F=Path(__file__).parent/'fixtures/generic_split_manifest'

def actual_arguments():
    args=json.loads((F/'b105_import_hint_bindings.json').read_bytes())
    args['part1_build_receipt_bytes']=(F/'b105_import_hint_part1_build_receipt.json').read_bytes()
    return (F/'b105_import_hint_original.json').read_bytes(),(F/'b105_import_hint_staged.json').read_bytes(),args

def encoded(doc,args,*,original=False):
    raw=(json.dumps(doc,sort_keys=True)+'\n').encode();desc=dict(sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    args['original_source'if original else'staged_source'].update(desc)
    if original:args['row']['generic_artifact_binding']['artifacts']['boundary_metadata'].update(desc)
    return raw

def test_actual105_all_scientific_bindings_pass_and_unused_history_is_not_a_cache_claim():
    old,new,args=actual_arguments();before=deepcopy((old,new,args))
    proof=validate_generic_split_manifest_staging(old,new,**args)
    assert proof['scientific_manifest_fields_unchanged'] is True
    retained=proof['retained_historical_build_metadata'][0]
    assert retained['backend_metadata']=='hailo10'
    assert retained['staged_cache_key_status']=='unverified_unused_reuse_metadata'
    assert retained['staged_cache_key_verified'] is retained['current_compile_claimed'] is retained['current_reuse_claimed'] is False
    assert 'hailo10h'not in json.loads(new)['hailo']['hefs']
    assert (old,new,args)==before

def test_explicit_true_deferred_marker_remains_valid():
    old,new,args=actual_arguments();doc=json.loads(new);doc['hailo']['builds_deferred_until_cache_preflight']=True
    assert validate_generic_split_manifest_staging(old,encoded(doc,args),**args)['scientific_manifest_fields_unchanged']

@pytest.mark.parametrize('fault',['deferred_false','deferred_number','deferred_null','unknown_rootfield',
    'unused_negative','unused_diagnostic','unused_cold_build','unused_wrong_case','unused_wrong_graphname',
    'unused_receipt_path','unused_cache_conflict','full_alias_part1','full_alias_unknown',
    'measured_hef','measured_source','input_shape','boundary_shape'])
def test_optional_metadata_never_relaxes_used_identity_or_unknown_backend_scope(fault):
    old,new,args=actual_arguments();original=json.loads(old);staged=json.loads(new)
    record=staged['hailo']['hefs']['hailo10']['part1_build']
    if fault=='deferred_false':staged['hailo']['builds_deferred_until_cache_preflight']=False
    elif fault=='deferred_number':staged['hailo']['builds_deferred_until_cache_preflight']=1
    elif fault=='deferred_null':staged['hailo']['builds_deferred_until_cache_preflight']=None
    elif fault=='unknown_rootfield':staged['hailo']['unbound']=True
    elif fault=='unused_negative':record['ok']=False
    elif fault=='unused_diagnostic':record['diagnostic_only']=True
    elif fault=='unused_cold_build':record['cache_hit']=record['skipped']=False
    elif fault=='unused_wrong_case':record['source_onnx_path']=record['source_onnx_path'].replace('/b105/','/b106/')
    elif fault=='unused_wrong_graphname':record['compiler_onnx_path']=record['source_onnx_path'].replace('part1_b105','part1_b106')
    elif fault=='unused_receipt_path':record['build_receipt_path']='/foreign/receipt.json'
    elif fault=='unused_cache_conflict':record['cache_key']='f'*64
    elif fault=='full_alias_part1':original['hailo']['hefs']['hailo10h']['part1']='hailo/hailo10/part1/compiled.hef'
    elif fault=='full_alias_unknown':original['hailo']['hefs']['hailo10h']['unknown_scientific_field']=True
    elif fault=='measured_hef':args['row']['generic_artifact_binding']['artifacts']['part1_runtime']['sha256']='f'*64
    elif fault=='measured_source':args['producer_model']['part1_onnx_sha256']='f'*64
    elif fault=='input_shape':args['row']['input_binding']['prepared_input']['shape']=[320,640,3]
    else:args['boundary_inputs'][0]['shape']=[1,1,1,1]
    old=encoded(original,args,original=True);new=encoded(staged,args)
    with pytest.raises(ValueError):validate_generic_split_manifest_staging(old,new,**args)
