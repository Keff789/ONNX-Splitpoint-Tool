"""Legacy positive own-H8 receipt flags; pure metadata, no artifact execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool import generic_split_manifest as manifests
from tests.test_generic_split_manifest_h8_original_hint import actual_arguments, encode_receipt

F=Path(__file__).parent/'fixtures/generic_split_manifest'


def test_actual_245_missing_flags_retains_positive_original_reuse_and_bound_science():
    args=json.loads((F/'b245_legacy_hint_bindings.json').read_bytes())
    original=json.loads((F/'b245_legacy_hint_original.json').read_bytes())
    staged=json.loads((F/'b245_legacy_hint_staged.json').read_bytes())
    raw=(F/'b245_legacy_hint_part1_build_receipt.json').read_bytes()
    bindings={k:{n:args['original_artifacts'][k][n]for n in ('sha256','size_bytes')}for k in manifests._ROLES}
    feed=args['row']['input_binding'];before=deepcopy((args,original,staged,raw))
    proof=manifests._h8_receipt_input_authority(raw,bindings,original,staged,
        feed['prepared_input'],feed['prepared_input_contract'],model='yolo26m',case='b245',
        full_model_sha256=args['full_model_sha256'])
    assert proof['legacy_publication_flags_absent'] is True
    assert proof['positive_original_build_for_legacy_flags']['historical_reuse_observation'] is True
    assert proof['hef']==bindings['part1_runtime']
    assert (args,original,staged,raw)==before


def legacy_arguments():
    original,staged,args=actual_arguments()
    receipt=json.loads(args['part1_build_receipt_bytes'])
    receipt.pop('publish_artifacts');receipt.pop('diagnostic_only');encode_receipt(receipt,args)
    return original,staged,args


def test_missing_flags_allow_existing_bound_positive_cold_build_without_receipt_rewrite():
    original,staged,args=legacy_arguments();before=deepcopy(args)
    proof=manifests.validate_generic_split_manifest_staging(original,staged,**args)
    assert proof['native_h8_input_authority']['legacy_publication_flags_absent'] is True
    assert proof['scientific_manifest_fields_unchanged'] is True
    assert args==before


@pytest.mark.parametrize('flags', [dict(publish_artifacts=False),dict(diagnostic_only=True),
    dict(publish_artifacts=True),dict(diagnostic_only=False),dict(publish_artifacts=True,diagnostic_only=True),
    dict(publish_artifacts=False,diagnostic_only=False),dict(publish_artifacts=1,diagnostic_only=0),
    dict(publish_artifacts=None,diagnostic_only=None)])
def test_explicit_negative_diagnostic_partial_or_untyped_flags_reject(flags):
    original,staged,args=legacy_arguments();receipt=json.loads(args['part1_build_receipt_bytes']);receipt.update(flags);encode_receipt(receipt,args)
    with pytest.raises(ValueError):manifests.validate_generic_split_manifest_staging(original,staged,**args)


@pytest.mark.parametrize('fault',['negative','mapping_failed','diagnostic','wrong_source','wrong_hef','wrong_cache','wrong_shape'])
def test_missing_flags_never_substitute_for_positive_original_or_exact_science(fault):
    original,staged,args=legacy_arguments()
    if fault in {'negative','mapping_failed','diagnostic'}:
        doc=json.loads(original);build=doc['hailo']['hefs']['hailo8']['part1_build']
        if fault=='negative':build['ok']=False
        elif fault=='mapping_failed':build['mapping_failed']=True
        else:build['diagnostic_only']=True
        original=(json.dumps(doc,sort_keys=True)+'\n').encode();args['original_source'].update(sha256=hashlib.sha256(original).hexdigest(),size_bytes=len(original))
        args['row']['generic_artifact_binding']['artifacts']['boundary_metadata'].update(sha256=hashlib.sha256(original).hexdigest(),size_bytes=len(original))
    else:
        receipt=json.loads(args['part1_build_receipt_bytes'])
        if fault=='wrong_source':receipt['source_onnx_sha256']='f'*64
        elif fault=='wrong_hef':receipt['hef_sha256']='f'*64
        elif fault=='wrong_cache':receipt['cache_key']='f'*64
        else:receipt['cache_payload']['net_input_shapes']=[1,3,320,640]
        encode_receipt(receipt,args)
    with pytest.raises(ValueError):manifests.validate_generic_split_manifest_staging(original,staged,**args)


def actual_245_arguments():
    args=json.loads((F/'b245_legacy_hint_bindings.json').read_bytes())
    args['part1_build_receipt_bytes']=(F/'b245_legacy_hint_part1_build_receipt.json').read_bytes()
    return (F/'b245_legacy_hint_original.json').read_bytes(),(F/'b245_legacy_hint_staged.json').read_bytes(),args


def test_actual_245_full_relation_retains_unused_new_key_only_as_unverified_metadata():
    original,staged,args=actual_245_arguments();before=deepcopy((original,staged,args))
    proof=manifests.validate_generic_split_manifest_staging(original,staged,**args)
    retained=proof['retained_historical_build_metadata'][0]
    assert retained['backend_metadata']=='hailo10'
    assert retained['staged_cache_key_status']=='unverified_unused_metadata'
    assert retained['staged_cache_key_verified'] is False
    assert retained['current_reuse_claimed'] is retained['current_compile_claimed'] is False
    assert proof['native_h8_input_authority']['legacy_publication_flags_absent'] is True
    assert proof['scientific_manifest_fields_unchanged'] is True
    assert (original,staged,args)==before


@pytest.mark.parametrize('fault',['known_conflict','known_null','original_negative','original_foreign_source',
                                 'staged_negative','staged_diagnostic','invalid_extra_key','wrong_H8_hef'])
def test_legacy_unused_key_never_overrides_known_or_negative_authority(fault):
    original,staged,args=actual_245_arguments();old=json.loads(original);new=json.loads(staged)
    ob=old['hailo']['hefs']['hailo10']['part1_build'];nb=new['hailo']['hefs']['hailo10']['part1_build']
    if fault=='known_conflict':ob['cache_key']='f'*64
    elif fault=='known_null':ob['cache_key']=None
    elif fault=='original_negative':ob['ok']=False
    elif fault=='original_foreign_source':ob['source_onnx_path']='/foreign/b245/foreign.onnx'
    elif fault=='staged_negative':nb['ok']=False
    elif fault=='staged_diagnostic':nb['diagnostic_only']=True
    elif fault=='invalid_extra_key':nb['cache_key']='unbound'
    else:args['row']['generic_artifact_binding']['artifacts']['part1_runtime']['sha256']='f'*64
    original=(json.dumps(old,sort_keys=True)+'\n').encode();args['original_source'].update(sha256=hashlib.sha256(original).hexdigest(),size_bytes=len(original))
    args['row']['generic_artifact_binding']['artifacts']['boundary_metadata'].update(sha256=hashlib.sha256(original).hexdigest(),size_bytes=len(original))
    staged=(json.dumps(new,sort_keys=True)+'\n').encode();args['staged_source'].update(sha256=hashlib.sha256(staged).hexdigest(),size_bytes=len(staged))
    with pytest.raises(ValueError):manifests.validate_generic_split_manifest_staging(original,staged,**args)


@pytest.mark.parametrize('flags',[dict(diagnostic_only=True),dict(publish_artifacts=False),
                                dict(diagnostic_only=0),dict(publish_artifacts=1)])
def test_original_cold_diagnostic_or_unpublished_record_cannot_authorize_missing_flags(flags):
    original,staged,args=legacy_arguments();old=json.loads(original);old['hailo']['hefs']['hailo8']['part1_build'].update(flags)
    feed=args['row']['input_binding'];bindings={k:{n:args['original_artifacts'][k][n]for n in ('sha256','size_bytes')}for k in manifests._ROLES}
    with pytest.raises(ValueError,match='unpublished or diagnostic'):
        manifests._h8_receipt_input_authority(args['part1_build_receipt_bytes'],bindings,old,json.loads(staged),
            feed['prepared_input'],feed['prepared_input_contract'],model='yolo26m',case='b140',full_model_sha256=args['full_model_sha256'])
