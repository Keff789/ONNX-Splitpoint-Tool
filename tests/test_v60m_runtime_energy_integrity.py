
from __future__ import annotations
import os, tempfile
from pathlib import Path

def test_v60m_energy_master_override():
    from onnx_splitpoint_tool.v60m_policy import normalize_profile
    p={'models':[{'id':'m'}],'workflow':{},'energy_measurement':{'enabled':False},'native_producer':{'native_energy':True,'energy_mode':'measure'}}
    q,a=normalize_profile(p)
    assert q['native_producer']['native_energy'] is False
    assert q['native_producer']['energy_mode']=='disabled'

def test_v60m_final_forces_strict_integrity():
    from onnx_splitpoint_tool.v60m_policy import normalize_profile
    p={'models':[{'id':'m'}],'workflow':{},'campaign':{'mode':'final'},'integrity_policy':{'mode':'off'}}
    q,a=normalize_profile(p)
    assert q['integrity_policy']['effective_mode']=='strict'

def test_v60m_dev_hash_cache_same_digest():
    from onnx_splitpoint_tool.v60m_policy import cached_sha256, strict_sha256
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'x'; p.write_bytes(b'abc')
        os.environ['ONNX_SPLITPOINT_INTEGRITY_MODE']='fast'
        assert cached_sha256(p)==strict_sha256(p)
        assert cached_sha256(p)==strict_sha256(p)

def test_v60m_validation_defaults():
    from onnx_splitpoint_tool.v60m_policy import normalize_profile
    p={'models':[{'id':'m','evaluation_role':'development'}],'workflow':{}}
    q,a=normalize_profile(p)
    assert q['validation_execution']['cadence']=='once_per_artifact'
    assert q['validation_execution']['max_items']['detection']==200
    assert q['validation_execution']['max_items']['classification']==500

def test_v60m_final_validation_uncapped():
    from onnx_splitpoint_tool.v60m_policy import normalize_profile
    p={'models':[{'id':'m'}],'workflow':{},'campaign':{'mode':'final'}}
    q,a=normalize_profile(p)
    assert q['validation_execution']['max_items']['detection']==0
    assert q['validation_execution']['max_items']['classification']==0


def test_v60m_development_keeps_screening_validation_on_auto_bind():
    from onnx_splitpoint_tool.v60m_policy import normalize_profile, install_dataset_binding_guards
    p={'models':[{'id':'m','evaluation_role':'development'}],'workflow':{},'dataset_manifests':{'detection_validation':''}}
    normalize_profile(p)
    g={}
    def bind_dataset_registry(profile):
        profile['dataset_manifests']['detection_validation']='/full/coco_val_manifest.json'
        profile['dataset_manifests']['detection_calibration']='/calib/coco_train_manifest.json'
        return profile
    bind_dataset_registry.__module__='x'; g.update({'__name__':'x','bind_dataset_registry':bind_dataset_registry})
    install_dataset_binding_guards(g)
    q=g['bind_dataset_registry'](p)
    assert q['dataset_manifests']['detection_validation']==''
    assert q['dataset_manifests']['detection_calibration'].endswith('coco_train_manifest.json')

