from pathlib import Path
import tempfile, json
from onnx_splitpoint_tool.native_full_quality import normalise_evaluation_profile
from onnx_splitpoint_tool.run_modes import official_coco_policy_for_mode

def test_v60z_full_matrix_and_native_energy_acceptance():
    p={'hardware_run_profiles': {'Hailo-8 Full':True,'Hailo-8 -> TensorRT':True,'Hailo-10H Full':True,'Hailo-10H -> TensorRT':True,'DEEPX Full':True,'DEEPX -> TensorRT':True,'TensorRT Full':True},
       'native_producers': {'enabled':True,'energy':{'enabled':True}}, 'energy':{'enabled':False}, 'execution_preset':{'overrides':{}}}
    normalise_evaluation_profile(p)
    m=p['native_producers']['full_baselines']['backends_by_producer']
    assert m['hailo8']==['hailo8','tensorrt']
    assert m['hailo10h']==['hailo10h','tensorrt']
    assert m['deepx']==['deepx','tensorrt']
    assert p['native_producers']['energy']['mode']=='measure'
    assert p['energy']['enabled'] is False

def test_v60z_official_coco_mode_policy():
    assert official_coco_policy_for_mode('smoke')['enabled'] is False
    assert official_coco_policy_for_mode('standard')['enabled'] is True
    assert official_coco_policy_for_mode('final') == official_coco_policy_for_mode('standard')

def test_v60z_sources_contain_runtime_integration():
    root=Path(__file__).resolve().parents[1]/'onnx_splitpoint_tool'
    texts='\n'.join(p.read_text(errors='ignore') for p in root.rglob('*.py'))
    assert 'backends_by_producer' in texts
    assert 'requested_native_energy' in texts
    assert 'task_quality_loss_decomposition' in texts
    assert 'official_coco_index.json' in texts
