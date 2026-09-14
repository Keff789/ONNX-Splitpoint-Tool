from pathlib import Path
import yaml
from onnx_splitpoint_tool.native_full_quality import save_normalised_yaml, load_and_normalise_yaml

def test_manual_yaml_native_energy_true_survives_roundtrip(tmp_path: Path):
    p=tmp_path/'p.yaml'
    data={
      'hardware_run_profiles': {'Hailo-8 Full':True,'Hailo-8 -> TensorRT':True,'TensorRT Full':True},
      'native_producers': {'enabled':True,'energy':{'enabled':True}},
      'energy': {'enabled':False},
      'execution_preset': {'overrides': {'energy_enabled':False}},
    }
    p.write_text(yaml.safe_dump(data),encoding='utf-8')
    loaded=load_and_normalise_yaml(p)
    save_normalised_yaml(p,loaded)
    final=yaml.safe_load(p.read_text())
    assert final['native_producers']['energy']['enabled'] is True
    assert final['native_producers']['energy']['mode']=='measure'
    assert final['energy']['requested_native_energy'] is True
    assert final['execution_preset']['overrides']['energy_enabled'] is True
    assert final['native_producers']['full_baselines']['backends_by_producer']['hailo8']==['hailo8','tensorrt']
