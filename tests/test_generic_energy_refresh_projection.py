from pathlib import Path
import pytest
from onnx_splitpoint_tool.workflow import generic_energy_parent as g
from onnx_splitpoint_tool.workflow.energy_current_projection import _generic_energy_preparation as reader
@pytest.fixture
def sample(tmp_path):
 root=tmp_path/'parent';root.mkdir();collector=str(tmp_path/'scripts/energy_measurement_cli.py');old={'path':collector,'sha256':'a'*64};new={'path':collector,'sha256':'b'*64}
 original=root/'measurement/generic_energy_plan.json';g.write(original,{'collector_script':collector,'collector_script_sha256':old['sha256'],'duration_s':60,'calls':[{'repetition_index':i}for i in (1,2,3)],'config':{'tasks':1000}})
 base={'normal_energy_plan':g.file_binding(original),'controller_sources':[old],'performance_admission':{},'duration_s':60};a=root/'controller_binding_refresh/a';g.write(a/'parent_plan.json',base);g.write(a/'PREPARED.json',{'ok':True,'parent_plan_sha256':g.sha(a/'parent_plan.json')});g.write(a/'measurement/generic_energy_plan.json',g.read(original));middle={**base,'normal_energy_plan':g.file_binding(a/'measurement/generic_energy_plan.json')};g.write(root/'parent_plan.json',middle)
 def receipt(history,before,after,context,collector_refresh):
  return {'at':'2026-10-12T20:00:00+00:00','original_parent_plan':g.file_binding(history/'parent_plan.json'),'current_parent_plan':g.file_binding(root/'parent_plan.json'),'normal_energy_plan_before':before['normal_energy_plan'],'normal_energy_plan_after':after['normal_energy_plan'],'prepared_measured_execution_context_restored':context,'collector_script_binding_refreshed':collector_refresh,'normal_energy_plan_unchanged':False,'scientific_inputs_and_runtime_unchanged':True,'hardware_started':False,'budget_before_sha256':'c'*64,'budget_after_sha256':'c'*64,'refreshed_controller_paths':[collector]}
 g.write(a/'REFRESH_RECEIPT.json',receipt(a,base,middle,True,False));b=root/'controller_binding_refresh/b';g.write(b/'parent_plan.json',middle);g.write(b/'PREPARED.json',{'ok':True,'parent_plan_sha256':g.sha(b/'parent_plan.json')});after=g.read(a/'measurement/generic_energy_plan.json');after['collector_script_sha256']=new['sha256'];g.write(b/'measurement/generic_energy_plan.json',after);current={**middle,'normal_energy_plan':g.file_binding(b/'measurement/generic_energy_plan.json'),'controller_sources':[new]};g.write(root/'parent_plan.json',current);g.write(b/'REFRESH_RECEIPT.json',receipt(b,middle,current,False,True));g.write(root/'production-start.json',{'started_at':'2026-10-12T20:01:00+00:00'});return root,current,b

def test_both_retained_refreshes_are_normally_reconstructed(sample):
 root,current,b=sample;prepared,original,receipts=reader(root,current,g);assert prepared==b/'measurement/generic_energy_plan.json';assert original==root/'measurement/generic_energy_plan.json';assert len(receipts)==2;assert g.read(original)['collector_script_sha256']=='a'*64

@pytest.mark.parametrize('fault',['duration','config','capture_before','old_hash','new_hash','original_parent','not_zero_hardware','budget_hash'])
def test_unbound_or_scientific_changes_fail(sample,fault):
 root,current,b=sample;receipt=g.read(b/'REFRESH_RECEIPT.json')
 if fault in ('duration','config'):
  plan=g.read(b/'measurement/generic_energy_plan.json');plan['duration_s' if fault=='duration' else 'config']=1;g.write(b/'measurement/generic_energy_plan.json',plan);current['normal_energy_plan']=g.file_binding(b/'measurement/generic_energy_plan.json');g.write(root/'parent_plan.json',current);receipt['normal_energy_plan_after']=current['normal_energy_plan'];receipt['current_parent_plan']=g.file_binding(root/'parent_plan.json')
 elif fault=='capture_before':g.write(root/'production-start.json',{'started_at':'2026-10-12T19:59:00+00:00'})
 elif fault=='old_hash':
  old=g.read(b/'parent_plan.json');old['controller_sources'][0]['sha256']='d'*64;g.write(b/'parent_plan.json',old);receipt['original_parent_plan']=g.file_binding(b/'parent_plan.json');g.write(b/'PREPARED.json',{'ok':True,'parent_plan_sha256':g.sha(b/'parent_plan.json')})
 elif fault=='new_hash':
  current['controller_sources'][0]['sha256']='d'*64;g.write(root/'parent_plan.json',current);receipt['current_parent_plan']=g.file_binding(root/'parent_plan.json')
 elif fault=='original_parent':receipt['original_parent_plan']['sha256']='d'*64
 elif fault=='not_zero_hardware':receipt['hardware_started']=True
 else:receipt['budget_after_sha256']='wrong'
 g.write(b/'REFRESH_RECEIPT.json',receipt)
 with pytest.raises(ValueError):reader(root,current,g)
