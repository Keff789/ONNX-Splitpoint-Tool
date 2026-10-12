"""Exact old DUT bundle remains mandatory for changed, unused UI/dispatch sources."""
import ast,copy,hashlib,json
from pathlib import Path
import pytest
from onnx_splitpoint_tool.workflow import generic_energy_parent
SOURCE=Path(generic_energy_parent.__file__)
RELS=['onnx_splitpoint_tool/gui/app.py','onnx_splitpoint_tool/gui/panels/panel_evaluation_workflow.py','onnx_splitpoint_tool/release_identity.py','onnx_splitpoint_tool/run_modes.py','onnx_splitpoint_tool/workflow/profile_options.py','onnx_splitpoint_tool/workflow/runner.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def make(tmp_path,rel):
 r=tmp_path/'report';w=tmp_path/'current';snapshot=tmp_path/'original';candidate=tmp_path/'reviewed.py'
 for root,content in [(w,'new controller'),(snapshot,'original source')]:
  (root/rel).parent.mkdir(parents=True,exist_ok=True);(root/rel).write_text(content)
 candidate.write_bytes((w/rel).read_bytes());freeze={'snapshot':str(snapshot),'runtime_files':[{'relative_path':rel,'sha256':sha(snapshot/rel)}]};final=r/'final_implementation_20261008/IMPLEMENTATION_FREEZE.json';write(final,freeze)
 binding={'path':str(final),'sha256':sha(final)};bundle=r/'generic_energy_extension/confirmation_final_20261008/runner-bundle.tar.gz';bundle.parent.mkdir(parents=True);bundle.write_bytes(b'original measured DUT bundle')
 write(bundle.parent/'BUNDLE_PREPARATION.json',{'final_source_freeze':binding,'normal_scope_validation':'pass','bundle':str(bundle),'bundle_sha256':sha(bundle)})
 supplement={'final_runtime_controller_changes':{'scoped_local_source_changes':True,'DUT_files_changed':False,'parent_freeze_sha256':sha(final),'original_measured_bundles_changed':False,'changes':[{'path':rel,'scope':'unexecuted_gui_or_dispatch','before_sha256':sha(snapshot/rel),'after_sha256':sha(w/rel),'candidate_path':str(candidate)}]}};write(r/'REPORTING_SOURCE_SUPPLEMENT.json',supplement)
 node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='check_final_local_sources');ns={'Path':Path,'R':r,'W':w,'sha':sha,'read':read};exec(compile(ast.Module(body=[node],type_ignores=[]),'source_gate','exec'),ns)
 return r,w,snapshot,candidate,freeze,bundle,supplement,ns['check_final_local_sources']
@pytest.mark.parametrize('rel',RELS)
def test_exact_reviewed_control_source_with_original_bundle_passes(tmp_path,rel):
 r,w,snapshot,candidate,freeze,bundle,supplement,check=make(tmp_path,rel);before=copy.deepcopy(supplement);check(freeze,w,frozen_bundle={'path':str(bundle),'sha256':sha(bundle)});assert read(r/'REPORTING_SOURCE_SUPPLEMENT.json')==before
@pytest.mark.parametrize('fault',['snapshot','candidate','current','before_sha','after_sha','wrong_scope','duplicate','dut_changed','original_bundle_changed','bundle_bytes','bundle_binding','no_bundle','freeze_binding','unlisted_runner'])
def test_original_and_explicit_local_bindings_remain_required(tmp_path,fault):
 rel=RELS[0];r,w,snapshot,candidate,freeze,bundle,supplement,check=make(tmp_path,rel);s=supplement['final_runtime_controller_changes'];b={'path':str(bundle),'sha256':sha(bundle)}
 if fault in ('snapshot','candidate','current'):{'snapshot':snapshot/rel,'candidate':candidate,'current':w/rel}[fault].write_text('unreviewed')
 elif fault in ('before_sha','after_sha'):s['changes'][0][fault+'256']='0'*64
 elif fault=='wrong_scope':s['changes'][0]['scope']='controller_or_postflight'
 elif fault=='duplicate':s['changes'].append(copy.deepcopy(s['changes'][0]))
 elif fault=='dut_changed':s['DUT_files_changed']=True
 elif fault=='original_bundle_changed':s['original_measured_bundles_changed']=True
 elif fault=='bundle_bytes':bundle.write_bytes(b'changed DUT runtime')
 elif fault=='bundle_binding':b['sha256']='0'*64
 elif fault=='no_bundle':b=None
 elif fault=='freeze_binding':s['parent_freeze_sha256']='0'*64
 elif fault=='unlisted_runner':
  new='onnx_splitpoint_tool/runners/generic_energy.py'
  for root in (w,snapshot):
   (root/new).parent.mkdir(parents=True,exist_ok=True);(root/new).write_bytes((root/rel).read_bytes())
  freeze['runtime_files'][0]['relative_path']=new;s['changes'][0]['path']=new;final=r/'final_implementation_20261008/IMPLEMENTATION_FREEZE.json';write(final,freeze);s['parent_freeze_sha256']=sha(final)
 write(r/'REPORTING_SOURCE_SUPPLEMENT.json',supplement)
 with pytest.raises(RuntimeError):check(freeze,w,frozen_bundle=b)
