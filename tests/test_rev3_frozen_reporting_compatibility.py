"""Original measured v2 arithmetic survives a reporting-only module change."""
from pathlib import Path
import ast,copy,json,os
import pytest
from onnx_splitpoint_tool import native_detection_postprocess as post

ARCHIVE=os.environ.get('ONNX_SPLITPOINT_REV3_ARCHIVE','')
R=Path(ARCHIVE).expanduser()
pytestmark=pytest.mark.skipif(not ARCHIVE,reason='Optional frozen-source replay requires ONNX_SPLITPOINT_REV3_ARCHIVE')

def contract():
    return json.loads((R/'energy/generic_completed/deepx_yolov7_b009_final_20261010/measurement/generic_energy_plan.json').read_text())['config']['completion_contract']['processor_contract']

def test_actual_frozen_v2_contract_survives_reporting_correction():
    original=contract();before=copy.deepcopy(original)
    assert post.verify_frozen_postprocess_contract(original)==original
    assert original==before

def decoded_contract():
    current=json.loads((R/'current_results/generic_current.json').read_text())['rows']
    saved=next(r for r in current if r['model_id']=='yolo11l' and r['case_id']=='b003'
               and r['setup_id']=='orin_nx_deepx_m1_01')
    plan=json.loads(Path(saved['execution_plan']).read_text())
    row=next(r for r in plan['rows'] if r['case_key']==saved['case_key'])
    return row['native_command_contract']['runtime_options']['completion_execution_contract']['processor_contract']

def test_actual_decoded_pre_nms_keeps_its_frozen_policy():
    original=decoded_contract();before=copy.deepcopy(original)
    assert post.verify_frozen_postprocess_contract(original)==original
    assert original==before

@pytest.mark.parametrize('change',['unknown_implementation','missing_policy','wrong_policy'])
def test_decoded_reporting_compatibility_keeps_implementation_and_score_gates(change):
    value=copy.deepcopy(decoded_contract())
    if change=='unknown_implementation':value['implementation_artifacts']['native_detection_postprocess']['sha256']='0'*64
    elif change=='missing_policy':value.pop('decoded_pre_nms_score_policy')
    else:value['decoded_pre_nms_score_policy']['policy_id']='unsupported'
    value['invariant_identity']=post.frozen_postprocess_invariant_identity(value)
    value['invariant_contract_sha256']=post.canonical_json_sha256(value['invariant_identity'])
    value['contract_sha256']=post.canonical_json_sha256({k:v for k,v in value.items()if k!='contract_sha256'})
    with pytest.raises(post.FrozenPostprocessError):post.verify_frozen_postprocess_contract(value)

@pytest.mark.parametrize('digest',['0'*64,post._V2776_NATIVE_DETECTION_POSTPROCESS_SHA256])
def test_unknown_and_pre_v2_implementations_cannot_borrow_v2_arithmetic(digest):
    value=contract();value['implementation_artifacts']['native_detection_postprocess']['sha256']=digest
    value['contract_sha256']=post.canonical_json_sha256({k:v for k,v in value.items()if k!='contract_sha256'})
    with pytest.raises(post.FrozenPostprocessError,match='yolov7_sigmoid_arithmetic_implementation_mismatch'):
        post.verify_frozen_postprocess_contract(value)

def test_original_measured_functions_are_unchanged():
    old=R/'final_implementation_20261008/source_snapshot/onnx_splitpoint_tool/native_detection_postprocess.py'
    def functions(path):
        return{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=functions(old),functions(Path(post.__file__))
    assert [k for k in sorted(a.keys()|b.keys())if a.get(k)!=b.get(k)]==[
        'build_normalized_detection_endpoint_attestation','verify_frozen_postprocess_contract']
