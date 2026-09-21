"""R9J accuracy contract; all inference inputs are recorded semantic outputs."""
from copy import deepcopy
import json
import math
import numpy as np
import pytest
from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY, assess_accuracy
from onnx_splitpoint_tool.quality_service import QualityEvaluationRequest, prepare_evaluation, _evaluate_payload, deterministic_resample_plan

@pytest.mark.parametrize('ref,cand,cls,loss',[(.8,.76,'reference_close',.05),(.4,.38,'reference_close',.05),(.8,.78,'reference_close',.025),(.2,.18,'accuracy_loss',.1),(.8,.9,'reference_close',-.125),(.4,0,'accuracy_loss',1)])
def test_relative_point_boundary(ref,cand,cls,loss):
    a=assess_accuracy(ref,cand)
    assert a['accuracy_class']==cls and a['relative_loss']==pytest.approx(loss)
    assert a['absolute_loss_pp']==pytest.approx(100*(ref-cand))
    assert a['technical_fail'] is False

@pytest.mark.parametrize('interval,expected', [([.02,.06],'borderline'),([-.01,.12],'inconclusive'),([.03,.08],'borderline'),([.02,.07],'borderline'),([0,.05],'supported'),([.051,.09],'inconclusive')])
def test_ci_boundaries(interval,expected):
    a=assess_accuracy(.5,.48,interval)
    assert a['uncertainty']==expected
    if interval[0]>.05:assert a['uncertainty_reason']=='point_interval_disagreement'

@pytest.mark.parametrize('ref,cand',[(float('nan'),.5),(.5,float('inf')),(-.1,0),(1,1.1),(True,.5)])
def test_invalid_metric_is_technical_error(ref,cand):
    with pytest.raises(ValueError):assess_accuracy(ref,cand)

def test_zero_reference_and_missing_ci():
    assert assess_accuracy(0,0)['uncertainty']=='not_estimable'
    assert assess_accuracy(0,.1)['accuracy_class'] is None
    assert assess_accuracy(.5,.4)['uncertainty']=='not_estimated'

def request(ref,cand,*,repetitions=100):
    refs=[{'image_id':str(i),'reference':{'top1_hit':bool(v),'top5_hit':True}} for i,v in enumerate(ref)]
    cands=[{'image_id':str(i),'candidate':{'top1_hit':bool(v),'top5_hit':bool(v)}} for i,v in enumerate(cand)]
    return QualityEvaluationRequest(refs,cands,[],{'task':'classification','reporting_policy':deepcopy(DEFAULT_REPORTING_POLICY),'guardrails':{'top5_accuracy_margin':.01}},repetitions,20260710,.95,.01,
        evaluator_factory='onnx_splitpoint_tool.quality_metrics:classification_quality_evaluator',reference_prediction_field='reference',candidate_prediction_field='candidate',reference_identity='bound-test-cpu')

def result():
    _,p=prepare_evaluation(request([1]*80+[0]*20,[1]*60+[0]*40))
    return _evaluate_payload(p)

def test_real_evaluator_ratio_bootstrap_keeps_fast_fail_samples_and_secondary_warning():
    ref=np.array([1]*80+[0]*20);cand=np.array([1]*60+[0]*40)
    _,payload=prepare_evaluation(request(ref,cand))
    ref=np.array([r["reference"]["top1_hit"] for r in payload["reference_records"]]);cand=np.array([r["candidate"]["top1_hit"] for r in payload["candidate_records"]])
    r=_evaluate_payload(payload);plan=deterministic_resample_plan(image_count=100,repetitions=100,seed=20260710)
    ratios=np.array([(ref[ix].mean()-cand[ix].mean())/ref[ix].mean() for ix in plan])
    assert r['accuracy_assessment']['relative_loss_ci']==pytest.approx(np.quantile(ratios,[.025,.975]))
    assert r['primary']['bootstrap_repetitions']==100
    assert r['primary']['bootstrap_skipped_reason']==''
    assert r['accuracy_warnings']==['top5_accuracy_relative_loss_gt_5pct']
    assert r['decision']=='accuracy_loss'

def test_no_undefined_draw_is_discarded():
    _,p=prepare_evaluation(request([1,0],[0,0],repetitions=20));r=_evaluate_payload(p)
    assert r['relative_loss_undefined_draws']>0
    assert r['accuracy_assessment']['relative_loss_ci'] is None
    assert r['accuracy_assessment']['uncertainty']=='not_estimable'

def test_wrong_reference_join_and_missing_quality_remain_failed():
    from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
    r=result();r.update(canonical_reference='management_cpu_ort_full_onnx',technical_status='completed')
    policy={'reporting_policy':deepcopy(DEFAULT_REPORTING_POLICY)}
    base={'task':'classification','buildable':True,'runtime_ok':True,'contract_consistent':True,'task_quality_gate':r}
    row=apply_accuracy_gate_to_row(deepcopy(base),policy)
    assert row['technical_status']=='ok' and row['accuracy_class']=='accuracy_loss'
    assert row['eligible_for_ranking'] and row['pareto_eligible']
    for damage in ('reference','missing','shape','runtime'):
        bad=deepcopy(base)
        if damage=='reference':bad['task_quality_gate']['canonical_reference']='worst_trt'
        if damage=='missing':bad.pop('task_quality_gate')
        if damage=='shape':bad['contract_consistent']=False
        if damage=='runtime':bad['runtime_ok']=False
        assert apply_accuracy_gate_to_row(bad,policy)['technical_status']=='failed'

def test_legacy_fast_fail_does_not_acquire_a_ratio_ci():
    from onnx_splitpoint_tool.quality_result_contract import project_quality_result
    old={'decision':'fail','primary':{'decision':'fail','delta':-.3,'ci_low':-.3,'ci_high':-.3,'bootstrap_repetitions':0,'bootstrap_skipped_reason':'point_estimate_below_non_inferiority_margin'}}
    r=project_quality_result(old)
    assert r['primary']['ci_low'] is None and r['decision']=='fail'
    assert 'accuracy_assessment' not in r and old['primary']['ci_low']==-.3

def test_coco_population_is_official_and_resampled_not_mean_image_ap():
    from onnx_splitpoint_tool.quality_metrics import detection_quality_evaluator
    box={'class_id':0,'x1':0,'y1':0,'x2':10,'y2':10,'score':.9}
    gt=[{'ground_truth':[box]},{'ground_truth':[box]}]
    ref=[{'reference':[box]},{'reference':[box]}];cand=[{'candidate':[box]},{'candidate':[]}]
    e=detection_quality_evaluator(ref,cand,gt,{'metric_gate_config':{'reporting_policy':DEFAULT_REPORTING_POLICY}})
    all_=e.evaluate(np.array([1,1]));perfect=e.evaluate(np.array([2,0]));empty=e.evaluate(np.array([0,2]))
    assert all_['primary']['reference']==pytest.approx(1)
    assert all_['primary']['candidate']==pytest.approx(51/101)
    assert perfect['primary']['candidate']==pytest.approx(1)
    assert empty['primary']['candidate']==0
    assert set(all_['guardrails'])=={'ap50','ap75'}

def test_projection_lifecycle_csv_and_real_tk(tmp_path):
    """HOST executes this Tk case; local tests select the CPU-only cases."""
    import tkinter as tk
    from onnx_splitpoint_tool.workflow.scientific_reporting import project_central_quality_status
    from onnx_splitpoint_tool.quality_lifecycle import summarize_requests
    from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI
    r=result();r.update(model_id='resnet50',task='classification',backend='deepx',variant='full',case_id='full')
    projection=project_central_quality_status({'status':'ok','request_count':1,'results':[r]})
    assert summarize_requests([r])['technical_failed_count']==0
    rows=projection['results']
    out=tmp_path/'reports/scientific';out.mkdir(parents=True)
    (out/'central_quality_results.json').write_text(json.dumps(rows))
    app=SplitPointAnalyserGUI();app.withdraw()
    try:
        app._select_main_tab('evaluation_workflow');app._eval_workflow_render_result({'run_dir':str(tmp_path),'status':'ok'});app.update_idletasks()
        def texts(w):
            for child in w.winfo_children():
                if isinstance(child,tk.Text):yield child.get('1.0','end')
                yield from texts(child)
        rendered='\n'.join(texts(app.panel_frames['evaluation_workflow']))
        assert 'Genauigkeitsverlust' in rendered and 'statistisch gestützt' in rendered
    finally:app.destroy()


def test_normal_loader_policy_reaches_both_generic_producer_hashes():
    import ast
    from pathlib import Path
    from test_v27926_deepx_full_decoded_pre_nms import _suite
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
    from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
    root=Path(__file__).resolve().parents[1]
    profile=load_evaluation_profile('/home/kmika/.local/share/onnx-splitpoint-codex/v283_R9J_20260919_211316_8daof7vt/profiles/A.yaml').raw_profile
    policy=profile['quality_gate'];expected=AccuracyGatePolicy.from_mapping(policy).sha256()
    suite=_suite();deepx=suite._deepx_normalize_quality_policy(policy)
    path=root/'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'
    module=__import__('types').ModuleType('r9j_generated');module.__file__=str(path)
    __import__('sys').modules[module.__name__]=module
    exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
    generic=module._load_task_quality_policy(policy)
    for p in (deepx,generic):
        assert p['reporting_policy']==DEFAULT_REPORTING_POLICY
        assert p['policy_sha256']==expected


def test_policy_mismatch_is_technical_despite_valid_accuracy():
    from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
    r=result();r.update(canonical_reference='management_cpu_ort_full_onnx',technical_status='completed',policy_sha256='0'*64)
    row=apply_accuracy_gate_to_row({'task':'classification','buildable':True,'runtime_ok':True,'contract_consistent':True,'task_quality_gate':r}, {'reporting_policy':DEFAULT_REPORTING_POLICY})
    assert row['technical_status']=='failed' and row['accuracy_gate_policy_match'] is False


def test_actual_report_writer_preserves_class_csv_and_frozen_policy(tmp_path):
    import csv
    from onnx_splitpoint_tool.workflow.scientific_reporting import _write_reports, project_central_quality_status
    from onnx_splitpoint_tool.quality_lifecycle import summarize_requests
    r=result();r.update(model_id='resnet50',task='classification',backend='deepx',variant='full',case_id='full')
    projection=project_central_quality_status({'status':'ok','request_count':1,'results':[r]})
    assert summarize_requests([r])['technical_failed_count']==0
    root=tmp_path/'scientific'
    _write_reports(root,{'run_id':'fixture','rows':[],'central_quality_results':projection['results'],
        'central_quality_reporting':projection,'quality_reporting_policy':DEFAULT_REPORTING_POLICY})
    assert json.loads((root/'quality_policy.json').read_text())==DEFAULT_REPORTING_POLICY
    stored=json.loads((root/'central_quality_results.json').read_text())
    assert stored[0]['accuracy_class']=='accuracy_loss'
    paths=list(root.rglob('*task_quality*.csv'))
    assert paths
    assert any('accuracy_loss' in p.read_text() and 'accuracy_relative_loss' in p.read_text() for p in paths)
    assert json.loads((root/'sentinel_coverage.json').read_text())['evaluated_images']==0


def test_cross_runner_rejects_raw_p2_and_logits_as_completed_rate():
    from onnx_splitpoint_tool.workflow.cross_runner_reporting import _completed_task_measurement, _quality_decision
    for endpoint in ('p2_output','classification_logits','raw_model_outputs'):
        assert not _completed_task_measurement({'measurement_endpoint':endpoint,'postprocess_completion_verified':True,'postprocess_completed_frames':100})
    assert _completed_task_measurement({'measurement_endpoint':'completed_detection','postprocess_completion_verified':True,'postprocess_completed_frames':100})
    assert not _completed_task_measurement({'measurement_endpoint':'completed_detection'})
    assert _quality_decision({'accuracy_assessment':assess_accuracy(.8,.6),'accuracy_gate_pass':True})=='accuracy_loss'


@pytest.mark.parametrize('field',['output_shape_match','tensor_ok','quality_identity_valid'])
def test_high_accuracy_cannot_override_explicit_technical_negative(field):
    from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
    _,p=prepare_evaluation(request([1]*32,[1]*32));r=_evaluate_payload(p)
    r.update(canonical_reference='management_cpu_ort_full_onnx',technical_status='completed')
    row=apply_accuracy_gate_to_row({'task':'classification','buildable':True,'runtime_ok':True,'structural_contract_pass':True,field:False,'task_quality_gate':r}, {'reporting_policy':DEFAULT_REPORTING_POLICY})
    assert row['technical_status']=='failed' and not row['eligible_for_ranking']


def test_actual_historical_style_ranking_does_not_mix_raw_generic_with_completed_native(tmp_path):
    from test_v276_cross_runner_reporting import _build_bounded_24_fixture
    from onnx_splitpoint_tool.workflow.cross_runner_reporting import compute_cross_runner_report
    rows=_build_bounded_24_fixture(tmp_path,tmp_path)
    for row in rows:row['measurement_endpoint']='p2_output'
    r=compute_cross_runner_report(tmp_path,rows)
    assert r['technical_pair_count']==0
    assert all(p['technical_transfer_eligibility_checks']['generic_completed_task_measured'] is False for p in r['pairs'])
    assert all(g['technical_spearman_rho'] is None for g in r['groups'])
