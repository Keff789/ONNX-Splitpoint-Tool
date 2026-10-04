"""Explicit bounded reentry keeps historical start/terminal evidence intact."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import yolo_augmentation as workflow
from onnx_splitpoint_tool.workflow import yolo_augmentation_reentry as reentry
from tests.test_yolo_augmentation_workflow import frozen, options


def test_reentry_requires_distinct_output_before_original_validation_or_owner_probe(frozen, monkeypatch):
    path, plan = frozen
    output = Path(plan['output_root'])
    (output / 'plan/workflow-options.json').write_text(json.dumps(options(plan)))
    (output / 'production-start.json').write_text('previous failed invocation')
    monkeypatch.setattr(reentry, 'validate_augmentation_reentry', lambda *a: pytest.fail('wrong output admitted'))
    with pytest.raises(ValueError, match='new attempt directory'):
        workflow.execute(path, resume=True)
    assert (output / 'production-start.json').read_text() == 'previous failed invocation'


@pytest.mark.parametrize('relative', ['outside', 'attempts', 'attempts/nested/attempt'])
def test_attempt_output_cannot_redirect_writers(frozen, relative):
    _, plan = frozen
    with pytest.raises(ValueError, match='direct child'):
        workflow.execution_output_path(plan, Path(plan['output_root']) / relative)


def test_explicit_attempt_preserves_old_logs_and_reports_concrete_stage_failure(frozen, monkeypatch):
    path, plan = frozen
    output = Path(plan['output_root'])
    (output / 'plan/workflow-options.json').write_text(json.dumps(options(plan)))
    originals = ['production-start.json', 'terminal-status.json', 'console.log', 'exitcode.txt']
    for name in originals:
        (output / name).write_text('historical ' + name)
    attempt = output / 'attempts/quality-reentry'
    monkeypatch.setattr(reentry, 'validate_augmentation_reentry', lambda *a: {'predecessor_error': 'original quality nonstart'})
    monkeypatch.setattr(workflow.completion, 'bound_targets', lambda *a: {})
    monkeypatch.setattr(workflow.completion, 'observe_owners', lambda *a: None)
    error = "augmentation startcase quality release failed: no completed Native split quality binding for ('yolo11l', 'b120', 'hailo8_to_trt')"
    class Runner:
        def __init__(self, opts, *args, **kwargs):
            assert opts.resume and opts.resume_missing
            assert kwargs['augmentation_resume'] is True
            assert kwargs['execution_output'] == attempt
        def run(self):
            return SimpleNamespace(status='partial', ok=False, stage_results=[{'status': 'failed', 'notes': error}])
    monkeypatch.setattr(workflow, 'YOLOAugmentationRunner', Runner)
    from onnx_splitpoint_tool import yolo_augmentation_reporting
    monkeypatch.setattr(yolo_augmentation_reporting, 'write_augmentation_reports', lambda *a, **k: {'new_completion_pair_count': 0})
    assert workflow.execute(path, resume=True, attempt_dir=attempt) == 1
    terminal = json.loads((attempt / 'terminal-status.json').read_text())
    assert terminal['error'] == error
    assert terminal['automatic_restart'] is False
    assert json.loads((attempt / 'production-start.json').read_text())['explicit_reentry'] is True
    assert all((output / name).read_text() == 'historical ' + name for name in originals)
    with pytest.raises(FileExistsError):
        workflow.execute(path, resume=True, attempt_dir=attempt)


def test_patch_compatibility_is_not_a_general_2912_release():
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    normal = object.__new__(EvaluationWorkflowRunner)
    assert normal._supplement_compatibility_reviewed({'tool_version':'2.91.1'})
    assert not normal._supplement_compatibility_reviewed({'tool_version':'2.91.2'})
    augmentation = object.__new__(workflow.YOLOAugmentationRunner)
    augmentation.augmentation_resume = False
    assert not augmentation._supplement_compatibility_reviewed({'tool_version':'2.91.2'})
    augmentation.augmentation_resume = True
    assert not augmentation._supplement_compatibility_reviewed({'tool_version':'2.92.0'})


def test_augmentation_patch_admission_requires_real_bound_predecessor(frozen):
    path, plan = frozen
    augmentation = object.__new__(workflow.YOLOAugmentationRunner)
    augmentation.augmentation_resume = True
    augmentation.run_dir = Path(plan['augmentation_run'])
    augmentation.augmentation_plan_path = path
    with pytest.raises(ValueError, match='missing_original'):
        augmentation._supplement_compatibility_reviewed({'tool_version':'2.91.2'})


@pytest.fixture
def completion_sources(tmp_path):
    import hashlib
    run = tmp_path / 'run'; run.mkdir()
    old = run / 'old-report.json'; old.write_text('[{"timing":123.45,"requests":{}}]')
    new = run / 'new-report.json'; new.write_text('[{"timing":0.01,"requests":{"new":1}}]')
    request = run / 'new-request.json'; request.write_text('{"new_candidate":true}')
    sha = hashlib.sha256(request.read_bytes()).hexdigest()
    identity = dict(model_id='yolo11l',case_id='b120',setup_id='orin_nx_hailo8_01')
    row = dict(identity,source_path=str(old),total_latency_ms=123.45,supplement_quality_source={
        'source_request':request.name,'source_request_sha256':sha,
        'candidate_report_source':str(new),'preserved_performance_source':str(old)})
    normalized = run / 'models/yolo11l/benchmark_results/normalized_results.json'
    normalized.parent.mkdir(parents=True)
    normalized.write_text(json.dumps({'results':[row]}))
    pair = dict(identity,generic_source_path=str(old),original_source=str(old),generic_cycle_ms=123.45)
    native = dict(identity,source_request_sha256=sha)
    return run,old,new,request,normalized,row,pair,native


def test_completion_uses_new_candidate_recipe_without_replacing_historical_proxy(completion_sources):
    run,old,new,request,normalized,row,pair,native=completion_sources
    before={p:p.read_bytes() for p in (old,new,request,normalized)}
    selected=workflow._completion_source_pairs(run,[pair],[native])[0]
    assert selected['original_source']==str(new)
    assert selected['completion_generic_source_path']==str(new)
    assert selected['generic_source_path']==str(old)
    assert selected['generic_cycle_ms']==123.45 and pair['original_source']==str(old)
    assert {p:p.read_bytes() for p in before}==before


@pytest.mark.parametrize('damage',['native_request','request_bytes','foreign_candidate','lost_performance','duplicate'])
def test_completion_source_handoff_rejects_unbound_projection(completion_sources,damage,tmp_path):
    run,old,new,request,normalized,row,pair,native=completion_sources
    if damage=='native_request':native['source_request_sha256']='f'*64
    elif damage=='request_bytes':request.write_text('{"changed":true}')
    elif damage=='foreign_candidate':
        outside=tmp_path/'foreign.json';outside.write_text('{}')
        row['supplement_quality_source']['candidate_report_source']=str(outside)
    elif damage=='lost_performance':row['supplement_quality_source']['preserved_performance_source']='other'
    normalized.write_text(json.dumps({'results':[row,row] if damage=='duplicate' else [row]}))
    with pytest.raises(ValueError):workflow._completion_source_pairs(run,[pair],[native])
