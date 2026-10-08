import copy
import threading
import time
import pytest
from onnx_splitpoint_tool.runners.task_completion import measure_split_completion, validate_split_completion_duration

class Processor:
    completed_count = 0
    def __init__(self, delay=0, fail=None): self.delay=delay; self.fail=fail; self.seen=[]
    def process(self, outputs):
        if self.fail == self.completed_count: raise RuntimeError('task error')
        if self.delay: time.sleep(self.delay)
        self.seen.append(outputs['x']); self.completed_count += 1

@pytest.mark.parametrize('depth,producer_delay,consumer_delay,clone', [(1,0,.001,False),(8,.001,0,False),(2,0,0,True)])
def test_duration_lossless_count_and_drain(depth,producer_delay,consumer_delay,clone):
    produced=[];transferred=[];p=Processor(consumer_delay)
    def first():
        if producer_delay: time.sleep(producer_delay)
        produced.append(len(produced));return {'x':produced[-1]}
    def transfer(x):transferred.append(x['x']);return x
    r=measure_split_completion(first,transfer,lambda x:x,p,warmup=0,tasks=None,duration_s=.035,
        queue_depth=depth,clone_payload=dict if clone else None,timeout_s=2)
    assert produced==transferred==p.seen
    assert len(produced)==r['completed_work_units']==r['admitted_task_count']
    assert r['measurement_end_ns']>=r['admission_stopped_ns']>=r['admission_deadline_ns']
    assert r['measured_makespan_seconds']>=.035
    assert r['handoff_payload_copy']==clone
    validate_split_completion_duration(r,duration_s=.035)
    assert not any(t.name.startswith('generic-completion-') for t in threading.enumerate())

@pytest.mark.parametrize('failure',['stage1','transfer','stage2','processor'])
def test_errors_stop_both_workers_without_success(failure):
    p=Processor(.002,fail=2 if failure=='processor' else None);n=[0]
    def first():
        n[0]+=1
        if failure=='stage1' and n[0]==4:raise RuntimeError('first error')
        return {'x':n[0]}
    def transfer(x):
        if failure=='transfer' and x['x']==3:raise RuntimeError('transfer error')
        return x
    def second(x):
        if failure=='stage2' and x['x']==3:raise RuntimeError('second error')
        return x
    with pytest.raises(RuntimeError) as e:
        measure_split_completion(first,transfer,second,p,warmup=0,tasks=None,duration_s=.1,timeout_s=1,queue_depth=1)
    assert e.value.partial_report['status']=='failed'
    assert not any(t.name.startswith('generic-completion-') for t in threading.enumerate())

def test_fixed_count_contract_unchanged():
    p=Processor();r=measure_split_completion(lambda:{'x':1},lambda x:x,lambda x:x,p)
    assert r['requested_warmup_count']==r['warmup_completed_count']==100
    assert r['requested_task_count']==r['completed_task_count']==1000
    assert r['processor_count_delta']==1100
    assert 'admission_deadline_ns' not in r and 'measurement_mode' not in r

@pytest.mark.parametrize('change',[{'completed_work_units':1},{'admitted_task_count':0},{'energy_work_units_exact':False},{'requested_warmup_count':1},{'accepted_work_drained':False}])
def test_duration_evidence_rejects_contradictions(change):
    p=Processor(.001);r=measure_split_completion(lambda:{'x':1},lambda x:x,lambda x:x,p,warmup=0,tasks=None,duration_s=.01,timeout_s=1)
    r.update(change)
    with pytest.raises(ValueError):validate_split_completion_duration(r,duration_s=.01)

@pytest.mark.parametrize('kw',[{'duration_s':0},{'duration_s':float('nan')},{'duration_s':True},{'duration_s':.1,'tasks':1000},{'duration_s':.1,'warmup':1},{'duration_s':.1,'timeout_s':.01}])
def test_invalid_duration_budget(kw):
    args=dict(warmup=0,tasks=None,timeout_s=1);args.update(kw)
    with pytest.raises(ValueError):measure_split_completion(lambda:{'x':1},lambda x:x,lambda x:x,Processor(),**args)
