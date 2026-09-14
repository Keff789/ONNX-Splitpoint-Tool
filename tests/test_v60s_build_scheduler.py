from __future__ import annotations
import threading, time
from onnx_splitpoint_tool.build_scheduler import BuildScheduler, BuildTaskSpec, scheduler_config_from_mapping


def test_different_families_overlap():
    barrier=threading.Barrier(2); started=[]
    def work(name):
        started.append(name); barrier.wait(timeout=2); time.sleep(.05); return name
    with BuildScheduler(max_workers=2,cpu_tokens=8,family_limits={'hailo8':1,'deepx':1}) as s:
        a=s.submit(BuildTaskSpec('a','hailo8',2),work,'a')
        b=s.submit(BuildTaskSpec('b','deepx',2),work,'b')
        assert {a.result(timeout=3),b.result(timeout=3)}=={'a','b'}
    assert len(started)==2


def test_same_family_limit_serialises():
    active=0; peak=0; lock=threading.Lock()
    def work():
        nonlocal active,peak
        with lock: active+=1; peak=max(peak,active)
        time.sleep(.05)
        with lock: active-=1
    with BuildScheduler(max_workers=2,cpu_tokens=8,family_limits={'hailo8':1}) as s:
        fs=[s.submit(BuildTaskSpec(str(i),'hailo8',1),work) for i in range(2)]
        [f.result(timeout=3) for f in fs]
    assert peak==1


def test_scheduler_defaults():
    cfg=scheduler_config_from_mapping({},'standard')
    assert cfg['enabled'] and cfg['max_workers']==3
    assert cfg['family_limits']['hailo8']==1
