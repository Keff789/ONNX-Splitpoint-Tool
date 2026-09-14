from __future__ import annotations
import concurrent.futures, contextvars, os, threading, time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


def _result_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def semantic_result_status(value: Any) -> str:
    """Classify a returned build result without changing its value.

    Scheduler invocation status (returned/raised) is deliberately separate
    from the semantic compiler outcome.  A structured returned failure must
    never be recorded as ``status=ok``.
    """

    raw_status = str(_result_field(value, "semantic_status", "") or
                     _result_field(value, "status", "") or "").strip().lower()
    if bool(_result_field(value, "timed_out", False)) or raw_status in {"timeout", "timed_out"}:
        return "timeout"
    if str(_result_field(value, "unsupported_reason", "") or "").strip() or raw_status == "unsupported":
        return "unsupported"
    ok = _result_field(value, "ok", None)
    if ok is True or raw_status in {"ok", "success", "succeeded", "completed", "pass", "passed"}:
        return "success"
    if ok is False or raw_status in {"failed", "failure", "error", "partial"}:
        return "failed"
    # Plain legacy return values have no failure contract. Preserve historical
    # success semantics, but identify the classification source in the event.
    return "success"


@dataclass(frozen=True)
class BuildTaskSpec:
    name:str; family:str; cpu_tokens:int=1; ram_mb:int=0; metadata:Mapping[str,Any]=field(default_factory=dict)

class _ResourcePool:
    def __init__(self,cpu_tokens:int,ram_mb:int)->None:
        self.cpu_total=max(1,int(cpu_tokens)); self.ram_total=max(0,int(ram_mb)); self.cpu_free=self.cpu_total; self.ram_free=self.ram_total; self.cond=threading.Condition()
    def impossible_reason(self,cpu:int,ram:int)->str:
        cpu=max(1,int(cpu)); ram=max(0,int(ram))
        if cpu>self.cpu_total:
            return f"cpu_request_exceeds_pool:{cpu}>{self.cpu_total}"
        if self.ram_total>0 and ram>self.ram_total:
            return f"ram_request_exceeds_pool:{ram}>{self.ram_total}"
        return ""
    def acquire(self,cpu:int,ram:int)->None:
        cpu=max(1,int(cpu)); ram=max(0,int(ram))
        impossible=self.impossible_reason(cpu,ram)
        if impossible:
            raise ValueError(impossible)
        with self.cond:
            while self.cpu_free<cpu or (self.ram_total>0 and self.ram_free<ram): self.cond.wait(timeout=.5)
            self.cpu_free-=cpu
            if self.ram_total>0: self.ram_free-=min(ram,self.ram_total)
    def release(self,cpu:int,ram:int)->None:
        cpu=max(1,min(int(cpu),self.cpu_total)); ram=max(0,int(ram))
        with self.cond:
            self.cpu_free=min(self.cpu_total,self.cpu_free+cpu)
            if self.ram_total>0: self.ram_free=min(self.ram_total,self.ram_free+min(ram,self.ram_total))
            self.cond.notify_all()

class BuildScheduler:
    def __init__(self,*,max_workers:int=3,cpu_tokens:int|None=None,ram_mb:int=0,family_limits:Optional[Mapping[str,int]]=None,log:Optional[Callable[[str],None]]=None)->None:
        detected=os.cpu_count() or 4; self.max_workers=max(1,int(max_workers)); self.pool=_ResourcePool(cpu_tokens or max(1,detected-2),ram_mb)
        limits={str(k):max(1,int(v)) for k,v in dict(family_limits or {}).items()}; self.family_semaphores={k:threading.Semaphore(v) for k,v in limits.items()}
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers,thread_name_prefix="osp-build"); self.log=log or (lambda _m:None); self._lock=threading.Lock(); self._events:list[dict[str,Any]]=[]
    def submit(self,spec:BuildTaskSpec,fn:Callable[...,Any],/,*args:Any,**kwargs:Any)->concurrent.futures.Future:
        impossible=self.pool.impossible_reason(spec.cpu_tokens,spec.ram_mb)
        if impossible:
            # Reject before a worker is occupied.  In particular, a RAM request
            # larger than the configured pool used to wait forever because no
            # other task could ever release enough memory.
            raise ValueError(f"build_scheduler_impossible_request:{spec.name}:{impossible}")
        # ThreadPoolExecutor does not propagate ContextVars.  Capture the
        # workflow's process-ownership context explicitly so compiler children
        # started by parallel Hailo/DeepX tasks remain cancellable.
        submit_context=contextvars.copy_context()
        def wrapped()->Any:
            sem=self.family_semaphores.get(spec.family)
            if sem is not None: sem.acquire()
            self.pool.acquire(spec.cpu_tokens,spec.ram_mb); started=time.time(); status="failed"; invocation_status="raised"; semantic_status="failed"; transport_status="failed"; artifact_status="failed"
            self.log(f"[build-scheduler] start {spec.name} family={spec.family} cpu={spec.cpu_tokens} ram_mb={spec.ram_mb}")
            try:
                value=fn(*args,**kwargs)
                invocation_status="returned"
                transport_status="success"
                semantic_status=semantic_result_status(value)
                artifact_status=semantic_status
                status="ok" if semantic_status == "success" else semantic_status
                return value
            finally:
                ended=time.time(); self.pool.release(spec.cpu_tokens,spec.ram_mb)
                if sem is not None: sem.release()
                event={"name":spec.name,"family":spec.family,"cpu_tokens":spec.cpu_tokens,"ram_mb":spec.ram_mb,"status":status,"invocation_status":invocation_status,"semantic_status":semantic_status,"transport_status":transport_status,"artifact_status":artifact_status,"started_at":started,"ended_at":ended,"elapsed_s":ended-started,"metadata":dict(spec.metadata)}
                with self._lock:self._events.append(event)
                self.log(f"[build-scheduler] finish {spec.name} transport={transport_status} invocation={invocation_status} artifact={artifact_status} semantic={semantic_status} status={status} elapsed={ended-started:.1f}s")
        return self.executor.submit(submit_context.run,wrapped)
    def events(self)->list[dict[str,Any]]:
        with self._lock:return list(self._events)
    def shutdown(self,wait:bool=True)->None:self.executor.shutdown(wait=wait,cancel_futures=False)
    def __enter__(self):return self
    def __exit__(self,*_):self.shutdown(True)

def scheduler_config_from_mapping(config:Mapping[str,Any]|None,mode:str="standard")->dict[str,Any]:
    raw=dict(config or {}); workers=int(raw.get("max_workers") or (3 if mode in {"smoke","standard"} else 2)); cpus=int(raw.get("cpu_tokens") or max(1,(os.cpu_count() or 4)-2))
    return {"enabled":bool(raw.get("enabled",True)),"max_workers":workers,"cpu_tokens":cpus,"ram_mb":int(raw.get("ram_mb") or 0),"ram_reserve_mb":max(0,int(raw.get("ram_reserve_mb") or 2048)),"family_limits":dict(raw.get("family_limits") or {"hailo8":1,"hailo10":1,"deepx":1}),"weights":dict(raw.get("weights") or {"hailo8":{"cpu_tokens":4,"ram_mb":6144},"hailo10":{"cpu_tokens":4,"ram_mb":6144},"deepx":{"cpu_tokens":2,"ram_mb":4096}}),"prefetch_deepx_full":bool(raw.get("prefetch_deepx_full",True)),"pipeline_next_model":bool(raw.get("pipeline_next_model",False))}

def family_for_hailo_arch(hw_arch:str)->str:return "hailo10" if "10" in str(hw_arch or "").lower() else "hailo8"
