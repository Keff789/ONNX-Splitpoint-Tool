
"""Runtime-budget helpers for generated benchmark suites and runners."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
from typing import Any


def task_limit(task: str, final: bool=False) -> int:
    if final or os.environ.get('SPLITPOINT_VALIDATION_MODE','').lower()=='final': return 0
    generic=os.environ.get('SPLITPOINT_VALIDATION_MAX_ITEMS')
    if generic not in (None,''):
        try: return max(0,int(generic))
        except Exception: pass
    task=(task or '').lower()
    key='SPLITPOINT_VALIDATION_MAX_DETECTION' if 'detect' in task or 'yolo' in task else 'SPLITPOINT_VALIDATION_MAX_CLASSIFICATION'
    default='200' if key.endswith('DETECTION') else '500'
    try: return max(0,int(os.environ.get(key,default)))
    except Exception: return int(default)


def limit_sequence(seq, task: str=''):
    try: n=task_limit(task)
    except Exception: n=0
    if not n: return seq
    try:
        if len(seq)<=n: return seq
        # deterministic and independent of predictions/rankings
        return type(seq)(sorted(seq,key=lambda x:str(x))[:n]) if not isinstance(seq,tuple) else tuple(sorted(seq,key=lambda x:str(x))[:n])
    except Exception: return seq


def stable_obj(obj: Any):
    if isinstance(obj,(str,int,float,bool,type(None))):
        if isinstance(obj,str) and len(obj)<4096:
            p=Path(obj).expanduser()
            try:
                if p.exists():
                    s=p.stat(); return {'path':str(p.resolve()),'size':s.st_size,'mtime_ns':s.st_mtime_ns}
            except Exception: pass
        return obj
    if isinstance(obj,dict): return {str(k):stable_obj(v) for k,v in sorted(obj.items(),key=lambda kv:str(kv[0]))}
    if isinstance(obj,(list,tuple)): return [stable_obj(v) for v in obj]
    return repr(obj)[:1000]


def cache_key(name,args,kwargs):
    payload={'name':name,'args':stable_obj(args),'kwargs':stable_obj(kwargs),
             'policy':os.environ.get('SPLITPOINT_QUALITY_POLICY_HASH',''),
             'mode':os.environ.get('SPLITPOINT_VALIDATION_MODE','')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()



def _looks_like_sample_sequence(value):
    if not isinstance(value,(list,tuple)) or len(value)<2: return False
    hits=0
    for x in value[:min(20,len(value))]:
        if isinstance(x,(str,Path)):
            sx=str(x).lower()
            if sx.endswith(('.jpg','.jpeg','.png','.bmp','.webp','.npy','.npz')) or '/' in sx: hits+=1
        elif isinstance(x,dict) and any(k in x for k in ('path','image','image_path','file_name','sample_id')): hits+=1
    return hits >= max(1,min(20,len(value))//2)


def _limit_arg(value,task):
    n=task_limit(task)
    if not n: return value
    if _looks_like_sample_sequence(value) and len(value)>n:
        vals=sorted(value,key=lambda x: str(x.get('sample_id') or x.get('path') or x.get('file_name') or x) if isinstance(x,dict) else str(x))[:n]
        return tuple(vals) if isinstance(value,tuple) else vals
    if isinstance(value,dict):
        out=value.copy(); changed=False
        for k,v in value.items():
            if str(k).lower() in {'items','samples','images','image_paths','validation_items','validation_images'}:
                nv=_limit_arg(v,task); out[k]=nv; changed = changed or nv is not v
        return out if changed else value
    return value


def install_validation_sequence_limits(globs):
    import functools, inspect
    for name,obj in list(globs.items()):
        nn=''.join(ch for ch in name.lower() if ch.isalnum())
        if not inspect.isfunction(obj) or getattr(obj,'_v60m_validation_limit',False): continue
        if not any(t in nn for t in ('runvalidation','evaluatevalidation','taskquality','qualitygate','loadvalidation','validationdataset','collectpredictions')): continue
        @functools.wraps(obj)
        def wrapped(*a,__orig=obj,**kw):
            task=str(kw.get('task') or kw.get('benchmark_task') or os.environ.get('SPLITPOINT_MODEL_TASK',''))
            aa=tuple(_limit_arg(v,task) for v in a)
            kk={k:_limit_arg(v,task) for k,v in kw.items()}
            return __orig(*aa,**kk)
        wrapped._v60m_validation_limit=True; globs[name]=wrapped


def install_task_quality_cache(globs):
    if os.environ.get('SPLITPOINT_TASK_QUALITY_CACHE','1').lower() in {'0','false','off'}: return
    import functools, inspect
    cache_dir=Path(os.environ.get('SPLITPOINT_TASK_QUALITY_CACHE_DIR','.task_quality_cache_v60m'))
    for name,obj in list(globs.items()):
        nn=''.join(ch for ch in name.lower() if ch.isalnum())
        if not inspect.isfunction(obj) or getattr(obj,'_v60m_quality_cache',False): continue
        if not any(t in nn for t in ('taskqualitygate','qualitygate','evaluatetaskquality','runtaskquality')): continue
        @functools.wraps(obj)
        def wrapped(*a,__orig=obj,__name=name,**kw):
            key=cache_key(__name,a,kw); path=cache_dir/(key+'.json')
            try:
                if path.exists(): return json.loads(path.read_text(encoding='utf-8'))['value']
            except Exception: pass
            value=__orig(*a,**kw)
            try:
                cache_dir.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.tmp')
                tmp.write_text(json.dumps({'created_at':time.time(),'value':value},sort_keys=True),encoding='utf-8')
                os.replace(tmp,path)
            except Exception: pass
            return value
        wrapped._v60m_quality_cache=True; globs[name]=wrapped
