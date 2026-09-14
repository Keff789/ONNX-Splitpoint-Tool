
"""v60m runtime policy.

This module deliberately separates *scientific final-campaign integrity* from
*developer convenience*:

- final campaign -> strict SHA-256 and complete immutable provenance;
- development -> cached SHA-256 (same digest, no repeated full-file reads);
- exploratory -> warning-only metadata mode, never final-claim eligible.

It also makes the top-level u.RECS energy switch authoritative for native
energy and provides development validation budgets/caching without changing
final-campaign semantics.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, MutableMapping

_LOCK = threading.RLock()
_ORIG_YAML_SAFE_LOAD = None
_ORIG_YAML_SAFE_DUMP = None


def _norm(s: Any) -> str:
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())


def _bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0,1):
        return bool(v)
    if isinstance(v, str):
        t=v.strip().lower()
        if t in {'1','true','yes','on','enabled','measure'}: return True
        if t in {'0','false','no','off','disabled','none'}: return False
    return None


def is_evaluation_profile(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    keys={_norm(k) for k in obj}
    has_models=bool(keys & {'models','model','modelentries'})
    has_workflow=bool(keys & {'workflow','execution','evaluation','targets','candidateconfiguration','validation','reporting'})
    schema=str(obj.get('schema','') or obj.get('schema_id','')).lower()
    return (has_models and has_workflow) or ('evaluation' in schema and 'profile' in schema)


def _walk(obj: Any, path=()):
    if isinstance(obj, dict):
        for k,v in list(obj.items()):
            yield path+(str(k),), obj, k, v
            yield from _walk(v, path+(str(k),))
    elif isinstance(obj, list):
        for i,v in enumerate(obj):
            yield from _walk(v, path+(str(i),))


def profile_is_final(profile: Mapping[str, Any]) -> bool:
    for path, parent, key, value in _walk(profile):
        pn='.'.join(_norm(x) for x in path)
        kn=_norm(key)
        if kn in {'campaignmode','evaluationmode','profilemode','mode'} and any(t in pn for t in ('campaign','evaluation','profile')):
            if str(value).strip().lower() in {'final','finalcampaign','production','claim'}:
                return True
        if kn in {'frozenbeforefinalcampaign','finalcampaign','strictfinal'} and _bool(value) is True:
            return True
    return False


def _master_energy(profile: Mapping[str, Any]) -> tuple[bool|None,str|None]:
    """Return the user-facing Native-energy master switch.

    Since v60o, Generic Runner energy is intentionally disabled.  Therefore
    ``energy.enabled = false`` is *not* a global energy-off signal when the
    materialised profile declares ``measurement_path: native_only``.  In that
    mode the authoritative value is the high-level
    ``requested_native_energy`` flag (or the native producer energy flag as a
    compatibility fallback).
    """
    energy = profile.get('energy') if isinstance(profile, Mapping) else None
    if isinstance(energy, Mapping) and str(energy.get('measurement_path') or '').strip().lower() == 'native_only':
        requested = _bool(energy.get('requested_native_energy'))
        if requested is not None:
            return requested, 'energy.requested_native_energy'
        requested = _bool(profile.get('requested_native_energy'))
        if requested is not None:
            return requested, 'requested_native_energy'
        preset = profile.get('execution_preset')
        if isinstance(preset, Mapping):
            overrides = preset.get('overrides')
            if isinstance(overrides, Mapping):
                requested = _bool(overrides.get('energy_enabled'))
                if requested is not None:
                    return requested, 'execution_preset.overrides.energy_enabled'
        native = profile.get('native_producers')
        if isinstance(native, Mapping):
            native_energy = native.get('energy')
            if isinstance(native_energy, Mapping):
                requested = _bool(native_energy.get('enabled'))
                if requested is not None:
                    return requested, 'native_producers.energy.enabled'
        return False, 'energy.measurement_path=native_only(default_off)'

    exact=[]; fallback=[]
    for path,parent,key,value in _walk(profile):
        b=_bool(value)
        if b is None: continue
        parts=[_norm(x) for x in path]
        joined='.'.join(parts)
        keyn=_norm(key)
        if 'native' in joined: continue
        if keyn in {'energyenabled','enableenergy','measureenergy','urecsenergy','urecsenergyenabled'}:
            exact.append((len(path),b,'.'.join(path)))
        elif keyn in {'enabled','measure'} and any(t in joined for t in ('energymeasurement','urecsenergy','energymeasure')):
            exact.append((len(path),b,'.'.join(path)))
        elif keyn=='enabled' and 'energy' in joined:
            fallback.append((len(path),b,'.'.join(path)))
    cand=sorted(exact or fallback, key=lambda x:x[0])
    return (cand[0][1],cand[0][2]) if cand else (None,None)


def enforce_energy_master(profile: MutableMapping[str, Any]) -> dict[str,Any]:
    master, source = _master_energy(profile)
    changed=[]; native_requested=[]
    if master is False:
        for path,parent,key,value in list(_walk(profile)):
            joined='.'.join(_norm(x) for x in path)
            keyn=_norm(key)
            if 'native' not in joined: continue
            if not any(t in joined for t in ('energy','power')) and 'energy' not in keyn:
                continue
            b=_bool(value)
            if b is True:
                native_requested.append('.'.join(path))
            if keyn in {'energymode','mode'} and isinstance(value,str) and value.lower() in {'measure','measured','model','enabled','auto'}:
                parent[key]='disabled'; changed.append('.'.join(path))
            elif b is not None and (keyn in {'energy','nativeenergy','energyenabled','enableenergy','measureenergy','enabled'} or 'energy' in keyn):
                if value is not False:
                    parent[key]=False; changed.append('.'.join(path))
    return {'master_enabled':master,'master_source':source,'native_requested':native_requested,'changed':changed}


def _role(profile: Mapping[str,Any]) -> str:
    if profile_is_final(profile): return 'final'
    vals=[]
    for path,parent,key,value in _walk(profile):
        if _norm(key) in {'evaluationrole','role','campaignmode','evaluationmode'}:
            vals.append(str(value).lower())
    if any(v in {'holdout','final','production','claim'} for v in vals): return 'final'
    if any(v in {'screening','development','dev','exploratory'} for v in vals): return 'development'
    return 'development'


def normalize_profile(profile: MutableMapping[str,Any]) -> tuple[MutableMapping[str,Any],dict[str,Any]]:
    final=profile_is_final(profile)
    ip=profile.get('integrity_policy')
    if not isinstance(ip,dict):
        ip={}; profile['integrity_policy']=ip
    requested=str(ip.get('mode','auto') or 'auto').lower()
    if requested not in {'auto','relaxed','fast','strict','off'}: requested='auto'
    effective='strict' if final else ('fast' if requested=='auto' else requested)
    if final and effective!='strict': effective='strict'
    ip.update({
        'mode':requested,
        'effective_mode':effective,
        'strict_required_for_final':True,
        'cache_unchanged_files': effective=='fast',
        'final_claim_eligible': effective=='strict' if final else False,
    })
    os.environ['ONNX_SPLITPOINT_INTEGRITY_MODE']=effective

    ve=profile.get('validation_execution')
    if not isinstance(ve,dict):
        ve={}; profile['validation_execution']=ve
    role=_role(profile)
    mode=str(ve.get('mode','auto') or 'auto').lower()
    if mode=='auto': mode='final' if role=='final' else 'screening'
    ve['mode']=mode
    ve.setdefault('cadence','once_per_artifact')
    ve.setdefault('cache_task_quality',True)
    limits=ve.get('max_items')
    if not isinstance(limits,dict): limits={}; ve['max_items']=limits
    if mode=='final':
        limits.setdefault('classification',0); limits.setdefault('detection',0)
    else:
        limits.setdefault('classification',500); limits.setdefault('detection',200)
    ve.setdefault('repeat_task_quality_for_timing_repeats',False)
    os.environ['SPLITPOINT_TASK_QUALITY_CACHE']='1' if ve.get('cache_task_quality',True) else '0'
    os.environ['SPLITPOINT_VALIDATION_MODE']=mode
    os.environ['SPLITPOINT_VALIDATION_MAX_CLASSIFICATION']=str(int(limits.get('classification') or 0))
    os.environ['SPLITPOINT_VALIDATION_MAX_DETECTION']=str(int(limits.get('detection') or 0))

    energy=enforce_energy_master(profile)
    audit={'final':final,'role':role,'integrity_mode':effective,'validation_execution':copy.deepcopy(ve),'energy':energy}
    return profile,audit


def _cache_path() -> Path:
    p=os.environ.get('ONNX_SPLITPOINT_HASH_CACHE')
    return Path(p).expanduser() if p else Path.home()/'.onnx_splitpoint_tool'/'cache'/'sha256_v60m.json'


def _load_cache(path: Path) -> dict[str,Any]:
    try:
        v=json.loads(path.read_text(encoding='utf-8'))
        return v if isinstance(v,dict) else {}
    except Exception:
        return {}


def _atomic_json(path: Path, value: Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value,sort_keys=True,separators=(',',':')),encoding='utf-8')
    os.replace(tmp,path)


def strict_sha256(path: str|os.PathLike[str], chunk_size: int=8*1024*1024) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk_size)
            if not b: break
            h.update(b)
    return h.hexdigest()


def _fast_content_probe(path: Path, size: int, chunk_size: int=64*1024) -> str:
    """Detect in-place rewrites when filesystem stat fields do not advance.

    Small contract/manifest files are read completely.  Large model files use
    fixed first/middle/last samples so the fast cache remains inexpensive; the
    strict mode continues to hash every byte.
    """
    h=hashlib.sha256()
    h.update(f'{int(size)}:'.encode('ascii'))
    with path.open('rb') as f:
        if int(size) <= 3*int(chunk_size):
            h.update(f.read())
        else:
            offsets=(0,max(0,int(size)//2-int(chunk_size)//2),max(0,int(size)-int(chunk_size)))
            for offset in offsets:
                f.seek(offset)
                h.update(f.read(int(chunk_size)))
    return h.hexdigest()


def cached_sha256(path: str|os.PathLike[str], *args, **kwargs) -> str:
    p=Path(path).expanduser().resolve()
    mode=os.environ.get('ONNX_SPLITPOINT_INTEGRITY_MODE','fast').lower()
    if mode=='strict':
        return strict_sha256(p)
    st=p.stat()
    sig=f'{st.st_size}:{st.st_mtime_ns}:{getattr(st,"st_ino",0)}'
    probe=_fast_content_probe(p,int(st.st_size))
    cp=_cache_path(); key=str(p)
    with _LOCK:
        cache=_load_cache(cp)
        row=cache.get(key)
        if (isinstance(row,dict) and row.get('signature')==sig
                and row.get('probe_sha256')==probe
                and isinstance(row.get('sha256'),str)):
            return row['sha256']
        digest=strict_sha256(p)
        cache[key]={'signature':sig,'probe_sha256':probe,'sha256':digest,'updated_ns':__import__('time').time_ns()}
        if len(cache)>50000:
            cache=dict(list(cache.items())[-40000:])
        _atomic_json(cp,cache)
        return digest


def install_hash_wrappers(module_globals: MutableMapping[str,Any]) -> None:
    for name,obj in list(module_globals.items()):
        if (not callable(obj) or getattr(obj, '_v60m_uncached', False)
                or getattr(obj, '_v60m_hash_wrapper', False)): continue
        nn=_norm(name)
        if 'sha256' not in nn and nn not in {'filehash','hashfile'}: continue
        try:
            import inspect
            sig=inspect.signature(obj)
            params=list(sig.parameters.values())
        except Exception:
            continue
        if not params: continue
        pn=_norm(params[0].name)
        if not any(t in pn for t in ('path','file','artifact','model','archive')): continue
        output_prefix = str(getattr(obj, '_v60m_hash_prefix', '') or '')
        @functools.wraps(obj)
        def wrapped(path,*a,__orig=obj,__output_prefix=output_prefix,**kw):
            mode=os.environ.get('ONNX_SPLITPOINT_INTEGRITY_MODE','fast').lower()
            if mode=='strict': return __orig(path,*a,**kw)
            try:
                digest=cached_sha256(path)
                if __output_prefix and not str(digest).startswith(__output_prefix):
                    return __output_prefix+str(digest)
                return digest
            except Exception: return __orig(path,*a,**kw)
        wrapped._v60m_hash_wrapper=True
        module_globals[name]=wrapped


def install_energy_object_guards(module_globals: MutableMapping[str,Any]) -> None:
    import inspect
    exact_master={'energy_enabled','enable_energy','measure_energy','urecs_energy_enabled','urecs_energy'}
    for name,cls in list(module_globals.items()):
        if not inspect.isclass(cls) or getattr(cls,'_v60m_energy_class',False): continue
        if cls.__module__ != module_globals.get('__name__'): continue
        orig=getattr(cls,'__init__',None)
        if not callable(orig): continue
        @functools.wraps(orig)
        def init(self,*a,__orig=orig,**kw):
            __orig(self,*a,**kw)
            masters=[]
            for n in exact_master:
                if hasattr(self,n):
                    b=_bool(getattr(self,n));
                    if b is not None: masters.append(b)
            if masters and False in masters:
                for n in dir(self):
                    nn=_norm(n)
                    if 'native' in nn and 'energy' in nn:
                        try:
                            v=getattr(self,n)
                            if _bool(v) is not None: setattr(self,n,False)
                        except Exception: pass
        cls.__init__=init
        cls._v60m_energy_class=True



def _screening_mode(profile: Mapping[str,Any]) -> bool:
    ve=profile.get('validation_execution') if isinstance(profile,dict) else None
    if isinstance(ve,dict):
        mode=str(ve.get('mode','auto') or 'auto').lower()
        if mode=='screening': return True
        if mode=='final': return False
    return not profile_is_final(profile)


def _profile_requests_final_registry(profile: Mapping[str, Any]) -> bool:
    preset = profile.get('execution_preset') if isinstance(profile, Mapping) else None
    if isinstance(preset, Mapping):
        snap = preset.get('snapshot') if isinstance(preset.get('snapshot'), Mapping) else {}
        data = snap.get('data') if isinstance(snap.get('data'), Mapping) else {}
        if bool(data.get('use_final_dataset_registry')):
            return True
        if str(preset.get('id') or '').strip().lower() in {'standard', 'final'}:
            return True
    campaign = profile.get('campaign') if isinstance(profile, Mapping) else None
    if isinstance(campaign, Mapping) and bool(campaign.get('auto_bind_dataset_registry')):
        return True
    return False


def _validation_manifest_leaf(path, key) -> bool:
    joined='.'.join(_norm(x) for x in path)
    kn=_norm(key)
    if 'calibration' in joined or 'calibration' in kn: return False
    return ('validationmanifest' in joined or 'validationmanifest' in kn or
            kn in {'classificationvalidation','detectionvalidation','clsvalidationmanifest','detvalidationmanifest'})


def _capture_validation_leaves(profile):
    out={}
    for path,parent,key,value in _walk(profile):
        if _validation_manifest_leaf(path,key): out[path]=copy.deepcopy(value)
    return out


def _restore_new_final_validation(profile, before):
    # Development/screening profiles keep explicit validation settings, but do
    # not get silently upgraded to full ImageNet/COCO manifests merely because
    # the final-data registry is ready.
    for path,parent,key,value in list(_walk(profile)):
        if not _validation_manifest_leaf(path,key): continue
        old=before.get(path,None)
        if path not in before or old in (None,'',[],{}):
            if isinstance(parent,dict): parent[key]=copy.deepcopy(old) if path in before else ''


def install_dataset_binding_guards(module_globals: MutableMapping[str,Any]) -> None:
    import inspect
    for name,obj in list(module_globals.items()):
        if not callable(obj) or getattr(obj,'_v60m_dataset_guard',False): continue
        nn=_norm(name)
        if 'bind' not in nn or not any(t in nn for t in ('dataset','registry','manifest')): continue
        @functools.wraps(obj)
        def wrapped(*a,__orig=obj,**kw):
            profile=next((x for x in list(a)+list(kw.values()) if is_evaluation_profile(x)),None)
            if profile is None or not _screening_mode(profile) or _profile_requests_final_registry(profile):
                return __orig(*a,**kw)
            before=_capture_validation_leaves(profile)
            result=__orig(*a,**kw)
            target=result if is_evaluation_profile(result) else profile
            _restore_new_final_validation(target,before)
            return result
        wrapped._v60m_dataset_guard=True; module_globals[name]=wrapped


def install_yaml_hooks() -> None:
    global _ORIG_YAML_SAFE_LOAD,_ORIG_YAML_SAFE_DUMP
    try: import yaml
    except Exception: return
    if getattr(yaml,'_onnx_splitpoint_v60m',False): return
    _ORIG_YAML_SAFE_LOAD=yaml.safe_load; _ORIG_YAML_SAFE_DUMP=yaml.safe_dump
    def safe_load(stream):
        obj=_ORIG_YAML_SAFE_LOAD(stream)
        if is_evaluation_profile(obj): normalize_profile(obj)
        return obj
    def safe_dump(data,*a,**kw):
        if is_evaluation_profile(data):
            data=copy.deepcopy(data); normalize_profile(data)
        return _ORIG_YAML_SAFE_DUMP(data,*a,**kw)
    yaml.safe_load=safe_load; yaml.safe_dump=safe_dump; yaml._onnx_splitpoint_v60m=True
