# Exact v2.79.29 cache/wrapper functions, kept for bounded regression reproduction.
from __future__ import annotations
import functools, hashlib, json, os, threading
from pathlib import Path
from typing import Any, MutableMapping
_LOCK = threading.RLock()

def _norm(s: Any) -> str:
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())

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
        if not callable(obj) or getattr(obj,'_v60m_hash_wrapper',False): continue
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
