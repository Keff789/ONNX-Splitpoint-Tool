"""HC01–HC08: full content verification independent of the development cache."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from onnx_splitpoint_tool import v60m_policy as policy
from onnx_splitpoint_tool.workflow import artifacts, runner as runner_module
from test_v2796_artifact_index_closure import _runner


def make_runner(tmp_path, count=3, repeats=3):
    root = tmp_path / 'run'
    root.mkdir()
    r = _runner(root)
    folder = root / 'reports'
    folder.mkdir()
    paths=[]
    for i in range(count):
        p=folder / f'payload_{i:05d}.bin'
        p.write_bytes(f'payload:{i:05d}'.encode())
        paths.append(p)
        r.artifact_index['artifacts'].extend(
            {'path':p.relative_to(root).as_posix(), 'kind':f'history_{j}', 'producer_stage':f'stage_{j}', 'model_id':'model'}
            for j in range(repeats)
        )
    return r, paths


class Counter:
    """Count actual direct reads/writes without replacing their implementation."""
    def __init__(self, runner):
        self.runner=runner
        self.hash_calls={}; self.byte_reads={}; self.index_writes=0; self.report_writes=0
        self.cache_calls={'_load_cache':0, '_atomic_json':0, 'cached_sha256':0}
        self.pending_visible=False
        self.hash_impl=runner_module.sha256_file_uncached
        self.write_impl=runner_module.atomic_write_json
        self.patches=[]

    def __enter__(self):
        def direct(path, *args, on_chunk=None, **kwargs):
            phase=getattr(self.runner, '_terminal_hash_state', {}).get('phase', 'alias_receipt')
            if Path(path).name != 'artifact_index_closure.json':
                self.hash_calls[phase]=self.hash_calls.get(phase,0)+1
            def chunk(n):
                if Path(path).name != 'artifact_index_closure.json':
                    self.byte_reads[phase]=self.byte_reads.get(phase,0)+n
                if on_chunk is not None: on_chunk(n)
            report=self.runner.run_dir / 'reports/artifact_index_closure.json'
            if not self.pending_visible:
                assert json.loads(report.read_text())['status']=='pending'
                self.pending_visible=True
            return self.hash_impl(path,*args,on_chunk=chunk,**kwargs)
        def write(path,payload):
            if Path(path)==self.runner.artifact_index_path: self.index_writes+=1
            if Path(path).name=='artifact_index_closure.json': self.report_writes+=1
            return self.write_impl(path,payload)
        self.patches=[patch.object(runner_module,'sha256_file_uncached',direct),patch.object(runner_module,'atomic_write_json',write)]
        for name in self.cache_calls:
            original=getattr(policy,name)
            def observe(*args,__name=name,__original=original,**kwargs):
                self.cache_calls[__name]+=1
                return __original(*args,**kwargs)
            self.patches.append(patch.object(policy,name,observe))
        for item in self.patches: item.start()
        return self

    def __exit__(self,*exc):
        for item in reversed(self.patches): item.stop()

    def assert_no_cache(self):
        assert self.cache_calls==dict.fromkeys(self.cache_calls,0)


def test_HC01_original_wrapper_reproduction(tmp_path, monkeypatch):
    path=Path(__file__).parent/'fixtures/v27930_old_hash_cache.py'
    spec=importlib.util.spec_from_file_location('old_v29_cache',path)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    monkeypatch.setenv('ONNX_SPLITPOINT_HASH_CACHE',str(tmp_path/'cache.json'))
    monkeypatch.setenv('ONNX_SPLITPOINT_INTEGRITY_MODE','fast')
    counts={'load':0,'write':0}
    for name,key in [('_load_cache','load'),('_atomic_json','write')]:
        orig=getattr(old,name)
        def track(*a,__orig=orig,__key=key,**kw):
            counts[__key]+=1
            return __orig(*a,**kw)
        monkeypatch.setattr(old,name,track)
    namespace={'sha256_file':lambda path:'unused'}
    old.install_hash_wrappers(namespace)
    paths=[tmp_path/f'{i}.bin' for i in range(3)]
    for i,p in enumerate(paths):p.write_bytes(bytes([i])*5)
    sizes=[]
    for p in paths+paths[:2]:
        assert namespace['sha256_file'](p)==hashlib.sha256(p.read_bytes()).hexdigest()
        sizes.append((tmp_path/'cache.json').stat().st_size)
    assert counts=={'load':5,'write':3}
    assert sizes[0]<sizes[1]<sizes[2] and sizes[2:]==[sizes[2]]*3


@pytest.mark.parametrize('size,chunk_size',[(0,13),(31,7),(3*1024*1024+21,65536)])
def test_HC02_direct_digest_and_chunks(tmp_path,size,chunk_size):
    p=tmp_path/'data'; data=(b'0123456789abcdef'*((size+15)//16))[:size];p.write_bytes(data)
    chunks=[]
    assert artifacts.sha256_file_uncached(p,chunk_size,on_chunk=chunks.append)=='sha256:'+hashlib.sha256(data).hexdigest()
    assert sum(chunks)==size and all(0<n<=chunk_size for n in chunks)


@pytest.mark.parametrize('chunk',[0,-1,True,1.5,'4'])
def test_HC02_invalid_chunk_missing_and_read_error(tmp_path,chunk):
    with pytest.raises(ValueError):artifacts.sha256_file_uncached(tmp_path/'missing',chunk)
    assert artifacts.sha256_file_uncached(tmp_path/'missing') is None
    p=tmp_path/'data';p.write_bytes(b'x')
    with patch.object(Path,'open',side_effect=PermissionError('denied')):
        with pytest.raises(PermissionError):artifacts.sha256_file_uncached(p)


def test_HC03_wrapper_exclusion_alias_and_default_file_record(tmp_path,monkeypatch):
    monkeypatch.setenv('ONNX_SPLITPOINT_INTEGRITY_MODE','fast')
    monkeypatch.setenv('ONNX_SPLITPOINT_HASH_CACHE',str(tmp_path/'cache.json'))
    direct=artifacts.sha256_file_uncached
    aliases={'sha256_alias':direct,'sha256_file':artifacts.sha256_file}
    policy.install_hash_wrappers(aliases);policy.install_hash_wrappers(aliases)
    assert aliases['sha256_alias'] is direct
    assert not getattr(direct,'_v60m_hash_wrapper',False)
    assert getattr(aliases['sha256_file'],'_v60m_hash_wrapper',False)
    p=tmp_path/'data';p.write_bytes(b'data')
    with patch.object(policy,'_load_cache',wraps=policy._load_cache) as load:
        direct(p);assert load.call_count==0
        artifacts.file_record(p,root=tmp_path,kind='x',producer_stage='x');assert load.call_count==1
        artifacts.file_record(p,root=tmp_path,kind='x',producer_stage='x',hash_fn=direct);assert load.call_count==1


@pytest.mark.parametrize('damage',['none','pending_mutation','pass_mutation','pass_exception'])
def test_HC04_all_terminal_calls_and_failure_paths_are_direct(tmp_path,monkeypatch,damage):
    r,paths=make_runner(tmp_path)
    orig=r._verify_terminal_artifact_index
    def verify(**kwargs):
        stage=kwargs['expected_closure_status']
        if damage==stage+'_mutation':paths[0].write_bytes(b'bad')
        if damage=='pass_exception' and stage=='pass':raise OSError('late read failure')
        return orig(**kwargs)
    monkeypatch.setattr(r,'_verify_terminal_artifact_index',verify)
    with Counter(r) as counts:
        if damage=='none':r._finalize_artifact_index(status='failed')
        else:
            with pytest.raises(RuntimeError):r._finalize_artifact_index(status='failed')
    counts.assert_no_cache()
    report=json.loads((r.run_dir/'reports/artifact_index_closure.json').read_text())
    assert report['status']==('pass' if damage=='none' else 'fail')
    assert report['workflow_status']=='failed'


@pytest.mark.parametrize('mode',['fast','strict'])
def test_HC05_fresh_process_mode_and_environment_isolation(tmp_path,mode):
    code="""
import json, os, sys
from pathlib import Path
from unittest.mock import patch
from onnx_splitpoint_tool import v60m_policy as p
from onnx_splitpoint_tool.workflow import artifacts as a
from test_v27930_terminal_hash_uncached import make_runner, Counter
r, paths=make_runner(Path(sys.argv[1])); before=dict(os.environ)
with Counter(r) as c:r._finalize_artifact_index(status='ok')
c.assert_no_cache(); assert dict(os.environ)==before
with patch.object(p,'_load_cache',wraps=p._load_cache) as m:
 a.sha256_file(paths[0]); assert m.call_count == (1 if sys.argv[2]=='fast' else 0)
print(json.dumps({'mode':sys.argv[2],'ok':True}))
"""
    root=Path(__file__).parents[1]
    env={**os.environ,'ONNX_SPLITPOINT_INTEGRITY_MODE':mode,'ONNX_SPLITPOINT_HASH_CACHE':str(tmp_path/'cache.json'),'PYTHONPATH':os.pathsep.join([str(root),str(root/'tests')])}
    proc=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path),mode],env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stdout+proc.stderr


@pytest.mark.parametrize('cache_state',['missing','corrupt','unwritable','large'])
def test_HC06_HC07_existing_cache_untouched(tmp_path,monkeypatch,cache_state):
    cache=tmp_path/'cache.json'
    if cache_state=='corrupt':cache.write_bytes(b'not JSON')
    if cache_state=='unwritable':cache.mkdir()
    if cache_state=='large':cache.write_text(json.dumps({'padding':'x'*18_500_000}))
    monkeypatch.setenv('ONNX_SPLITPOINT_HASH_CACHE',str(cache))
    before=cache.stat() if cache.exists() else None
    digest=hashlib.sha256(cache.read_bytes()).hexdigest() if cache.is_file() else None
    r,_=make_runner(tmp_path)
    with Counter(r) as counts:r._finalize_artifact_index(status='ok')
    counts.assert_no_cache()
    if before:
        after=cache.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    else:assert not cache.exists()
    if digest:assert hashlib.sha256(cache.read_bytes()).hexdigest()==digest
    assert not list(tmp_path.glob('cache.json.tmp-*'))


def test_HC08_equal_size_mutation_outside_fast_samples_detected(tmp_path):
    r,paths=make_runner(tmp_path,count=1)
    p=paths[0];p.write_bytes(b'x'*(4*1024*1024))
    r._finalize_artifact_index(status='ok')
    stat=p.stat(); probe=policy._fast_content_probe(p,stat.st_size)
    with p.open('r+b') as stream:stream.seek(256*1024);stream.write(b'y')
    os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert policy._fast_content_probe(p,p.stat().st_size)==probe
    check=r._verify_terminal_artifact_index(required_paths=r._terminal_evidence_candidates())
    assert not check['ok'] and any(row['error']=='sha256_mismatch' for row in check['errors'])
