"""IX01–IX08: current-path closure, fixed transactions and measured costs."""
from __future__ import annotations
import copy, hashlib, json, os, time, tracemalloc
from pathlib import Path
import pytest
from onnx_splitpoint_tool.workflow import runner as runner_module
from test_v27930_terminal_hash_uncached import Counter, make_runner
from test_v27924_terminal_hailo_aliases import published
PHASES=('refresh_current_records','verify_pending_index','verify_pass_index')


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('ONNX_SPLITPOINT_HASH_CACHE',str(tmp_path/'normal_fixture_cache.json'))



def test_IX01_history_dedup_before_hash_and_last_metadata(tmp_path):
    r,paths=make_runner(tmp_path,count=4,repeats=25)
    with Counter(r) as c:r._finalize_artifact_index(status='ok')
    assert c.hash_calls==dict.fromkeys(PHASES,len(paths))
    assert c.byte_reads==dict.fromkeys(PHASES,sum(p.stat().st_size for p in paths))
    rows=r.artifact_index['artifacts']
    assert [x['path'] for x in rows]==sorted(x['path'] for x in rows)
    assert all(x['kind']=='history_24' and x['producer_stage']=='stage_24' and x['model_id']=='model' for x in rows if x['path'].endswith('.bin'))


@pytest.mark.parametrize('invalid',[None,{}, {'path':'reports/../reports/payload_00000.bin'}, {'path':'artifact_index.json'}, {'path':'/etc/passwd'}])
def test_IX02_bad_history_not_hidden_by_valid_row(tmp_path,invalid):
    r,_=make_runner(tmp_path,count=1);r.artifact_index['artifacts'].insert(0,invalid)
    with Counter(r) as c:
        with pytest.raises(RuntimeError):r._finalize_artifact_index(status='ok')
    assert c.hash_calls=={};c.assert_no_cache()


def test_IX02_terminal_duplicate_still_invalid(tmp_path):
    r,_=make_runner(tmp_path);r._finalize_artifact_index(status='ok')
    r.artifact_index['artifacts'].append(copy.deepcopy(r.artifact_index['artifacts'][0]))
    runner_module.atomic_write_json(r.artifact_index_path,r.artifact_index)
    result=r._verify_terminal_artifact_index(required_paths=r._terminal_evidence_candidates())
    assert any(e['error']=='duplicate_path' for e in result['errors'])


def test_IX03_all_required_directories_and_types_covered(tmp_path):
    r,_=make_runner(tmp_path,count=0);expected=set()
    for directory in r._TERMINAL_EVIDENCE_DIRECTORIES:
        for name in ('image.jpg','model.dxnn','energy.parquet','data.json','important.tmp'):
            path=r.run_dir/directory/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'full payload')
            expected.add(path.relative_to(r.run_dir).as_posix())
    (r.run_dir/'root.json').write_text('{}');expected.add('root.json')
    path=r.run_dir/'custom/out.bin';path.parent.mkdir();path.write_bytes(b'extra')
    r.outputs={'custom':str(path)};expected.add('custom/out.bin')
    with Counter(r) as c:r._finalize_artifact_index(status='ok')
    assert expected <= {x['path'] for x in r.artifact_index['artifacts']}
    assert c.hash_calls==dict.fromkeys(PHASES,len(expected))


def test_IX04_HC04_real_hailo_publisher_no_transitive_cache(published):
    r,alias,generation=published
    r.artifact_index['artifacts']=[{'path':alias.relative_to(r.run_dir).as_posix()}]*3
    with Counter(r) as c:r._finalize_artifact_index(status='ok')
    c.assert_no_cache()
    out=Path(os.environ.get('ONNX_SPLITPOINT_TERMINAL_REPORT_DIR',str(r.run_dir.parent)))
    out.mkdir(parents=True,exist_ok=True)
    (out/'terminal_hailo_alias_counters.json').write_text(json.dumps({
        'synthetic':True, 'sha_cache_calls':c.cache_calls,
        'content_calls_by_phase':c.hash_calls,'content_bytes_by_phase':c.byte_reads,
        'alias_receipt_checks_counted_separately':True,
    },indent=2)+'\n')
    assert generation.relative_to(r.run_dir).as_posix() in {x['path'] for x in r.artifact_index['artifacts']}
    alias.unlink();alias.symlink_to('/etc/passwd')
    with pytest.raises(RuntimeError):r._terminal_evidence_candidates()


@pytest.mark.parametrize('count',[1,20])
def test_IX05_pending_early_fixed_atomic_writes(tmp_path,count):
    r,_=make_runner(tmp_path,count=count)
    with Counter(r) as c:r._finalize_artifact_index(status='ok')
    assert c.pending_visible and (c.index_writes,c.report_writes)==(2,3)


@pytest.mark.parametrize('stage',['pending','pass'])
def test_IX06_independent_reads_detect_mutation(tmp_path,monkeypatch,stage):
    r,paths=make_runner(tmp_path);real=r._verify_terminal_artifact_index
    def verify(**kwargs):
        if kwargs['expected_closure_status']==stage:
            st=paths[0].stat();data=paths[0].read_bytes();paths[0].write_bytes(b'X'+data[1:]);os.utime(paths[0],ns=(st.st_atime_ns,st.st_mtime_ns))
        return real(**kwargs)
    monkeypatch.setattr(r,'_verify_terminal_artifact_index',verify)
    with Counter(r) as c:
        with pytest.raises(RuntimeError,match='sha256_mismatch'):r._finalize_artifact_index(status='ok')
    c.assert_no_cache();assert json.loads(r.artifact_index_path.read_text())['terminal_closure']['status']=='fail'


@pytest.mark.parametrize('write_number',[1,2,3,4,5])
def test_IX07_atomic_error_never_leaves_proven_success(tmp_path,monkeypatch,write_number):
    r,_=make_runner(tmp_path);real=runner_module.atomic_write_json;n=0
    def write(path,payload):
        nonlocal n;n+=1
        if n==write_number:raise OSError('synthetic atomic I/O failure')
        return real(path,payload)
    monkeypatch.setattr(runner_module,'atomic_write_json',write)
    with pytest.raises(OSError):r._finalize_artifact_index(status='ok')
    if r.artifact_index_path.exists():
        assert not r._verify_terminal_artifact_index(required_paths=r._terminal_evidence_candidates())['ok']


@pytest.mark.parametrize('count',[1000,5000,10000])
def test_IX08_scale_with_large_cache(tmp_path,monkeypatch,count):
    cache=tmp_path/'synthetic_cache.json';cache.write_text(json.dumps({'synthetic_padding':'x'*18_500_000}))
    digest=hashlib.sha256(cache.read_bytes()).hexdigest();st=cache.stat()
    monkeypatch.setenv('ONNX_SPLITPOINT_HASH_CACHE',str(cache))
    r,paths=make_runner(tmp_path,count=count,repeats=3)
    history=copy.deepcopy(r.artifact_index['artifacts']);payload_bytes=sum(p.stat().st_size for p in paths)
    tracemalloc.start();start=time.perf_counter()
    with Counter(r) as c:r._finalize_artifact_index(status='failed')
    elapsed=time.perf_counter()-start;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    c.assert_no_cache()
    assert c.hash_calls==dict.fromkeys(PHASES,count)
    assert c.byte_reads==dict.fromkeys(PHASES,payload_bytes)
    assert (c.index_writes,c.report_writes)==(2,3)
    assert digest==hashlib.sha256(cache.read_bytes()).hexdigest()
    assert (cache.stat().st_size,cache.stat().st_mtime_ns,cache.stat().st_ino)==(st.st_size,st.st_mtime_ns,st.st_ino)
    assert not list(tmp_path.glob('synthetic_cache.json.tmp-*'))
    # Same history admission/inventory and two real verifiers; no reused digests.
    tracemalloc.start();start=time.perf_counter();latest={}
    for row in history:
        path=r._artifact_record_path(row);assert path.is_file();latest[path]=row
    required=r._terminal_evidence_candidates()
    for path in required:
        admitted=r._artifact_record_path({'path':path.relative_to(r.run_dir).as_posix()})
        assert admitted.is_file();latest[admitted]={}
    for path in sorted(latest):assert runner_module.sha256_file_uncached(path)
    for _ in range(2):assert r._verify_terminal_artifact_index(required_paths=required)['ok']
    baseline=time.perf_counter()-start;_,baseline_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    report={'synthetic':True,'unique_payload_files':count,'mutable_registration_rows_before_dedup':len(history),
        'payload_hash_calls_by_phase':c.hash_calls,'payload_bytes_read_by_phase':c.byte_reads,
        'sha_cache_load_calls':c.cache_calls['_load_cache'],'sha_cache_rewrite_calls':c.cache_calls['_atomic_json'],
        'cached_sha256_calls':c.cache_calls['cached_sha256'],'index_write_calls':c.index_writes,'closure_report_write_calls':c.report_writes,
        'elapsed_s':elapsed,'peak_memory_bytes':peak,'memory_measurement':'tracemalloc Python allocations during closure; fixture construction and OS page cache excluded','baseline_elapsed_s':baseline,'baseline_peak_memory_bytes':baseline_peak,
        'baseline_ratio':elapsed/baseline,'baseline_description':'same historical and inventory path checks, direct refresh, two real verifiers'}
    out=Path(os.environ.get('ONNX_SPLITPOINT_TERMINAL_REPORT_DIR',str(tmp_path)));out.mkdir(parents=True,exist_ok=True)
    (out/f'terminal_scale_{count}.json').write_text(json.dumps(report,indent=2)+'\n')
