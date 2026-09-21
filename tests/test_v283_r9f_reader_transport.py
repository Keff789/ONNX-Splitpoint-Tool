"""Normal early metadata staging, with only SSH/SCP/SDK executables substituted."""
import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.runners import native_split_quality_runtime as rt
from test_v283_r9f_metadata import sdk, fixture_contract


@pytest.fixture
def remote(tmp_path, sdk, monkeypatch):
    tools = tmp_path / 'transport'
    tools.mkdir()
    calls = tmp_path / 'transport_calls.jsonl'
    body = '''
import json, os, sys, shutil
from pathlib import Path
with Path(os.environ['TEST_TRANSPORT_CALLS']).open('a') as f:
    f.write(json.dumps(sys.argv)+'\\n')
if Path(sys.argv[0]).name == 'scp':
    fault = os.environ.get('TEST_TRANSFER_FAULT')
    if fault == 'exit': raise SystemExit(23)
    shutil.copyfile(sys.argv[-2], sys.argv[-1].split(':', 1)[1])
    if fault == 'receiver': Path(sys.argv[-1].split(':', 1)[1]).write_bytes(b'wrong receiver bytes')
    if fault == 'source': Path(sys.argv[-2]).write_bytes(b'changed local bytes')
    if fault == 'source_deleted': Path(sys.argv[-2]).unlink()
    if fault == 'cancel': Path(os.environ['TEST_CANCEL_MARKER']).touch()
else:
    pos = sys.argv.index('controlled-test-only')
    os.execvp('bash', ['bash', '-c', ' '.join(sys.argv[pos+1:])])
'''
    for name in ('ssh', 'scp'):
        path = tools / name
        path.write_text('#!' + sys.executable + '\n' + body)
        path.chmod(0o755)
    monkeypatch.setenv('PATH', str(tools) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('TEST_TRANSPORT_CALLS', str(calls))
    # Real principal interpreter has no vendor SDK; only the target reader does.
    monkeypatch.setenv('HAILO_PY', sys.executable)
    activate = tmp_path / 'activate'
    activate.write_text('export PATH=' + shlex.quote(str(sdk[0])) + ':"$PATH"\n')
    base = tmp_path / 'remote_base'
    base.mkdir()
    target = {'id': 'configured_setup', 'accelerator': 'hailo10h', 'remote': {
        'enabled': True, 'host': 'controlled-test-only', 'remote_venv': str(activate),
        'remote_base_dir': str(base)}}
    return {'targets': [target], 'observations': {}, 'artifact_store_root': str(tmp_path/'empty_store')}, base, calls


def test_normal_remote_reader_without_binding_or_remote_reference(tmp_path, remote, sdk):
    args, receipt = fixture_contract(tmp_path)
    context, base, calls = remote
    result = rt.resolve_hailo_output_suitability(**args, metadata_context=context)
    assert result['status'] == 'INCOMPATIBLE', result
    assert result['output_metadata']['part1_artifact_sha256'] == receipt['hef_sha256']
    assert result['output_metadata']['device_opened'] is False
    assert result['metadata_source']['kind'] == 'remote_hef_reader'
    assert result['metadata_source']['staged'] is True
    assert result['metadata_source']['cleanup']['ok'] is True
    assert not Path(result['metadata_source']['path']).exists()
    assert not (tmp_path/'native_split_quality_binding.json').exists()
    assert not list(base.iterdir())
    before = calls.read_text()
    assert len(before.splitlines()) == 4
    assert rt.resolve_hailo_output_suitability(**args, metadata_context=context) == result
    assert calls.read_text() == before
    assert len((sdk[0]/'reads').read_text().splitlines()) == 1


def test_conflicting_targets_do_not_choose_arbitrary_host(tmp_path, remote):
    import copy
    args, _ = fixture_contract(tmp_path)
    context, base, calls = remote
    second = copy.deepcopy(context['targets'][0]); second['id'] = 'another_setup'
    context['targets'].append(second)
    result = rt.resolve_hailo_output_suitability(**args, metadata_context=context)
    assert result['status'] == 'UNKNOWN'
    assert 'remote_setup_ambiguous' in result['reason']
    assert not calls.exists()


def test_generator_stages_without_binding_and_backfills_only_h10(tmp_path, remote, sdk, monkeypatch):
    from test_v283_r9f_metadata import test_normal_generator_real_resolver_and_ranked_backfill as exercise
    context, base, calls = remote
    exercise(tmp_path, sdk, monkeypatch, False, metadata_targets=context['targets'])
    # The next graph boundary is a feature contract; it needs no score reader.
    assert len(calls.read_text().splitlines()) == 4
    assert not list(base.iterdir())
    assert not list(tmp_path.rglob('native_split_quality_binding.json'))


@pytest.mark.parametrize('fault', ['exit', 'receiver', 'source', 'source_deleted'])
def test_transfer_fault_never_rejects_and_cleans(tmp_path, remote, monkeypatch, fault):
    args, _ = fixture_contract(tmp_path)
    context, base, calls = remote
    monkeypatch.setenv('TEST_TRANSFER_FAULT', fault)
    result = rt.resolve_hailo_output_suitability(**args, metadata_context=context)
    assert result['status'] == 'UNKNOWN', result
    assert result['metadata_source']['cleanup']['ok'] is True
    assert not list(base.iterdir())
    if fault == 'source': assert 'local_hef_changed' in result['reason']
    phases = [e['phase'] for e in result['metadata_source']['transport_events']]
    assert phases.count('transfer') == 1
    if fault == 'exit': assert 'reader' not in phases


@pytest.mark.parametrize('fault', ['missing', 'timeout', 'fragmented', 'ambiguous', 'mapping', 'rounding', 'shape', 'cleanup', 'both'])
def test_remote_reader_faults_keep_primary_and_cleanup(tmp_path, remote, sdk, monkeypatch, fault):
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseRegistry, bind_remote_process_registry
    args, _ = fixture_contract(tmp_path, {'shape':[1,12,11]} if fault=='shape' else None)
    context, base, calls = remote
    bootstrap = sdk[0]/'bootstrap.py'
    if fault == 'missing':
        (sdk[0]/'hailo_platform.py').unlink()
    elif fault == 'timeout':
        bootstrap.write_text('import time\ntime.sleep(30)\n')
        monkeypatch.setattr(rt, '_HEF_METADATA_TIMEOUT_S', 1)
    elif fault in ('fragmented', 'ambiguous'):
        bootstrap.write_text('print(' + repr('{"outputs":[' if fault == 'fragmented' else '{"outputs":[],"outputs":[]}') + ')\n')
    elif fault in ('mapping', 'rounding'):
        sdk_source = sdk[0]/'hailo_platform.py'
        s = sdk_source.read_text()
        if fault == 'mapping': s = s.replace('return [self._info(True).name]', "return ['wrong-stream']")
        else: s = s.replace("if 'rounding' in r:q.rounding=r['rounding']", "q.rounding = 'ceil' if native else 'floor'")
        sdk_source.write_text(s)
    elif fault in ('cleanup', 'both'):
        # A foreign unexpected file prevents rmdir: report it, do not recurse.
        bootstrap.write_text("from pathlib import Path\nimport sys\nPath(sys.argv[4]).parent.joinpath('foreign').write_text('preserve')\n" +
                            ("raise RuntimeError('primary_reader_failure')\n" if fault == 'both' else bootstrap.read_text()))
    registry = RemoteProcessLeaseRegistry()
    with bind_remote_process_registry(registry):
        result = rt.resolve_hailo_output_suitability(**args, metadata_context=context)
    assert result['status'] == 'UNKNOWN', result
    source = result['metadata_source']
    assert source['cleanup']['ok'] is (fault not in ('cleanup', 'both'))
    if fault in ('cleanup', 'both'):
        assert registry.cancelled
        assert list(base.glob('*/foreign'))
        assert ('remote_reader_exit' if fault == 'both' else 'cleanup_failed') in result['reason']
    else:
        assert not list(base.iterdir())
    assert sum(e['phase'] == 'reader' for e in source['transport_events']) == 1


def test_cancel_after_transfer_closes_owned_staging(tmp_path, remote, monkeypatch):
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseRegistry, bind_remote_process_registry
    marker = tmp_path/'cancelled'
    class Registry(RemoteProcessLeaseRegistry):
        @property
        def cancelled(self): return marker.exists()
    args, _ = fixture_contract(tmp_path)
    context, base, _ = remote
    monkeypatch.setenv('TEST_TRANSFER_FAULT', 'cancel')
    monkeypatch.setenv('TEST_CANCEL_MARKER', str(marker))
    with bind_remote_process_registry(Registry()):
        result = rt.resolve_hailo_output_suitability(**args, metadata_context=context)
    assert result['status'] == 'UNKNOWN'
    assert result['metadata_source']['cleanup']['ok'] is True
    assert not list(base.iterdir())


def test_changed_source_cannot_reuse_remote_observation(tmp_path, remote):
    args, _ = fixture_contract(tmp_path)
    context, _, calls = remote
    assert rt.resolve_hailo_output_suitability(**args,metadata_context=context)['status']=='INCOMPATIBLE'
    before=calls.read_text()
    args['part1'].write_bytes(b'changed HEF identity')
    result=rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert result['status']=='UNKNOWN' and result['reason']=='exact_hef_source_receipt_unavailable'
    assert calls.read_text()==before


def test_disabled_target_not_selected(tmp_path, remote):
    args, _ = fixture_contract(tmp_path)
    context, _, calls = remote
    context['targets'][0]['enabled']=False
    result=rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert result['status']=='UNKNOWN' and not calls.exists()


def test_receipt_failure_prevents_any_transport(tmp_path, remote):
    args, _ = fixture_contract(tmp_path)
    args['part1'].write_bytes(b'foreign bytes')
    context, _, calls = remote
    result = rt.resolve_hailo_output_suitability(**args, metadata_context=context)
    assert result['reason'] == 'exact_hef_source_receipt_unavailable'
    assert not calls.exists()


@pytest.mark.parametrize('exists', [True, False])
def test_bound_reference_checked_today_or_staged_without_search(tmp_path, remote, exists):
    args, receipt = fixture_contract(tmp_path)
    context, base, calls = remote
    path = tmp_path/'previously_bound.hef'
    if exists: path.write_bytes(args['part1'].read_bytes())
    context.update(setup_id='configured_setup',part1_runtime={
        'path':str(path),'sha256':receipt['hef_sha256'],'size_bytes':receipt['hef_size_bytes']})
    result = rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert result['status'] == 'INCOMPATIBLE', result
    assert result['metadata_source']['staged'] is (not exists)
    assert len(calls.read_text().splitlines()) == (1 if exists else 6)
    assert path.exists() is exists
    assert not list(base.iterdir())


def test_staging_failure_is_unknown_with_both_diagnostics(tmp_path, remote):
    args, _ = fixture_contract(tmp_path)
    context, base, _ = remote
    base.rmdir()
    result = rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert result['status'] == 'UNKNOWN' and 'staging_exit' in result['reason']
    assert result['metadata_source']['cleanup']['ok'] is False


def test_current_lease_scope_reused_without_platform_lock(tmp_path, remote):
    import fcntl
    from onnx_splitpoint_tool.remote.process_lease import (
        RemoteProcessLeaseRegistry, RemoteProcessLeaseScope, bind_remote_process_registry)
    args, _ = fixture_contract(tmp_path)
    context, base, calls = remote
    registry=RemoteProcessLeaseRegistry()
    scope=RemoteProcessLeaseScope('already-running-workflow','existing-session',str(tmp_path/'leases'))
    registry.configure_journal(scope=scope,journal_dir=tmp_path/'journal')
    with (tmp_path/'platform.lock').open('a') as lock, bind_remote_process_registry(registry):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result=rt.resolve_hailo_output_suitability(**args,metadata_context=context)
    assert result['status']=='INCOMPATIBLE',result
    assert scope.run_sha256 in calls.read_text()
    assert registry.active_count()==0
    assert not list(base.iterdir())


@pytest.mark.parametrize('fault', [None, 'quota', 'membership', 'unknown_contract', 'frozen'])
def test_reconciliation_uses_existing_backend_quota_not_union_length(fault):
    from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import reconcile_candidate_plan_after_generation
    plan={'requested_cases':1,'selected_candidates':[{'case_id':'b003','boundary':3}]}
    accepted=[{'case_id':'b003','boundary':3,'backend_selection_contracts':['trt']},
              {'case_id':'b004','boundary':4,'backend_selection_contracts':['h10']}]
    state={'enabled':True,'quota':1,'contract_definitions':[{'id':'trt'},{'id':'h10'}],
           'contracts':[{'id':'trt','selected_case_ids':['b003']},{'id':'h10','selected_case_ids':['b004']}]}
    if fault=='quota':state['contracts'][1]['selected_case_ids'].append('b003')
    if fault=='membership':accepted[1]['backend_selection_contracts']=['trt']
    if fault=='unknown_contract':state['contracts'][1]['id']='foreign'
    if fault=='frozen':state=None  # Historical/frozen caller carries no active backend selection.
    kwargs=dict(prediction={'candidates':accepted},accepted_cases=accepted,rejected_cases=[],backend_selection_state=state)
    if fault:
        with pytest.raises(ValueError):reconcile_candidate_plan_after_generation(plan,**kwargs)
    else:
        final,trace=reconcile_candidate_plan_after_generation(plan,**kwargs)
        assert final['requested_cases']==1
        assert trace['final_case_ids']==['b003','b004']
        assert final['selected_candidates'][1]['backend_selection_contracts']==['h10']
