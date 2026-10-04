"""Real Quality writer -> copied transfer bytes -> Full runtime guards."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import pytest

from onnx_splitpoint_tool.trt_quality_chain import load_producer_set_from_central_quality_summary
from scripts import native_full_baseline_eval_runner as runtime
from tests.test_v269d_trt_quality_first_runtime import _write_quality_producer_set, _digest, _namespace
from tests.test_v269d_trt_quality_chain import _summary, _result
from tests.test_v269d_trt_central_quality_producer import _producer
from tests.test_v275_workflow_energy_contract import test_vendor_full_quality_binding_is_hash_sealed_and_exact as vendor_writer_roundtrip


def _producer_writer(tmp_path):
    suite, producer_file, producer = _write_quality_producer_set(tmp_path)
    complete = _producer()
    for key in ('dataset', 'preprocessing', 'quality_record_endpoint',
                'quality_record_endpoint_contract_sha256', 'endpoint_attestor',
                'quality_contract', 'quality_contract_sha256', 'preprocessing_contract_sha256',
                'implementation_runner_sha256', 'vendored_endpoint_attestor_sha256'):
        producer[key] = complete[key]
    producer.pop('producer_identity_sha256')
    producer['producer_identity_sha256'] = _digest(producer)
    summary = tmp_path / 'central.json'
    failed = _result(copy.deepcopy(producer))
    failed['model_id'] = 'unavailable-model'
    failed['request_identity']['model_id'] = 'unavailable-model'
    failed['producer_identity']['model_id'] = 'unavailable-model'
    failed['request_identity']['producer_identity']['model_id'] = 'unavailable-model'
    failed['status'] = 'failed'
    summary.write_text(json.dumps(_summary([_result(producer), failed])))
    payload = load_producer_set_from_central_quality_summary(
        summary, eval_run_id='eval-a', setup_id='setup-a', model_ids=['resnet50'])
    producer_file.write_text(json.dumps(payload))
    return suite, producer_file, producer


@pytest.mark.parametrize('attempt', [False, True])
def test_trt_writer_payload_runtime_campaign_and_attempt(tmp_path, attempt):
    suite, producer_file, producer = _producer_writer(tmp_path)
    root = producer_file.parent.parent
    if attempt:
        root = root / 'attempts' / 'attempt-a'
        transferred = root / 'quality_first' / producer_file.name
        transferred.parent.mkdir(parents=True)
        shutil.copyfile(producer_file, transferred)
        producer_file = transferred
        suite = root / 'resnet50' / 'benchmark_set'
        suite.mkdir(parents=True)
    ns = _namespace(producer_file)
    ns.eval_run_id = 'eval-a'
    ns.attempt_id = 'attempt-a' if attempt else ''
    accepted, status, _ = runtime._quality_first_trt_producer_identity(suite, 'resnet50', ns)
    assert accepted == producer and status == 'quality_first_identity_verified_exact'
    for field, wrong in [('eval_run_id', 'wrong-run'), ('setup_id', 'wrong-setup'), ('attempt_id', 'wrong-attempt')]:
        bad = copy.deepcopy(ns)
        setattr(bad, field, wrong)
        assert runtime._quality_first_trt_producer_identity(suite, 'resnet50', bad)[0] is None
    assert runtime._quality_first_trt_producer_identity(suite, 'wrong-model', ns)[0] is None
    engine = Path(producer['engine']['path'])
    engine.write_bytes(engine.read_bytes() + b'changed')
    assert runtime._quality_first_trt_producer_identity(suite, 'resnet50', ns)[0] is None


@pytest.mark.parametrize('backend,setup,source,prefix', [
    ('deepx', 'orin_nx_deepx_m1_01', 'deepx_full', 'deepx_dxnn_sha256'),
    ('hailo8', 'orin_nx_hailo8_01', 'hailo8', 'hailo_hef_sha256'),
    ('hailo10h', 'orin_nx_hailo10_01', 'hailo10', 'hailo_hef_sha256'),
])
def test_vendor_original_writer_and_runtime_attempt_scope(tmp_path, backend, setup, source, prefix):
    # This existing regression performs the real request writer and runtime
    # attach, with wrong model/input/compiled artifact/preprocessing controls.
    vendor_writer_roundtrip(tmp_path, backend, backend, source, setup, 'yolov7_paper', prefix)
    campaign = tmp_path / 'run_275'
    original = campaign / 'quality_first/vendor_full_quality_request_binding_set.json'
    from onnx_splitpoint_tool.workflow.runner import _native_full_quality_binding_set_v275
    derived_summary = campaign / 'reports/attempts/local_statistics/central_quality_summary.json'
    derived_summary.parent.mkdir(parents=True)
    shutil.copyfile(campaign / 'quality_management/central_quality_summary.json', derived_summary)
    derived_set, errors = _native_full_quality_binding_set_v275(
        derived_summary, eval_run_id=campaign.name, setup_id=setup, producer=backend,
        full_backends=[backend], model_ids=['yolov7_paper'], case_release_run_root=campaign)
    assert not errors
    assert derived_set['bindings_by_backend_model'] == json.loads(original.read_text())['bindings_by_backend_model']
    rejected, errors = _native_full_quality_binding_set_v275(
        derived_summary, eval_run_id='wrong', setup_id=setup, producer=backend,
        full_backends=[backend], model_ids=['yolov7_paper'], case_release_run_root=campaign)
    assert rejected['complete'] is False and 'central_quality_summary_role_path_mismatch' in errors
    root = campaign / 'attempts/attempt-vendor'
    transferred = root / 'quality_first' / original.name
    transferred.parent.mkdir(parents=True)
    shutil.copyfile(original, transferred)
    ns = argparse.Namespace(root=str(root), eval_run_id='run_275', attempt_id='attempt-vendor',
        setup_id=setup, comparison_backend=backend, quality_request_binding_set=str(transferred))
    payload, status = runtime._load_full_quality_binding_set(ns)
    assert status == 'quality_request_binding_set_verified'
    assert payload == json.loads(original.read_text())
    for field, wrong in [('eval_run_id', 'wrong'), ('attempt_id', ''), ('setup_id', 'wrong'), ('comparison_backend', 'wrong')]:
        bad = copy.deepcopy(ns); setattr(bad, field, wrong)
        assert runtime._load_full_quality_binding_set(bad)[0] == {}
    forged = copy.deepcopy(payload)
    forged['eval_run_id'] = 'wrong'
    forged.pop('binding_set_sha256'); forged['binding_set_sha256'] = _digest(forged)
    transferred.write_text(json.dumps(forged))
    assert runtime._load_full_quality_binding_set(ns)[0] == {}


def test_runtime_mirror_exact():
    source = Path(runtime.__file__)
    mirror = source.parents[1] / 'onnx_splitpoint_tool/resources/remote_scripts' / source.name
    assert source.read_bytes() == mirror.read_bytes()


def test_normal_metadata_only_reentry_still_writes_binding_sets(tmp_path, monkeypatch):
    from tests.test_v2911_native_completion import test_normal_native_stage_only_dispatches_missing_leaf_and_reentry_skips_it as stage_roundtrip
    from onnx_splitpoint_tool.workflow import runner as workflow
    snapshots = []
    real_write = workflow.write_json
    def record_write(path, value, *args, **kwargs):
        if Path(path).name == 'native_producer_stage.json':
            snapshots.append(copy.deepcopy(value))
        return real_write(path, value, *args, **kwargs)
    monkeypatch.setattr(workflow, 'write_json', record_write)
    stage_roundtrip(tmp_path, monkeypatch)
    reused = [s for s in snapshots if s.get('selective_completion', {}).get('pending_row_count') == 0]
    assert reused
    assert all(s['vendor_full_quality_requests']['required'] for s in reused)
    assert all(s['vendor_full_quality_requests']['binding_sets_by_setup'] for s in reused)
    assert all(not s.get('started_performance_count') for s in reused)
    full_commands = [step['cmd'] for s in snapshots for row in s.get('backend_results', [])
                     for step in row.get('steps', []) if step.get('name') == 'run_native_full_baselines']
    assert full_commands
    assert all('--eval-run-id controlled_native_completion' in cmd and '--attempt-id ' in cmd for cmd in full_commands)
    assert all('scripts/native_full_baseline_eval_runner_' in cmd for cmd in full_commands)
    assert all('scripts/native_full_baseline_eval_runner.py' not in cmd for cmd in full_commands)


def test_attempt_script_sync_and_real_command_writer_preserve_legacy_path(tmp_path, monkeypatch):
    import hashlib
    import subprocess
    from onnx_splitpoint_tool.workflow import runner as workflow
    from scripts.native_producer_energy_plan import _full_preflight_argv
    calls = []
    script_sha = hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()
    def transport(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps({'sha256': script_sha, 'missing_tokens': []}), '')
    scoped_name = 'native_full_baseline_eval_runner_attempt42.py'
    workflow._sync_remote_script_v60i(ssh='fixture-only', remote_tool_dir='/remote/tool',
        script_name='native_full_baseline_eval_runner.py', remote_script_name=scoped_name,
        required_tokens=('--attempt-id', '--eval-run-id'), process_runner=transport)
    uploads = [cmd[-1] for cmd in calls if cmd[0] == 'rsync']
    assert uploads == ['fixture-only:/remote/tool/scripts/' + scoped_name]
    assert not any(value.endswith('/scripts/native_full_baseline_eval_runner.py') for value in uploads)
    renamed = tmp_path / scoped_name
    shutil.copyfile(runtime.__file__, renamed)
    monkeypatch.setattr(runtime, '__file__', str(renamed))
    root = tmp_path / 'campaign/attempts/attempt42'
    suite = root / 'model/benchmark_set'; suite.mkdir(parents=True)
    ns = _namespace(root / 'quality_first/tensorrt_quality_producer_set.json')
    contract = runtime._full_command_contract(row={'backend': 'unsupported', 'model': 'model'},
        root=root, benchmark_set=suite, model='model', backend_arg='unsupported', ns=ns)
    assert contract['runner'] == 'scripts/' + scoped_name
    assert contract['runner_sha256'] == script_sha
    command = _full_preflight_argv(contract, remote_tool_dir='/remote/tool',
        authoritative_root=str(root), preflight_nonce='test', preflight_attestation='/remote/attestation', max_age_s=300)
    assert command[2] == '/remote/tool/scripts/' + scoped_name
    assert command[command.index('--expected-runner-path') + 1] == command[2]


def test_normal_stage_real_trt_writer_transfer_and_attempt_consumer(tmp_path, monkeypatch):
    import shlex
    import subprocess
    from onnx_splitpoint_tool.workflow import runner as workflow
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
    from tests.test_v2911_native_completion import write
    suite, original_set, producer = _producer_writer(tmp_path)
    run_dir = original_set.parent.parent
    central = run_dir / 'quality_management/central_quality_summary.json'
    central.parent.mkdir()
    shutil.copyfile(tmp_path / 'central.json', central)
    local_suite = run_dir / 'models/resnet50/benchmark_set'
    write(local_suite / 'benchmark_set.json', {'task': 'classification', 'cases': []})
    options = workflow.WorkflowOptions(profile='', out=str(tmp_path))
    options.resume_missing = True
    runner = workflow.EvaluationWorkflowRunner(options)
    runner.run_id = 'eval-a'; runner.run_dir = run_dir
    runner._supplement_original_root = tmp_path / 'original'
    write(runner._supplement_original_root / 'reports/native_producer_summary.json', {'rows': []})
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id), journal_dir=tmp_path / 'leases')
    runner.manifest = {'models': {'resnet50': {}}}
    runner.profile_payload = {'execution_preset': {'id': 'smoke'},
        'run_profiles': [{'id': 'native_full_tensorrt', 'type': 'full', 'backend': 'tensorrt', 'expected_setup_id': 'setup-a'}]}
    remote_campaign = tmp_path / 'remote/eval-a'
    cfg = {'enabled': True, 'models': ['resnet50'], 'backends': ['deepx'], 'split_backends': [],
        'precision': 'fp16', 'copy_benchmarksets': False, 'cleanup_remote_native_root': False,
        'remotes': {'deepx': {'ssh': 'fixture-only', 'setup_id': 'setup-a', 'remote_root': str(remote_campaign)}},
        'validation': {'enabled': False}, 'energy': {'enabled': False},
        'full_baselines': {'enabled': True, 'backends_by_producer': {'deepx': ['tensorrt']}}}
    payloads = []
    def transport(command, *args, **kwargs):
        if kwargs.get('label') == 'quality-first-transfer:deepx':
            source = Path(command[-2]); target = Path(command[-1].split(':', 1)[1])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            payloads.append(target)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(workflow, 'run_streaming', transport)
    monkeypatch.setattr(workflow, 'normalize_hardware_targets', lambda *_: [])
    monkeypatch.setattr(workflow, 'benchmark_set_postcondition_v60v', lambda p: {'valid': True, 'selected_suite_dir': str(p)})
    monkeypatch.setattr(workflow, '_sync_remote_script_v60i', lambda *a, **k: [])
    monkeypatch.setattr(workflow, 'native_remote_package_closure', lambda: [])
    monkeypatch.setattr(runner, '_finish_native_direct_remote_lease', lambda *a, **k: None)
    monkeypatch.setattr(runner, '_native_producer_config', lambda: cfg)
    runner._stage_run_native_producers()
    stage = json.loads((run_dir / 'reports/native_producer_stage.json').read_text())
    full_steps = [s for b in stage['backend_results'] for s in b.get('steps', []) if s.get('name') == 'run_native_full_baselines']
    assert len(full_steps) == 1, stage
    tokens = shlex.split(full_steps[0]['cmd'])
    ns = _namespace(payloads[0])
    for flag, attr in [('--root', 'root'), ('--eval-run-id', 'eval_run_id'), ('--attempt-id', 'attempt_id'),
                       ('--setup-id', 'setup_id'), ('--trt-quality-producer-json', 'trt_quality_producer_json')]:
        setattr(ns, attr, tokens[tokens.index(flag) + 1])
    remote_suite = Path(ns.root) / 'resnet50/benchmark_set'; remote_suite.mkdir(parents=True)
    accepted, status, _ = runtime._quality_first_trt_producer_identity(remote_suite, 'resnet50', ns)
    assert accepted == producer and status == 'quality_first_identity_verified_exact'
    assert ns.eval_run_id == runner.run_id and ns.attempt_id == runner.session_id
