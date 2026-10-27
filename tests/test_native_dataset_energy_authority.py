"""Fresh Native-Q authority is separate from legacy manifests and Full pairs."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool import native_split_quality_authority as authority
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_energy_quality_admission import validate_energy_quality_admission_axes


def put(path, value):
    path.write_text(json.dumps(value))
    return authority._dataset_descriptor(path)


@pytest.fixture
def closed(tmp_path, monkeypatch):
    """The existing Native request validator is its own tested API boundary."""
    from onnx_splitpoint_tool import native_command_contract as commands
    from onnx_splitpoint_tool import native_dataset_quality_consumer as consumer
    from onnx_splitpoint_tool import quality_service
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    command = {'contract_sha256': 'c' * 64, 'native_split_quality_binding': {'binding_sha256': 'b' * 64}}
    timed = {'ok': True, 'completed_frames': 1000, 'native_command_contract': command,
             'endpoint_contract_hash': 'a' * 64, 'output_endpoint_id': 'raw',
             'native_command_contract_sha256': command['contract_sha256'],
             'native_split_quality_binding': command['native_split_quality_binding'],
             'completion_execution_contract': {'actual_completed_operation': True}}
    timed_descriptor = put(tmp_path / 'timed.json', timed)
    row = {'backend': 'hailo8_to_trt', 'model': 'yolov7_paper', 'case': 'b009',
           'repetition_count_valid': 3, 'fps_repetition_samples': [10., 11., 12.],
           'setup_id': 'hailo8_setup', 'precision': 'uint8_dequant_fp16', 'comparison_backend': 'hailo8',
           'report': timed_descriptor['path'], 'native_command_contract': command,
           'native_command_contract_sha256': command['contract_sha256'],
           'native_split_quality_binding': command['native_split_quality_binding']}
    candidate = put(tmp_path / 'candidate.json', {'records': []})
    completion = put(tmp_path / 'completion.json', {'records': []})
    freeze = put(tmp_path / 'sourcefreeze.json', {'actual_published_source': True})
    producer = {'schema': 'onnx-splitpoint/native-dataset-quality-producer',
        'model': {'source_onnx_sha256': 'a' * 64},
        'backend': row['backend'], 'model_id': row['model'], 'case_id': row['case'], 'setup_id': row['setup_id'],
        'runtime_precision_identity': row['precision'], 'eval_run_id': 'genuine-new-capture',
        'timed_fps_authority': timed_descriptor, 'prior_artifact_selection': command['native_split_quality_binding'],
        'candidate_execution_completion_contract': timed['completion_execution_contract'], 'completion_evidence': completion}
    manifest = {'model_id': row['model'], 'producer_identity': producer, 'policy_sha256': '9' * 64,
                'metric_gate_config': {'metric': 'map_50_95', 'non_inferiority_margin': .01}, 'candidate': candidate}
    request = put(tmp_path / 'composed_request.json', manifest)
    config = put(tmp_path / 'capture_config.json', {'timed_fps_authority': timed_descriptor})
    terminal = put(tmp_path / 'terminal.json', {'status': 'completed', 'capture_completed': True,
        'capture_returncode': 0, 'collection_returncode': 0, 'completed_images': 5000, 'request': request})
    handoff = put(tmp_path / 'handoff.json', {'request': request, 'completed_images': 5000, 'config': config, 'terminal': terminal})
    identity = {'identity_valid': True, 'producer_identity_validated': True, 'producer_identity': producer,
        'producer_binding_eligible': True, 'source_request_sha256': 'sha256:' + request['sha256'],
        'eval_run_id': producer['eval_run_id'], 'model_id': row['model'], 'case_id': row['case'],
        'setup_id': row['setup_id'], 'runtime_precision_identity': row['precision'],
        **{key: 'a' * 64 for key in ('endpoint_contract_hash', 'quality_contract_sha256',
          'preprocessing_contract_sha256', 'decoder_contract_sha256', 'nms_contract_sha256', 'source_model_sha256')}}
    result = {'schema': 'onnx-splitpoint/management-paired-quality-result', 'schema_version': 1,
        'status': 'completed', 'technical_status': 'completed', 'decision': 'pass',
        'execution_location': 'central_management', 'producer_binding_eligible': True,
        'producer_identity': producer, 'request_identity': identity, 'model_id': row['model'], 'case_id': row['case'],
        'source_setup_id': row['setup_id'], 'runtime_precision_identity': row['precision'], 'backend': row['backend'],
        'source_request_sha256': identity['source_request_sha256'], 'policy_sha256': manifest['policy_sha256'],
        'metric_gate_config': manifest['metric_gate_config'],
        'native_dataset_producer_source': {'source_freeze': freeze},
        'primary': {'n': 5000, 'bootstrap_repetitions': 1000, 'candidate': .5, 'reference': .5, 'ci_low': 0., 'ci_high': 0.},
        'management_cpu_reference': {}, 'validation_image_ids_sha256': 'd' * 64,
        'validation_ground_truth_sha256': 'e' * 64, 'validation_dataset_sha256': 'f' * 64,
        'model_sha256': 'a' * 64,
        **{key: identity[key] for key in ('endpoint_contract_hash','quality_contract_sha256','preprocessing_contract_sha256',
                                         'decoder_contract_sha256','nms_contract_sha256')}}
    result_descriptor = put(tmp_path / 'normal_result.json', result)
    shutdown = put(tmp_path / 'shutdown.json', {'finished': True, 'admission_closed': True,
        'manager_alive': False, 'dispatcher_alive': False, 'queue_size': 0, 'inflight_keys': [], 'group_keys': [],
        'workers': [{'alive': False, 'exitcode': 0}]})
    service_terminal = put(tmp_path / 'service_terminal.json', {'completed_case_keys': ['native_case'],
        'controller_returncode': 130, 'controller_status': 'interrupted_for_authorized_source_transition'})
    index = {'normal_native_admission': 'completed_verified_actual_native_producer', 'normal_result': result_descriptor,
        'normal_service_shutdown': shutdown, 'normal_source_epoch_terminal': service_terminal,
        'shared_pool_terminal_pending': False, 'case_key': 'native_case',
        'actual_capture_handoff': handoff,
        'source_request': request, 'candidate': candidate, 'completion_evidence': completion,
        'producer_source_freeze': freeze, 'consumer_source_freeze': freeze}
    descriptor = put(tmp_path / 'NORMAL_COMPLETED_CASE.json', index)
    calls = []
    @contextmanager
    def source_scope(value):
        assert value == freeze
        calls.append('source_scope')
        yield {'source_freeze': freeze}
    monkeypatch.setattr(consumer, 'published_native_producer_source_scope', source_scope, raising=False)
    monkeypatch.setattr(quality_service, 'quality_request_from_manifest',
        lambda *args, **kwargs: (calls.append('normal_request') or SimpleNamespace(
            candidate_records=[None] * 5000, repetitions=1000, seed=20260710, confidence_level=.95)))
    monkeypatch.setattr(EvaluationWorkflowRunner, '_quality_request_identity', lambda *a, **kw: deepcopy(identity))
    monkeypatch.setattr(commands, 'verify_native_energy_command_contract', lambda *a, **kw: (deepcopy(command), 'verified'))
    monkeypatch.setattr(authority, '_dataset_completion_same_operation', lambda new, old: new == old or pytest.fail('operation changed'))
    return SimpleNamespace(row=row, command=command, result=result, index=index, descriptor=descriptor,
                           tmp=tmp_path, calls=calls)


def refresh(closed):
    closed.index['normal_result'] = put(closed.tmp / 'normal_result.json', closed.result)
    closed.descriptor = put(closed.tmp / 'NORMAL_COMPLETED_CASE.json', closed.index)


@pytest.mark.parametrize('decision', ['pass', 'fail', 'inconclusive', 'reference_close', 'accuracy_loss', 'not_estimable'])
def test_closed_quality_keeps_own_result_without_legacy_workflow(closed, decision):
    closed.result['decision'] = decision
    refresh(closed)
    verified = authority.resolve_native_dataset_energy_quality(closed.descriptor, closed.row, closed.command)
    assert verified['validation']['accuracy_gate_pass'] is (decision in {'pass', 'reference_close'})
    assert verified['evidence']['fresh_eval_run_id'] == 'genuine-new-capture'
    assert verified['evidence']['authority_kind'] == 'native_dataset_quality'
    assert 'run_manifest' not in verified['evidence']
    assert closed.calls == ['source_scope', 'normal_request']


@pytest.mark.parametrize('fault', ['wrong_identity', 'incomplete', 'wrong_N', 'old_producer', 'request_drift', 'service_open', 'wrong_fps_repetitions'])
def test_wrong_identity_and_incomplete_quality_fail_closed(closed, fault):
    row = deepcopy(closed.row)
    if fault == 'wrong_identity': row['case'] = 'b043'
    if fault == 'incomplete': closed.result['technical_status'] = 'failed'
    if fault == 'wrong_N': closed.result['primary']['n'] = 4999
    if fault == 'old_producer': closed.result['producer_identity'] = {'schema': 'legacy-old-quality'}
    if fault == 'request_drift': closed.result['source_request_sha256'] = '0' * 64
    if fault == 'service_open': closed.index['shared_pool_terminal_pending'] = True
    if fault == 'wrong_fps_repetitions': row['repetition_count_valid'] = 1
    refresh(closed)
    with pytest.raises(ValueError):
        authority.resolve_native_dataset_energy_quality(closed.descriptor, row, closed.command)


def test_executor_revalidates_immutable_fresh_sources(closed):
    verified = authority.resolve_native_dataset_energy_quality(closed.descriptor, closed.row, closed.command)
    evidence = verified['evidence']
    row = {**closed.row, 'native_dataset_quality_FPS_source': closed.row,
        'native_split_energy_quality_binding': evidence,
        'native_split_energy_quality_binding_sha256': evidence['evidence_sha256'],
        'source_request_sha256': evidence['source_request_sha256'],
        'successful_command_contract_sha256': closed.command['contract_sha256']}
    assert authority.verify_native_dataset_energy_binding(row)[1] == 'fresh_native_dataset_quality_binding_verified'
    (closed.tmp / 'normal_result.json').write_text('{}')
    with pytest.raises(ValueError): authority.verify_native_dataset_energy_binding(row)


def test_normal_own_energy_scope_never_releases_a_full_comparison(closed):
    spec = importlib.util.spec_from_file_location('native_energy_planner_test', Path(__file__).parents[1] / 'scripts/native_producer_energy_plan.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    fresh = authority.resolve_native_dataset_energy_quality(closed.descriptor, closed.row, closed.command)
    admission, status = module._prepair_energy_quality_admission(
        {**closed.row, '_native_dataset_quality': fresh}, fresh['validation'], setup=closed.row['setup_id'],
        command_contract_sha256=closed.command['contract_sha256'], screening_only=False, smoke_diagnostic=False,
        historical_diagnostic_only=False)
    assert admission is not None, status
    assert admission['admission_scope'] == 'native_dataset_energy'
    assert admission['diagnostic_only'] is False
    assert admission['energy_claim_eligible'] is True
    assert admission['claim_comparable'] is False
    assert admission['full_baseline_comparison_released'] is False
    validate_energy_quality_admission_axes(admission)
    admission['claim_comparable'] = True
    with pytest.raises(ValueError): validate_energy_quality_admission_axes(admission)


def test_json_row_cannot_self_declare_a_verified_fresh_authority(closed):
    spec = importlib.util.spec_from_file_location('native_energy_untrusted_test', Path(__file__).parents[1] / 'scripts/native_producer_energy_plan.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    fresh = authority.resolve_native_dataset_energy_quality(closed.descriptor, closed.row, closed.command)
    untrusted = json.loads(json.dumps(fresh))
    evidence, _ = module._split_quality_energy_evidence(
        {**closed.row, '_native_dataset_quality': untrusted}, closed.command,
        {'valid': False, 'mode': 'invalid', 'native_split_quality_required': True})
    assert evidence is None
