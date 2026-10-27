"""Real CPU duration hook/serial validator; accelerator callbacks are fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from onnx_splitpoint_tool.runners import generic_energy, generic_only, task_completion
from onnx_splitpoint_tool import native_detection_postprocess as detection
from tests.test_generic_only_generated import execute, prepared_case, descriptor, TEMPLATE


def energy_configuration(config, output):
    """Upstream normal admission is explicit synthetic authority for this fixture."""
    old_control = config['mixed_input_control']
    config['timeout_s'] = 120
    normal = {'admission_mode': 'generic_only', 'case_key': 'synthetic-only',
              'eligible_for_generic_only_execution': True, 'native_performance_claimed': False,
              'completion_attestations': [{}, {}, {}]}
    report = output / 'upstream_normal_admission.json'; report.write_text(json.dumps({'rows': [normal]}))
    mixed = {'status': 'passed', 'checked_outside_throughput_timer': True,
             'serial_reference_count': 32, 'mixed_completed_count': 32, 'drain_completed': True,
             'stage1_execution': config['stage1_execution'], 'control_sidecar': old_control}
    performance = output / 'upstream_performance.json'
    performance.write_text(json.dumps({'identity': config['identity'], 'execution_id': config['execution_id'],
        'process_repetition_policy': 'separate_processes',
        'repetitions': [{'mixed_input_control': mixed} for _ in range(3)]}))
    source_map = {item['path']: item['sha256'] for item in config['runtime_source_files']}
    for path in [Path(generic_energy.__file__), Path(generic_only.__file__), report, performance]:
        source_map[str(path)] = descriptor(path)['sha256']
    config['runtime_source_files'] = [{'path': path, 'sha256': value} for path, value in sorted(source_map.items())]
    config['mixed_input_control'] = False
    mode = detection.create_detection_completion_runtime(config['completion_contract']).completion_runtime_mode
    contract = generic_energy.build_contract(config, source_files=config['runtime_source_files'],
        runner_path=str(TEMPLATE), completion_runtime_mode=mode)
    config['energy_capture'] = {'contract': contract, 'performance_result_sha256': descriptor(performance)['sha256'],
        'generic_only_control_reuse': {'schema': 'onnx-splitpoint/generic-only-energy-control-reuse/v1',
            'normal_admission_report': descriptor(report), 'performance_result': descriptor(performance),
            'normal_admission': normal, 'original_mixed_input_control': old_control}}
    nonce = 'synthetic-energy-only-00000000000001'
    attestation = generic_energy.build_preflight(config, nonce=nonce,
        expected_sha256=contract['command_contract_sha256'])
    path = output / 'preflight.json'; path.write_text(json.dumps(attestation))
    config['_runtime_energy'] = {'energy_preflight_nonce': nonce,
        'energy_preflight_attestation': str(path), 'energy_command_contract_sha256': contract['command_contract_sha256']}


@pytest.mark.parametrize('processor_mode', ['strict', 'auto'])
def test_generated_generic_only_duration_has_no_mixed_probe_and_uses_actual_count(tmp_path, monkeypatch, capsys, processor_mode):
    measure = task_completion.measure_split_completion
    monkeypatch.setattr(task_completion, 'measure_split_completion',
        lambda *a, **kw: measure(*a, **kw, clock=lambda: time.perf_counter_ns() * 3000))
    case = execute(monkeypatch, tmp_path, 'deepx', processor_mode, configure_energy=energy_configuration)
    row = generic_energy.validate_energy_result(case.result, case.config, result_root=case.output)
    assert row['warmup_completed_count'] == 0
    assert row['completed_work_units'] == row['completed_task_count'] > 0
    assert row['accepted_work_drained'] is True
    assert row['serial_reference']['observed_task_count'] == 1
    assert not any(event == 'mixed_start' for event, _ in case.events)
    assert row['completion_execution_attestation']['completed_work_units'] == row['completed_work_units']
    wrong = deepcopy(row)
    wrong['completed_work_units'] += 1
    with pytest.raises(ValueError):
        generic_only.validate_serial_reference(wrong, case.config, result_root=case.output)
    fps_config = deepcopy(case.config); fps_config.pop('energy_capture'); fps_config.pop('_runtime_energy')
    fps_config['mixed_input_control'] = case.config['energy_capture']['generic_only_control_reuse']['original_mixed_input_control']
    with pytest.raises(ValueError):
        generic_only.validate_serial_reference(row, fps_config, result_root=case.output)


def test_remote_evidence_has_exact_controller_alias_but_no_dut_bypass(tmp_path):
    config, output, _ = prepared_case(tmp_path, 'deepx')
    energy_configuration(config, output)
    capture = config['energy_capture']; aliases = []
    for role in ['performance_result', 'normal_admission_report']:
        local = deepcopy(capture['generic_only_control_reuse'][role])
        remote = str(tmp_path / 'absent_dut' / (role + '.json'))
        capture['generic_only_control_reuse'][role]['path'] = remote
        aliases.append({'role': role, 'local': local, 'runtime_path': remote})
        for item in config['runtime_source_files']:
            if item['path'] == local['path']:
                item['path'] = remote
    config['runtime_source_files'].sort(key=lambda item: item['path'])
    capture['verifier_binding'] = {'evidence': aliases}
    contract = capture['contract']; contract['source_files'] = deepcopy(config['runtime_source_files'])
    contract.pop('command_contract_sha256')
    contract['command_contract_sha256'] = generic_energy.canonical_hash(contract)
    generic_only.validate_generic_only_config(config)
    with pytest.raises((OSError, ValueError)):
        generic_only.validate_generic_only_config(config, check_files=True)
    aliases[0]['local']['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        generic_only.validate_generic_only_config(config)
