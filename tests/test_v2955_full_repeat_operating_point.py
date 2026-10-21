"""Real Full writer seals remain independent while authorized workloads match."""
import copy
import json
from pathlib import Path
import statistics

import pytest

from scripts import native_producer_final_report as reporter
from scripts import native_full_baseline_eval_runner as writer
from test_v2955_full_semantic_artifact_roles import actual_full_case


@pytest.fixture(scope='module')
def full_cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp('full_writer_three')
    rows, contexts = [], []
    for index in (1, 2, 3):
        with pytest.MonkeyPatch.context() as patch:
            case = actual_full_case(root / f'rep{index:02d}_inputs', patch,
                                    frames=1000, warmup=100, numerical_only=False)
        rows.extend(reporter._rows_from_native_full(Path(case.ns.out_dir)))
        contexts.append({'setup_id': case.command_contract['setup_id'],
            'remote_root': case.command_contract['root'], 'remote_tool_dir': str(writer.ROOT)})
    return rows, contexts


def test_three_real_seals_prove_one_operating_point_and_preserve_every_observation(full_cohort):
    rows, contexts = copy.deepcopy(full_cohort)
    seals = [row['full_command_contract_sha256'] for row in rows]
    manifests = [row['full_command_contract']['artifacts']['input_manifest']['sha256'] for row in rows]
    files = {row['operating_point_source_report']: Path(row['operating_point_source_report']).read_bytes() for row in rows}
    assert len(set(seals)) == len(set(manifests)) == 3
    proofs = [reporter._verified_repetition_operating_point(row, remote_execution_contexts=contexts) for row in rows]
    assert all(proofs) and len({proof['operating_point_sha256'] for proof in proofs}) == 1
    result = reporter._aggregate_repetitions(rows, remote_execution_contexts=contexts)[0]
    assert result['ok'] is True and result['repetition_count_valid'] == 3
    assert result['repetition_claim_identity_drift'] is False
    assert result['repetition_operating_point_verified'] is True
    assert result['fps_repetition_samples'] == [row['fps_makespan'] for row in rows]
    assert result['fps_median'] == statistics.median(row['fps_makespan'] for row in rows)
    assert [record['full_command_contract_sha256'] for record in result['repetition_records']] == seals
    assert result['task_complete'] is True
    rate = reporter._final_rate_fields(result, remote_execution_contexts=contexts)
    assert rate['completed_task_fps'] == statistics.median(row['fps_makespan'] for row in rows)
    assert rate['completed_task_work_unit_counts'] == [1000, 1000, 1000]
    assert rate['completed_task_measurement_times_s'] == [record['measured_duration_s'] for record in result['repetition_records']]
    valid, reasons = reporter._validate_repeat_claim_evidence(result, remote_execution_contexts=contexts)
    assert valid, reasons
    assert [row['full_command_contract_sha256'] for row in rows] == seals
    assert {path: Path(path).read_bytes() for path in files} == files
    # Dataset quality is not supplied by this SDK/process fixture.
    assert result.get('quality_evidence_verified') is not True


def test_missing_external_root_authority_keeps_distinct_contracts_uncomparable(full_cohort):
    rows, _ = copy.deepcopy(full_cohort)
    assert all(not reporter._verified_repetition_operating_point(row) for row in rows)
    result = reporter._aggregate_repetitions(rows)[0]
    assert result['ok'] is False and result['repetition_claim_identity_drift'] is True


def changed_report(tmp_path, row, mutate):
    result = copy.deepcopy(row)
    original = Path(result['operating_point_source_report'])
    raw = json.loads(original.read_text())
    contract = raw['rows'][0]['full_command_contract']
    mutate(contract)
    contract.pop('contract_sha256')
    contract['contract_sha256'] = writer._canonical_json_sha256(contract)
    raw['rows'][0]['full_command_contract_sha256'] = contract['contract_sha256']
    result.update(full_command_contract=contract, full_command_contract_sha256=contract['contract_sha256'])
    report = tmp_path / 'outputs/analysis_tables/native_full_baseline_eval.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(raw))
    result['operating_point_source_report'] = str(report)
    result['operating_point_source_report_sha256'] = reporter._hash_file(report)
    return result


@pytest.mark.parametrize('mutation', ['runner', 'model', 'input', 'tensor', 'preprocess', 'mode', 'inflight',
                                    'warmup', 'endpoint', 'unknown', 'other_repeat_tensor'])
def test_real_workload_or_unlisted_provenance_drift_never_coalesces(full_cohort, tmp_path, mutation):
    rows, contexts = copy.deepcopy(full_cohort)
    def mutate(contract):
        if mutation == 'runner': contract['runner_sha256'] = 'f' * 64
        elif mutation == 'model': contract['artifacts']['dxnn']['sha256'] = 'e' * 64
        elif mutation == 'input': contract['input_image_sha256'] = 'd' * 64
        elif mutation == 'tensor': contract['artifacts']['runtime_input_tensor']['sha256'] = 'c' * 64
        elif mutation == 'preprocess': contract['energy_workload']['runtime_numeric_input_sha256'] = 'b' * 64
        elif mutation == 'mode': contract['runtime_options']['deepx_runtime_mode'] = 'S1'
        elif mutation == 'inflight': contract['runtime_options']['deepx_max_inflight'] = 8
        elif mutation == 'warmup': contract['runtime_options']['warmup'] = 0
        elif mutation == 'endpoint': contract['energy_workload']['endpoint_output_names'] = ['other_logits']
        elif mutation == 'unknown': contract['unexpected_source_path'] = '/other/workload'
        else: contract['artifacts']['runtime_input_tensor']['path'] = rows[0]['full_command_contract']['artifacts']['runtime_input_tensor']['path']
    changed = changed_report(tmp_path, rows[1], mutate)
    first = reporter._verified_repetition_operating_point(rows[0], remote_execution_contexts=contexts)
    second = reporter._verified_repetition_operating_point(changed, remote_execution_contexts=contexts)
    assert first
    assert not second or first['operating_point_sha256'] != second['operating_point_sha256']


@pytest.mark.parametrize('change', ['root', 'tool', 'setup', 'duplicate'])
def test_each_full_leaf_requires_one_exact_external_context(full_cohort, change):
    rows, contexts = copy.deepcopy(full_cohort)
    if change == 'root': contexts[1]['remote_root'] += '/foreign'
    elif change == 'tool': contexts[1]['remote_tool_dir'] += '/foreign'
    elif change == 'setup': contexts[1]['setup_id'] = 'other_setup'
    else: contexts.append(copy.deepcopy(contexts[1]))
    assert reporter._verified_repetition_operating_point(rows[1], remote_execution_contexts=contexts) == {}


@pytest.mark.parametrize('artifact', ['input_manifest', 'runtime_input_tensor', 'input_dump'])
def test_actual_payload_hash_tampering_fails_closed(tmp_path, monkeypatch, artifact):
    case = actual_full_case(tmp_path, monkeypatch, frames=1000, warmup=100, numerical_only=False)
    row = reporter._rows_from_native_full(Path(case.ns.out_dir))[0]
    contexts = [{'setup_id': case.setup_id, 'remote_root': case.command_contract['root'], 'remote_tool_dir': str(writer.ROOT)}]
    assert reporter._verified_repetition_operating_point(row, remote_execution_contexts=contexts)
    if artifact == 'input_dump':
        manifest = Path(case.command_contract['artifacts']['input_manifest']['path'])
        payload = json.loads(manifest.read_text())
        target = manifest.parent / payload['input_dump']
    else:
        target = Path(case.command_contract['artifacts'][artifact]['path'])
    target.write_bytes(b'changed payload')
    assert reporter._verified_repetition_operating_point(row, remote_execution_contexts=contexts) == {}


def detection_repeat(path, monkeypatch, model='yolo11l'):
    """Same real Full writers for the existing decoded-pre-NMS endpoint."""
    import sys
    from types import SimpleNamespace
    from test_v27930_native_full_semantic_merge import prepare_runner_case, write_json
    from test_v27930_deepx_semantic_pre_nms import load_semantic_module
    from test_v295_native_deepx_full_a4 import _sdk, _suite
    import numpy as np
    bn6 = model == 'yolo26s'
    outputs = np.array([[[10, 10, 20, 20, .9, 0], [11, 11, 21, 21, .8, 0]]], dtype=np.float32) if bn6 else None
    case = prepare_runner_case(path, monkeypatch, model=model, task='detection', output=outputs)
    if bn6:
        case.contract.update(model_id=model, contract_family='decoded_nms', endpoint_mode='decoded', host_tail_required=False,
            outputs=[{'name': 'model_outputs', 'shape': [1, 2, 6], 'dtype': 'float32'}])
        write_json(case.full / 'output_contract.json', case.contract)
        write_json(case.root / 'output_contracts.json', {'model_id': model, 'task': 'detection', 'contracts': [{
            'model_id': model, 'backend': 'deepx_m1', 'variant': 'full', 'task': 'detection', 'contract_status': 'recorded',
            'endpoint_mode': 'decoded', 'host_tail_required': False, 'postprocessing_required': False,
            'source_coordinate_space': 'model_input_letterbox_xyxy_pixels'}]})
    case.ns.deepx_runtime_mode = 'A4'
    case.ns.frames, case.ns.warmup, case.ns.repetitions = 3, 0, 1
    case.ns.out_dir = str(path / 'separate_report_output')
    case.ns.workspace_mb, case.ns.inflight = 4096, 4
    case.ns.engine_build_python = sys.executable
    case.ns.no_shapes = case.ns.diagnostic_deepx_input_probes = False
    case.ns.deepx_numerical_images, case.ns.deepx_numerical_only = '', False
    _sdk(monkeypatch, case.contract, case.values)
    semantic_module, suite = load_semantic_module(), _suite()
    def process(command, **kwargs):
        if 'native_full_semantic_dump.py' in command[1]:
            monkeypatch.setattr(sys, 'argv', command[1:])
            rc = semantic_module.main()
        else:
            args = SimpleNamespace(runs=3, warmup=0, throughput_frames=3, throughput_warmup_frames=0,
                deepx_runtime_mode='A4', energy_measurement_only=True, prepared_feed_image=str(case.image),
                prepared_input_manifest=command[command.index('--prepared-input-manifest') + 1],
                native_full_postprocess_runtime_mode='auto', quality_evidence_model_id=case.model,
                quality_evidence_setup_id=case.setup_id, deepx_numerical_images='', deepx_numerical_only=False)
            run = {'id': 'deepx_m1_full', 'benchmark_task': case.task, 'backend': 'deepx_m1',
                   'provider': 'deepx_m1', 'variants': ['full'], 'model_id': case.model}
            rows = suite._run_deepx_full_run(case.root, run, args,
                expected_endpoint_identity={'model_id': case.model, 'backend': 'deepx_m1', 'variant': 'full'})
            write_json(case.root / 'benchmark_results_deepx_m1_full_auto.json', rows)
            rc = 0 if rows[0].get('runtime_ok') else 1
        return {'rc': rc, 'timed_out': False, 'stdout': '', 'stderr': '', 'cmd': command}
    monkeypatch.setattr(writer, '_run', process)
    semantic, blocked = writer._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None, blocked
    case.ns.deepx_full_precomputed_semantic = semantic
    measured = writer._row_for_backend(case.root, case.model, 'deepx', case.ns)
    assert measured['ok'] is True, measured
    with monkeypatch.context() as patch:
        patch.setattr(writer, '_deepx_full_series_preflight', lambda *args: (semantic, None))
        patch.setattr(writer, '_row_for_backend', lambda *args: copy.deepcopy(measured))
        patch.setattr(sys, 'argv', [writer.__file__, '--root', str(case.root.parent.parent), '--models', case.model,
            '--backends', 'deepx', '--frames', '3', '--warmup', '0', '--repetitions', '1', '--setup-id', case.setup_id,
            '--comparison-backend', 'deepx', '--engine-build-python', sys.executable, '--deepx-runtime-mode', 'A4',
            '--dump-outputs', '--out-dir', case.ns.out_dir])
        assert writer.main() == 0
    return reporter._rows_from_native_full(Path(case.ns.out_dir))[0]


@pytest.mark.parametrize('model', ['yolo11l', 'yolo26s'])
def test_three_actual_detection_writers_keep_frozen_task_endpoint_stable(tmp_path, monkeypatch, model):
    rows = [detection_repeat(tmp_path / f'rep{index:02d}_inputs', monkeypatch, model) for index in (1, 2, 3)]
    contexts = [{'setup_id': row['setup_id'], 'remote_root': row['full_command_contract']['root'],
                 'remote_tool_dir': str(writer.ROOT)} for row in rows]
    proofs = [reporter._verified_repetition_operating_point(row, remote_execution_contexts=contexts) for row in rows]
    assert all(proofs)
    assert len({proof['operating_point_sha256'] for proof in proofs}) == 1
    result = reporter._aggregate_repetitions(rows, remote_execution_contexts=contexts)[0]
    assert result['ok'] is True and result['repetition_count_valid'] == 3
    assert result['repetition_claim_identity_drift'] is False
    rate = reporter._verified_separate_process_rate_fields(result, remote_execution_contexts=contexts)
    assert rate is not None and rate['completed_task_rate']['repetition_count_valid'] == 3
    assert rate['completed_task_work_unit_counts'] == [3, 3, 3]
    if model == 'yolo26s':
        from onnx_splitpoint_tool import native_detection_postprocess as post
        from scripts.validate_output_dumps import load_dump
        def change_normalization(command):
            workload = command['energy_workload']
            frozen = workload['frozen_decoded_nms_normalization_contract']
            frozen['confidence_threshold'] = .5
            frozen['materialization_contract']['confidence_threshold'] = .5
            material = frozen['materialization_contract']
            material.pop('contract_sha256')
            material['contract_sha256'] = writer._canonical_json_sha256(material)
            frozen.pop('contract_sha256')
            frozen['contract_sha256'] = writer._canonical_json_sha256(frozen)
            workload['frozen_decoded_nms_normalization_contract_sha256'] = frozen['contract_sha256']
        changed = changed_report(tmp_path / 'changed_normalization', rows[1], change_normalization)
        proof = reporter._verified_repetition_operating_point(changed, remote_execution_contexts=contexts)
        assert not proof or proof['operating_point_sha256'] != proofs[0]['operating_point_sha256']
        # Exercise the real collected layout when original remote paths cannot
        # be opened. The collector owns only its copied canonical authority.
        import shutil
        portable = tmp_path / 'portable/outputs'
        report = portable / 'analysis_tables/native_full_baseline_eval.json'
        report.parent.mkdir(parents=True)
        original_report = Path(rows[0]['operating_point_source_report'])
        report.write_bytes(original_report.read_bytes())
        benchmark = Path(rows[0]['full_command_contract']['benchmark_set'])
        local = portable / model / 'benchmark_set'
        shutil.copytree(benchmark / 'native_full_outputs', local / 'native_full_outputs')
        shutil.copyfile(benchmark / 'output_contracts.json', local / 'output_contracts.json')
        collected = copy.deepcopy(rows[0])
        collected['operating_point_source_report'] = str(report)
        original_is_file = Path.is_file
        with monkeypatch.context() as patch:
            roots = [Path(context['remote_root']) for context in contexts]
            patch.setattr(Path, 'is_file', lambda path: False if any(path.is_relative_to(root) for root in roots) else original_is_file(path))
            assert reporter._verified_repetition_operating_point(collected, remote_execution_contexts=contexts)
            (local / 'output_contracts.json').unlink()
            assert reporter._verified_repetition_operating_point(collected, remote_execution_contexts=contexts) == {}
        # Missing or altered authoritative declaration bytes remain unproved.
        declaration = Path(rows[0]['full_command_contract']['benchmark_set']) / 'output_contracts.json'
        original_bytes = declaration.read_bytes()
        try:
            declaration.write_bytes(b'changed original declaration')
            assert reporter._verified_repetition_operating_point(rows[0], remote_execution_contexts=contexts) == {}
        finally:
            declaration.write_bytes(original_bytes)


@pytest.mark.parametrize('change', ['missing', 'invalid', 'S1', 'hailo'])
def test_task_completion_projection_has_no_default_or_other_backend_effect(full_cohort, tmp_path, change):
    rows, _ = full_cohort
    raw = json.loads(Path(rows[0]['operating_point_source_report']).read_text())
    row = raw['rows'][0]
    if change == 'missing': row.pop('task_complete')
    elif change == 'invalid': row['task_complete'] = 1
    elif change == 'S1': row['full_command_contract']['runtime_options']['deepx_runtime_mode'] = 'S1'
    else: row['backend'] = 'native_full_hailo8'
    path = tmp_path / 'analysis_tables/native_full_baseline_eval.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(raw))
    projected = reporter._rows_from_native_full(tmp_path)[0]
    assert 'task_complete' not in projected



def test_explicit_invalid_repeat_seal_is_never_filled_from_parent(full_cohort):
    rows, contexts = copy.deepcopy(full_cohort)
    rows[0]['repetition_records'][0]['workload_contract_sha256'] = False
    result = reporter._aggregate_repetitions(rows, remote_execution_contexts=contexts)[0]
    assert result['repetition_records'][0]['workload_contract_sha256'] is False
    assert reporter._validate_repeat_claim_evidence(result, remote_execution_contexts=contexts)[0] is False
