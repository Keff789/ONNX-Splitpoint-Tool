"""R9J A2 transitions: original evidence is read only; SDK boundaries are fake."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import onnx_splitpoint_tool.native_detection_postprocess as pp
import onnx_splitpoint_tool.native_output_endpoint as ep
from test_v27926_deepx_full_decoded_pre_nms import _suite, _install_runtime

ROOT = Path(__file__).resolve().parents[1]
TASK = Path('/home/kmika/.local/share/onnx-splitpoint-codex/v283_R9J_Fortsetzung_20260920_103442_70_d56qj')
PARENT = TASK.parent / 'v283_R9J_20260919_211316_8daof7vt'
A2 = Path('/home/kmika/Models/EvaluationRuns/v283_R9J_20260919_211316_8daof7vt/a_02__od8ynnc/r9j_a_20260919_225345')
MODEL = Path('/home/kmika/Models/yolo26s.onnx')
A3 = Path('/home/kmika/Models/EvaluationRuns/v283_R9J_Fortsetzung_20260920_103442_70_d56qj/a_03_7__6wky9/r9j_a_20260920_113843')
B1 = Path('/home/kmika/Models/EvaluationRuns/v283_R9J_Fortsetzung_20260920_103442_70_d56qj/b_01_1kvlpxrr/r9j_b_20260920_112416')
DXNN = Path('/home/kmika/Models/BackendArtifacts/deepx/deepx_m1_full_4b8ff9dce56ca705_958dacf708b58fc3/model.dxnn')
B2 = Path('/home/kmika/Models/EvaluationRuns/v283_R9J_Fortsetzung_20260920_103442_70_d56qj/b_02_xch8a_jv/r9j_b_20260920_124238')
A5 = Path('/home/kmika/Models/EvaluationRuns/v283_R9J_Fortsetzung_20260920_103442_70_d56qj/a_05_8pdexx2x/r9j_a_20260920_133559')
ARCHIVE = Path('/home/kmika/.local/share/onnx-splitpoint-codex/v283_r3_lokal_20260914_192352_8vql14m7/archivierte_codex_runs/v283-nightfix-20260914_161840-ZUNmFW/fortsetzung_r2_20260914_173733_k7C9wO')


def read(path):
    return json.loads(Path(path).read_text())


def b2_raw_full(model, backend):
    path = B2 / f'models/{model}/benchmark_results/benchmark_results_{backend}_auto.json'
    rows = read(path)
    assert len(rows) == 1
    return path, rows[0]


@pytest.mark.parametrize('model', ['yolo11l', 'yolo26s'])
@pytest.mark.parametrize('backend', ['hailo8', 'hailo10'])
def test_b2_measured_generic_tail_survives_real_normalization(model, backend):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
    from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
    path, raw = b2_raw_full(model, backend)
    original = read(path.parent / 'normalized_results.json')
    old = next(r for r in original['results'] if r['backend'] == backend)
    assert old['structural_contract_reason'] == 'raw_head_host_tail_missing'
    row = normalize_benchmark_row(raw, model_id=model, source_path=path)
    assert row['host_postprocessing_evidence_status'] == 'passed'
    assert row['structural_contract_pass'] is True
    assert row['measurement_endpoint'] == 'completed_detection'
    assert row['generic_completion_evidence'] == raw['generic_completion_evidence']
    assert row['postprocess_completed_frames'] == 5
    # Continue through the real central-quality gate using the archived result;
    # loss of accuracy remains loss of accuracy, not a missing host tail.
    row['task_quality_gate'] = deepcopy(old['task_quality_gate'])
    apply_accuracy_gate_to_row(row, row['task_quality_policy'])
    assert row['structural_contract_pass'] is True
    assert row['task_quality_status'] == 'accuracy_loss'


@pytest.mark.parametrize('damage', ['threshold', 'contract', 'model', 'backend', 'variant', 'frames', 'outside', 'negative_alias'])
def test_b2_generic_tail_cannot_bypass_contracts(damage):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
    from onnx_splitpoint_tool.validation.host_postprocess import resolve_host_postprocess_evidence
    path, raw = b2_raw_full('yolo11l', 'hailo8')
    row = normalize_benchmark_row(raw, model_id='yolo11l', source_path=path)
    assert resolve_host_postprocess_evidence(row)['available'] is True
    e = row['generic_completion_evidence']
    if damage == 'threshold': e['postprocess_contract']['confidence_threshold'] = .5
    elif damage == 'contract': e['postprocess_contract'] = {}
    elif damage == 'model': row['model_id'] = 'foreign'
    elif damage == 'backend': row['backend'] = 'deepx_m1'
    elif damage == 'variant': row['variant'] = 'split'
    elif damage == 'frames': e['expected_frames'] += 1
    elif damage == 'outside': e['frames'][0]['tail_end_ns'] = e['frames'][0]['timer_end_ns'] + 1
    elif damage == 'negative_alias': row['host_tail_available'] = False
    assert resolve_host_postprocess_evidence(row)['available'] is False


@pytest.mark.parametrize('damage', [
    'host_tail_available', 'host_postprocessing_available', 'postprocess_included',
    'postprocess_completion_verified', 'postprocess_completed_frames',
    'scoped_host_tail', 'tail_binding', 'evidence_alias',
])
def test_b2_raw_completion_conflicts_rejected_before_projection(damage):
    from onnx_splitpoint_tool.workflow.results import expand_normalized_benchmark_rows
    path, raw = b2_raw_full('yolo11l', 'hailo8')
    if damage == 'scoped_host_tail':
        raw['deployment_contracts_by_variant']['full']['host_tail_available'] = False
    elif damage == 'tail_binding':
        raw['deployment_contract']['full_primary_host_tail_sha256'] = '0' * 64
    elif damage == 'evidence_alias':
        raw['generic_full_completion_evidence']['frames'][0]['timer_end_ns'] += 1
    else:
        raw[damage] = 4 if damage == 'postprocess_completed_frames' else False
    # The conflict is in the real incoming report, not injected after the
    # normalizer has already replaced the explicit declaration with success.
    with pytest.raises(ValueError, match='generic_completion_.*conflict'):
        expand_normalized_benchmark_rows(raw, model_id='yolo11l', source_path=path)


@pytest.mark.parametrize('model', ['yolo11l', 'yolo26s'])
def test_b2_real_file_ingestion_scope_and_quality_keep_missing_trt_negative(model):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
    from onnx_splitpoint_tool.workflow.runner import _bind_results_to_required_scope_v2796, required_profile_outcomes_v282
    from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
    folder = B2 / f'models/{model}/benchmark_results'
    old = read(folder / 'normalized_results.json')
    sources = [Path(s['path']) for s in old['sources']]
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    rows, _ = normalize_benchmark_files(model_id=model, source_paths=sources)
    bound, errors = _bind_results_to_required_scope_v2796(old['required_profile_results'], rows)
    assert not errors
    outcome = required_profile_outcomes_v282(old['required_profile_results'], bound, bound, {})
    assert outcome['duplicate_result_count'] == 0
    assert {(r['backend'], r['variant']) for r in outcome['missing_results']} == {('tensorrt', 'full'), ('tensorrt', 'split')}
    full = [r for r in bound if r['variant'] == 'full']
    assert {r['backend'] for r in full} == {'hailo8', 'hailo10'}
    for row in full:
        previous = next(r for r in old['results'] if r['backend'] == row['backend'])
        row['task_quality_gate'] = deepcopy(previous['task_quality_gate'])
        apply_accuracy_gate_to_row(row, row['task_quality_policy'])
        assert row['structural_contract_pass'] is True
        assert row['measurement_endpoint'] == 'completed_detection'
        assert row['task_quality_status'] == 'accuracy_loss'
    assert originals == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def b2_dispatch_case(model):
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
    from onnx_splitpoint_tool.workflow.hardware_matrix import matrix_for_runtime
    profile = load_evaluation_profile(TASK / 'profiles/B.yaml').raw_profile
    plan = read(B2 / f'models/{model}/benchmark_set/legacy_suite/benchmark_plan.json')
    return profile, matrix_for_runtime(profile), plan['runs']


def test_b2_real_generic_timer_decoder_to_report(monkeypatch):
    import ast
    from onnx_splitpoint_tool.runners.task_completion import TimedTaskCompletion
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
    path, raw = b2_raw_full('yolo11l', 'hailo8')
    contract = raw['generic_completion_evidence']['postprocess_contract']
    outputs = {t['name']: np.zeros(t['shape'], np.float32)
               for t in contract['raw_output_tensor_signature']['tensors']}
    processor = pp.FrozenDetectionPostprocessor(contract)
    timer = TimedTaskCompletion('generic_hailo_full', 'detection', lambda: processor.completed_count, contract=contract)
    template = ROOT / 'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'
    tree = ast.parse(template.read_text())
    callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_run_full_frozen_detection_hotloop')
    calls = []
    def infer(inputs):
        calls.append(inputs)
        return outputs
    ns = {'Dict': dict, 'np': np, 'hailo_full': SimpleNamespace(infer=infer),
          'full_inputs_hailo': {'input': np.zeros((1, 640, 640, 3), np.uint8)},
          'generic_full_frozen_postprocessor': processor,
          'generic_full_frozen_original_wh': contract['original_wh'], 'generic_full_completion_timer': timer}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(template), 'exec'), ns)
    for _ in range(6): ns[callback.name]()
    evidence = timer.report(1, 5)
    raw['generic_completion_evidence'] = raw['generic_full_completion_evidence'] = evidence
    row = normalize_benchmark_row(raw, model_id='yolo11l', source_path=path)
    assert len(calls) == processor.completed_count == 6
    assert processor.last_result['detection_count'] == 0
    assert row['structural_contract_pass'] is True
    assert row['host_postprocessing_evidence_status'] == 'passed'
    assert row['completed_task_mean_ms'] >= row['raw_stage_mean_ms'] + row['host_tail_mean_ms']


@pytest.mark.parametrize('model', ['yolo11l', 'yolo26s'])
@pytest.mark.parametrize('reverse', [False, True])
def test_b2_trt_dispatch_reaches_sealed_full_and_split_cases(model, reverse):
    from onnx_splitpoint_tool.workflow.setup_local_trt_dispatch import build_setup_local_tensorrt_quality_dispatch
    profile, targets, rows = b2_dispatch_case(model)
    if reverse: targets.reverse()
    contract = build_setup_local_tensorrt_quality_dispatch(profile, hardware_targets=targets, plan_rows=rows)
    assert contract['ok'], contract['errors']
    trt = next(r for r in rows if r['id'] == 'ort_tensorrt')
    assert contract['performance_owner_setup_id'] == trt['expected_setup_id']
    suite = _suite()
    bench = read(B2 / f'models/{model}/benchmark_set/legacy_suite/benchmark_set.json')
    cases = bench['cases']
    owners = [d for d in contract['setup_dispatches'] if 'ort_tensorrt' not in d['quality_only_run_ids']]
    assert len(owners) == 1
    prepared = suite._assign_full_baseline_owners(rows)
    run = next(r for r in prepared if r['id'] == 'ort_tensorrt')
    selected = suite._cases_for_run(cases, run, setup_id=owners[0]['setup_id'])
    assert [r['case_dir'] for r in selected] == trt['case_ids']
    historical = read(B2 / f'models/{model}/benchmark_results/setup_local_tensorrt_dispatch_preflight.json')
    assert historical['performance_owner_setup_id'] != trt['expected_setup_id']


@pytest.mark.parametrize('model', ['yolo11l', 'yolo26s'])
def test_b2_dispatch_owner_survives_real_execution_binding(tmp_path, monkeypatch, model):
    import shlex
    import onnx_splitpoint_tool.workflow.execution_binding as binding
    profile, targets, runs = b2_dispatch_case(model)
    source = B2 / f'models/{model}/benchmark_set/legacy_suite'
    suite = tmp_path / 'suite'
    write(suite / 'benchmark_plan.json', read(source / 'benchmark_plan.json'))
    write(suite / 'benchmark_set.json', read(source / 'benchmark_set.json'))
    result_dir = tmp_path / 'results'
    result_dir.mkdir()
    calls = []
    def remote_boundary(**kwargs):
        calls.append(kwargs)
        return binding.ExecutionBindingResult(artifacts={}, metrics={}, status='ok', message='controlled remote boundary')
    # Only stop the hardware operation; use the real scheduler, argument
    # construction, plan and generated suite's case selector.
    monkeypatch.setattr(binding, '_run_remote_dispatch_once', remote_boundary)
    result = binding._remote_execution_if_requested(
        run_root=tmp_path, model_id=model,
        options=SimpleNamespace(no_remote=False, benchmark_execution_backend='remote', parallel_remote_setups=False),
        profile_payload=profile, suite_dir=suite, benchmark_set_json=suite / 'benchmark_set.json',
        result_dir=result_dir, contains_hailo=True, gates={}, log=None)
    # No remote result bytes were created: the real consumer must retain an
    # incomplete execution outcome even though dispatch selection is correct.
    assert result.status == 'partial', result
    assert len(calls) == len(targets) == 2
    trt = next(r for r in _suite()._assign_full_baseline_owners(runs) if r['id'] == 'ort_tensorrt')
    owners = []
    for call in calls:
        args = shlex.split(call['runtime_override']['add_args'])
        assert 'ort_tensorrt' in args[args.index('--run-ids') + 1].split(',')
        if '--quality-only-run-ids' in args:
            assert args[args.index('--quality-only-run-ids') + 1] == 'ort_tensorrt'
            continue
        owners.append(call['target_id'])
        cases = _suite()._cases_for_run(read(source / 'benchmark_set.json')['cases'], trt, setup_id=call['target_id'])
        assert [r['case_dir'] for r in cases] == trt['case_ids']
    assert owners == [trt['expected_setup_id']]


@pytest.mark.parametrize('model', ['resnet50', 'regnet_x_1_6gf', 'yolo26s'])
def test_a5_archived_full_reports_survive_file_ingestion(model):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
    from onnx_splitpoint_tool.workflow.runner import _bind_results_to_required_scope_v2796, required_profile_outcomes_v282
    old = read(A5 / f'models/{model}/benchmark_results/normalized_results.json')
    rows, _ = normalize_benchmark_files(model_id=model, source_paths=[s['path'] for s in old['sources']])
    bound, errors = _bind_results_to_required_scope_v2796(old['required_profile_results'], rows)
    assert not errors
    outcome = required_profile_outcomes_v282(old['required_profile_results'], bound, bound, {})
    assert outcome['missing_result_count'] == outcome['duplicate_result_count'] == 0
    full = [r for r in bound if r['variant'] == 'full']
    assert {r['backend'] for r in full} == {'deepx_m1', 'tensorrt'}
    assert all(r['structural_contract_pass'] is True for r in full)


@pytest.mark.parametrize('damage', ['conflict', 'foreign', 'selection_conflict'])
def test_b2_trt_dispatch_rejects_inconsistent_physical_owner(damage):
    from onnx_splitpoint_tool.workflow.setup_local_trt_dispatch import build_setup_local_tensorrt_quality_dispatch
    profile, targets, rows = b2_dispatch_case('yolo11l')
    trt = next(r for r in rows if r['id'] == 'ort_tensorrt')
    if damage == 'conflict': trt['setup_id'] = 'orin_nx_hailo10_01'
    elif damage == 'foreign':
        trt['setup_id'] = trt['expected_setup_id'] = 'foreign'
        trt['backend_selection_contracts'][0]['setup_id'] = 'foreign'
    else: trt['backend_selection_contracts'][0]['setup_id'] = 'orin_nx_hailo10_01'
    result = build_setup_local_tensorrt_quality_dispatch(profile, hardware_targets=targets, plan_rows=rows)
    assert result['ok'] is False
    assert any('performance_owner' in e for e in result['errors'])


def stored_endpoint_case(run, model, backend='cuda_ort'):
    root = run / f'models/{model}/benchmark_set/legacy_suite'
    task = 'classification' if model in ('resnet50', 'regnet_x_1_6gf') else 'detection'
    declaration = ep.load_authoritative_output_contract(root, backend=backend, model_id=model, task=task)
    assert declaration['contract_resolution_status'] == 'attested'
    # Controlled runtime tensors at the inference boundary; the real declaration,
    # attestor, exporter and request reader are exercised below.
    stage = declaration['stage']
    if task == 'classification':
        outputs = {'output': np.arange(1000, dtype=np.float32)[None, :]}
    elif stage == 'decoded_pre_nms':
        outputs = {'output0': np.zeros((1, 84, 8400), np.float32)}
    else:
        outputs = {'output0': saved('000000052891')}
    return root, task, declaration, outputs


@pytest.mark.parametrize('run,model', [(A3, 'resnet50'), (A3, 'regnet_x_1_6gf'), (A3, 'yolo26s'), (B1, 'yolo11l'), (B1, 'yolo26s')])
def test_current_host_endpoint_transition_to_real_quality_export(tmp_path, run, model):
    import onnx
    from test_v275_preprocessing_contract import _runner_module
    runner = _runner_module()
    root, task, declaration, outputs = stored_endpoint_case(run, model)
    fmt = {'classification_logits': 'classification_logits', 'decoded_nms': 'bn6_detections',
           'decoded_pre_nms': 'ultralytics_decoded'}[declaration['stage']]
    diagnostics = {}
    source = root / f'models/{model}.onnx'
    graph = onnx.load(source, load_external_data=False).graph
    outputs = {graph.output[0].name: next(iter(outputs.values()))}
    suite = tmp_path / 'suite'
    case = suite / 'case'
    write(suite / 'benchmark_set.json', read(root / 'benchmark_set.json'))
    write(suite / 'output_contracts.json', read(root / 'output_contracts.json'))
    write(case / 'split_manifest.json', {'full_model': str(source)})
    declaration = runner._recorded_suite_endpoint_declaration(base_dir=case,
        full_model=source, terminal_model=source, variant='full', provider='tensorrt',
        task=task, output_names=list(outputs), outputs=list(outputs.values()), diagnostics=diagnostics)
    assert declaration, diagnostics
    endpoint = runner._central_quality_endpoint_contract(task=task,
        output_names=list(outputs), outputs=list(outputs.values()),
        detected_output_format=fmt, declared_endpoint_contract=declaration, diagnostics=diagnostics)
    assert endpoint.get('endpoint_contract_complete') is True, diagnostics
    assert pp.canonical_json_sha256(endpoint['producer_endpoint_identity']) == endpoint['endpoint_contract_hash']
    assert endpoint['producer_endpoint_identity']['semantic']['source_onnx_detection_endpoint'] == declaration['source_onnx_detection_endpoint']
    policy = {'statistics': {'execution_location': 'central_management'}}
    records = {'classification_rows': [{'image': 'renamed.jpg', 'label_id': 1,
        'gt': {'top1_hit': True, 'top5_hit': True}, 'gt_reference': {'top1_hit': True, 'top5_hit': True}}]} if task == 'classification' else {
        'gt_by_image': {'renamed.jpg': []}, 'candidate_by_image': {'renamed.jpg': []}, 'reference_by_image': {'renamed.jpg': []}}
    exported = runner._export_central_quality_inputs(out_dir=tmp_path, task=task,
        variant='full', policy=policy, endpoint_contract=endpoint,
        runtime_precision_identity='fp16', **records)
    request_path = Path(exported['request']['path'])
    request = read(request_path)
    assert request['endpoint_contract_hash'] == endpoint['endpoint_contract_hash']
    candidate = read(request_path.parent / request['candidate']['path'])
    assert len(candidate['records']) == 1
    # Original remote failures are evidence, never edited into successful runs.
    logs = list((root.parents[1] / 'benchmark_results/remote_diagnostics').glob('*/logs/runner.log'))
    assert logs and all('runtime_endpoint_identity_hash_mismatch' in p.read_text() for p in logs)


@pytest.mark.parametrize('damage', ['hash', 'signature', 'stage', 'format', 'graph', 'declaration', 'unavailable'])
def test_central_endpoint_reconstruction_still_rejects_drift(monkeypatch, damage):
    from test_v275_preprocessing_contract import _runner_module
    runner = _runner_module()
    _, task, declaration, outputs = stored_endpoint_case(A3, 'yolo26s')
    original = runner.runtime_output_contract
    def corrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result['endpoint_contract_complete'] is True
        if damage == 'hash':
            result['endpoint_contract_hash'] = 'f' * 64
        elif damage == 'signature':
            result['tensor_signature']['tensors'][0]['shape'][1] += 1
        elif damage == 'stage':
            result['stage'] = 'raw_head'
        elif damage == 'format':
            result['output_format'] = 'raw_detection_tensors'
        return result
    monkeypatch.setattr(runner, 'runtime_output_contract', None if damage == 'unavailable' else corrupted if damage not in ('graph', 'declaration') else original)
    if damage == 'graph':
        declaration['source_onnx_detection_endpoint']['candidate_selection']['source_onnx_sha256'] = '0' * 64
    elif damage == 'declaration':
        declaration['source_contracts_sha256'] = '0' * 64
    diagnostics = {}
    result = runner._central_quality_endpoint_contract(task=task, output_names=list(outputs),
        outputs=list(outputs.values()), detected_output_format='bn6_detections',
        declared_endpoint_contract=declaration, diagnostics=diagnostics)
    assert not result.get('endpoint_contract_complete'), diagnostics
    assert diagnostics['runtime_endpoint']['status'] == 'failed'


def test_unextended_v3_endpoint_identity_remains_unchanged(tmp_path):
    stored = read(A3 / 'models/resnet50/benchmark_set/legacy_suite/output_contracts.json')
    for row in stored['contracts']:
        row.pop('source_onnx_detection_endpoint', None)
    write(tmp_path / 'output_contracts.json', stored)
    declaration = ep.load_authoritative_output_contract(tmp_path, backend='cuda_ort',
        model_id='resnet50', task='classification')
    outputs = {'renamed': np.array([[1, 2, 3]], np.float32)}
    result = ep.runtime_output_contract('classification', outputs, raw_fallback=False, declared_contract=declaration)
    legacy = {'schema': 'onnx-splitpoint/output-endpoint-contract', 'schema_version': 3,
        'task': 'classification', 'stage': 'classification_logits', 'output_format': 'classification_logits',
        'tensor_signature': {'tensor_count': 1, 'tensors': [{'index': 0, 'rank': 1, 'shape': [3]}]}, 'semantic': {}}
    assert result['endpoint_contract_hash'] == pp.canonical_json_sha256(legacy)


@pytest.mark.parametrize('model', ['resnet50', 'regnet_x_1_6gf', 'yolo26s'])
def test_current_deepx_quality_identity_matches_measured_pre_adapter_endpoint(model):
    root, task, declaration, outputs = stored_endpoint_case(A3, model, 'deepx_m1')
    artifact = read(root / 'deepx/deepx_m1/full/output_contract.json')
    measured = ep.runtime_output_contract(task, outputs, raw_fallback=False, declared_contract=declaration)
    identity = _suite()._deepx_standard_runtime_endpoint_identity(task=task,
        runtime_observation={'outputs': [{'shape': list(v.shape)} for v in outputs.values()]},
        output_contract=artifact,
        source_has_integrated_nms=artifact['endpoint_semantic_attestation']['source_endpoint_has_integrated_nms'] if task == 'detection' else None,
        classification_observed_stage=declaration['stage'], expected_model_id=model,
        expected_backend='deepx_m1', expected_variant='full')
    assert identity['stage'] == measured['stage']
    assert pp.canonical_json_sha256(identity) == measured['endpoint_contract_hash']


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def saved(image):
    with np.load(ARCHIVE / 'deepx_remote/results' / f'{image}_raw.npz') as data:
        return data[data.files[0]].copy()


def candidate_suite(root, *, model='renamed_detector', proof=None):
    """Regenerate the existing declarations, bound to the same ONNX/DXNN bytes."""
    source = A2 / 'models/yolo26s/benchmark_set/legacy_suite'
    declared = read(source / 'output_contracts.json')
    declared['model_id'] = model
    proof = ep.inspect_bn6_candidate_graph(MODEL) if proof is None else proof
    for row in declared['contracts']:
        row['model_id'] = model
        row['source_onnx_detection_endpoint']['candidate_selection'] = proof
    write(root / 'output_contracts.json', declared)
    artifact = read(source / 'deepx/deepx_m1/full/output_contract.json')
    artifact['model_id'] = model
    artifact['endpoint_semantic_attestation'] = _suite()._deepx_authoritative_endpoint_attestation(
        root, {'model_id': model, 'benchmark_task': 'detection'})
    write(root / 'deepx/deepx_m1/full/output_contract.json', artifact)
    return artifact


def processor(root, array, *, threshold=.25, original_wh=(640, 480)):
    declaration = ep.load_authoritative_output_contract(root, backend='deepx_m1', model_id='renamed_detector', task='detection')
    outputs = {'model_outputs': array}
    source = ep.runtime_output_contract('detection', outputs, raw_fallback=True, declared_contract=declaration)
    contract = pp.build_frozen_decoded_nms_normalization_contract(
        source_completed=True, model_id='renamed_detector', outputs=outputs,
        input_hw=[640, 640], original_wh=original_wh,
        preprocess={'mode': 'letterbox', 'letterbox_pad_value': 114, 'color_space': 'RGB'},
        source_coordinate_space='model_input_letterbox_xyxy_pixels',
        source_endpoint_contract_hash=source['endpoint_contract_hash'],
        source_output_endpoint_attestation=source['output_endpoint_attestation'],
        confidence_threshold=threshold)
    return pp.FrozenDecodedNmsPostprocessor(contract), outputs, source, declaration


@pytest.mark.parametrize('image,bad_count', [('000000052891', 5), ('000000395801', 1), ('000000000139', 0), ('000000000285', 0)])
def test_original_outputs_transition_from_strict_final_to_graph_bound_candidates(tmp_path, monkeypatch, image, bad_count):
    array = saved(image)
    before = array.copy()
    _install_runtime(monkeypatch, array)
    suite = _suite()
    old_root = ARCHIVE / 'tmp/deepx_stage/suite'
    old = suite._deepx_bind_authoritative_endpoint_contract(old_root, {'model_id': 'yolo26s', 'benchmark_task': 'detection'}, read(old_root / 'deepx/deepx_m1/full/output_contract.json'))
    assert suite._deepx_bn6_runtime_semantic_attestation(array, old)['pass'] is (bad_count == 0)
    artifact = candidate_suite(tmp_path)
    bound = suite._deepx_bind_authoritative_endpoint_contract(tmp_path, {'model_id': 'renamed_detector', 'benchmark_task': 'detection'}, artifact)
    attestation = suite._deepx_bn6_runtime_semantic_attestation(array, bound)
    assert attestation['pass'], attestation
    assert attestation['raw_invalid_geometry_count'] == bad_count
    runtime, outputs, _, _ = processor(tmp_path, array)
    result = runtime.process(outputs, original_wh=[640, 480])
    assert result['host_nms_applied'] is False
    observation = runtime._materializer.last_result['candidate_selection_observation']
    assert observation['raw_invalid_geometry_count'] == bad_count
    assert observation['retained_record_count'] == int(np.sum(array[..., 4] >= .25))
    assert observation['retained_invalid_geometry_count'] == 0
    np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize('damage', ['retained_box', 'nonfinite_low', 'nonfinite_high', 'class', 'score', 'threshold', 'graph_shape', 'graph_artifact'])
def test_candidate_contract_negatives(tmp_path, monkeypatch, damage):
    array = saved('000000052891')
    artifact = candidate_suite(tmp_path)
    _install_runtime(monkeypatch, array)
    if damage == 'retained_box':
        array[0, 13, 4] = .9
    elif damage.startswith('nonfinite'):
        array[0, 13, 0] = np.nan
        array[0, 13, 4] = .9 if damage.endswith('high') else .00001
    elif damage == 'class':
        array[0, 13, 5] = 80
    elif damage == 'score':
        array[0, 13, 4] = 1.1
    elif damage == 'graph_shape':
        array = array[:, :50]
    elif damage == 'graph_artifact':
        artifact['build_onnx_sha256'] = '0' * 64
        bound = _suite()._deepx_bind_authoritative_endpoint_contract(tmp_path, {'model_id': 'renamed_detector', 'benchmark_task': 'detection'}, artifact)
        assert bound['endpoint_contract_binding_status'] == 'conflict'
        return
    with pytest.raises((RuntimeError, KeyError)):
        runtime, outputs, _, _ = processor(tmp_path, array, threshold=.001 if damage == 'threshold' else .25)
        runtime.process(outputs, original_wh=[640, 480])


def test_empty_evaluated_image_and_overlaps_are_preserved(tmp_path):
    candidate_suite(tmp_path)
    array = np.zeros((1, 300, 6), np.float32)
    array[0, 0] = [10, 0, 5, 10, .01, 1]
    runtime, outputs, _, _ = processor(tmp_path, array)
    assert runtime.process(outputs, original_wh=[640, 480])['detection_count'] == 0
    array[0, :2] = [[10, 150, 110, 250, .9, 0], [11, 151, 111, 251, .8, 0]]
    assert runtime.process(outputs, original_wh=[640, 480])['detection_count'] == 2
    assert runtime.completed_count == 2


def test_graph_inspection_uses_connected_operations_not_names_or_shape(tmp_path):
    import onnx
    model = onnx.load(MODEL, load_external_data=False)
    proof = ep.inspect_bn6_candidate_graph(MODEL)
    assert proof['source_onnx_sha256'] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert proof['candidate_count'] == 300 and proof['class_count'] == 80
    for i, node in enumerate(model.graph.node):
        node.name = f'anonymous_{i}'
    path = tmp_path / 'unrelated.onnx'
    onnx.save(model, path)
    assert ep.inspect_bn6_candidate_graph(path)['candidate_count'] == 300
    split = next(n for n in reversed(model.graph.node) if n.op_type == 'Split')
    axis = next(a for a in split.attribute if a.name == 'axis')
    axis.i = 1
    onnx.save(model, path)
    assert ep.inspect_bn6_candidate_graph(path) == {}
    axis.i = -1
    model.graph.node[-1].op_type = 'Identity'
    onnx.save(model, path)
    assert ep.inspect_bn6_candidate_graph(path) == {}


def test_bound_graph_proof_cannot_be_replaced_in_materialization(tmp_path):
    candidate_suite(tmp_path)
    runtime, outputs, source, _ = processor(tmp_path, saved('000000052891'))
    contract = deepcopy(runtime.contract['materialization_contract'])
    contract['candidate_selection']['class_count'] = 99
    contract.pop('contract_sha256')
    contract['contract_sha256'] = pp.canonical_json_sha256(contract)
    with pytest.raises(RuntimeError, match='candidate_graph_mismatch'):
        pp.verify_attested_decoded_nms_materialization_contract(contract, outputs=outputs,
            source_output_endpoint_attestation=source['output_endpoint_attestation'])


def test_actual_dxnn_exposes_cpu_topk_candidates_without_nms_or_confidence_filter():
    """Read the existing artifact; never create an SDK engine or recompile it."""
    import onnx
    raw = DXNN.read_bytes()
    header = json.JSONDecoder().raw_decode(raw[raw.index(b'{'):8192].decode().rstrip('\0'))[0]
    assert header['signature'] == 'DXNN'
    def section(entry):
        start = header['size'] + entry['offset']
        return raw[start:start + entry['size']]
    config = json.loads(section(header['data']['compile_config']))
    assert config['compile_version'] == '2.3.0-rc.5'
    graph_info = json.loads(section(header['data']['graph_info']))
    assert graph_info['outputs'] == ['output0'] and graph_info['offloading'] is False
    assert graph_info['toposort_order'] == ['npu_0', 'cpu_0']
    cpu = onnx.load_model_from_string(section(header['data']['cpu_models']['cpu_0']))
    source = onnx.load(MODEL, load_external_data=False)
    producers = {v: n for n in cpu.graph.node for v in n.output}
    final = producers['output0']
    assert final.op_type == 'Concat' and len(final.input) == 3
    assert producers[final.input[0]].op_type == 'GatherElements'
    score_shape = producers[final.input[1]]
    assert score_shape.op_type == 'Reshape'
    assert producers[score_shape.input[0]].op_type == 'TopK'
    assert not {'NonMaxSuppression', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual', 'Where', 'NonZero'} & {n.op_type for n in cpu.graph.node}
    source_info = {m.key: m.value for m in source.metadata_props}
    assert source_info['version'] == '8.4.51'
    assert source_info['end2end'] == 'True'
    assert ep.inspect_bn6_candidate_graph(MODEL)['output_shape'] == [1, 300, 6]


def test_actual_a2_full_matrix_replay_and_negative_bindings():
    from onnx_splitpoint_tool.workflow.runner import _bind_results_to_required_scope_v2796, required_profile_outcomes_v282
    full_count = 0
    for model in ('resnet50', 'regnet_x_1_6gf', 'yolo26s'):
        payload = read(A2 / f'models/{model}/benchmark_results/normalized_results.json')
        expected, rows = payload['required_profile_results'], payload['results']
        assert payload['missing_required_profile_result_count'] > 0
        bound, errors = _bind_results_to_required_scope_v2796(expected, rows)
        assert not errors, errors
        assert [r['measurement_endpoint'] for r in bound] == [r['measurement_endpoint'] for r in rows]
        outcome = required_profile_outcomes_v282(expected, bound, bound, {})
        assert outcome['missing_result_count'] == outcome['duplicate_result_count'] == 0
        full_count += sum(r['variant'] == 'full' for r in bound)
        original = next(r for r in rows if r['variant'] == 'full')
        for key, value in [('model_id', 'foreign'), ('setup_id', 'foreign'), ('backend', 'native_full_deepx'), ('variant', 'split'), ('measurement_endpoint', 'p2_output'), ('generic_completion_evidence', {})]:
            damaged = deepcopy(original)
            damaged[key] = value
            if key == 'variant':
                damaged['quality_source_variant'] = 'split'
            _, errors = _bind_results_to_required_scope_v2796(expected, [damaged])
            assert errors, (model, key)
        assert required_profile_outcomes_v282(expected, bound + [bound[0]], bound, {})['duplicate_result_count'] == 1
        assert required_profile_outcomes_v282(expected, [r for r in bound if r['variant'] != 'full'], bound, {})['missing_result_count'] == 2
    assert full_count == 6


def test_actual_a2_uncertainty_is_overlapping_axis(tmp_path):
    from onnx_splitpoint_tool.quality_lifecycle import summarize_requests, progress_text
    from onnx_splitpoint_tool.workflow.scientific_reporting import project_central_quality_status, _write_reports
    from onnx_splitpoint_tool.workflow.evidence_status import workflow_completion_projection
    source = read(A2 / 'quality_management/central_quality_summary.json')
    counts = summarize_requests(source['results'])
    assert counts['completed_count'] == 14
    assert counts['quality_decision_counts']['reference_close'] == 13
    assert counts['quality_decision_counts']['accuracy_loss'] == 1
    assert counts['quality_uncertainty_counts']['inconclusive'] == 2
    assert '2 statistisch unsicher' in progress_text(counts)
    projection = project_central_quality_status(source)
    completion = workflow_completion_projection('ok', central_quality=projection)
    assert completion['counts']['central_quality_uncertainty_inconclusive'] == 2
    assert completion['counts']['central_quality_completed'] == 14
    assert completion['severity'] == 'warning'
    _write_reports(tmp_path / 'reports/scientific', {'run_id': 'replay', 'rows': [], 'central_quality_results': projection['results'], 'central_quality_reporting': projection})
    csvs = list(tmp_path.rglob('*task_quality*.csv'))
    assert csvs and any('accuracy_uncertainty' in p.read_text() and 'inconclusive' in p.read_text() for p in csvs)


def guard():
    # These operator scripts use bare imports. Do not leak this task's scope
    # module into R9I/R9J historical guards run later by the same pytest process.
    previous_path = list(sys.path)
    previous = {name: sys.modules.get(name) for name in ('scope_contract', 'quality_reference')}
    try:
        for name in previous:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(TASK / 'operator'))
        spec = importlib.util.spec_from_file_location('r9j_continuation_guard', TASK / 'operator/strict_postcheck.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_scope = sys.modules['scope_contract'].validate_scope
        return module
    finally:
        sys.path[:] = previous_path
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_actual_a2_sentinel_union_is_not_deepx_full():
    from onnx_splitpoint_tool.accuracy_reporting import observed_coverage
    rows = read(A2 / 'reports/scientific/central_quality_results.json')
    coverage = observed_coverage(rows, run_id=A2.name, run_root=A2)
    assert {'52891', '395801'} <= set(coverage['observed_image_ids'])
    assert not any(c['backend'] == 'deepx_m1' and c['variant'] == 'full' for c in coverage['consumers'])
    errors = guard().continuation_checks(A2, 'A', rows, {'central_quality_uncertainty_inconclusive': 2})
    assert 'sentinel_deepx_full_quality_absent' in errors


def test_current_profiles_effective_scope(monkeypatch):
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
    oracle = guard()
    scope = oracle.CELLS
    for cell in ('A', 'B', 'EVAL'):
        monkeypatch.setenv('R9J_CELL', cell)
        profile = load_evaluation_profile(TASK / f'profiles/{cell}.yaml').raw_profile
        oracle.validate_scope(profile, 'eval' if cell == 'EVAL' else 'smoke')
        assert profile['native_producers']['energy']['enabled'] == scope[cell]['energy']


def test_a2_v1_v2_mismatch_then_real_native_consumer_chain(tmp_path, monkeypatch):
    from test_v27930_native_full_semantic_merge import prepare_runner_case, attach, owned, runner
    from test_v27930_deepx_semantic_pre_nms import _set_endpoint
    base = A2 / 'native_producers/deepx/yolo26s/benchmark_set'
    historical = read(base / 'results/deepx_m1_full/deepx_prepared_feed_benchmark.json')
    semantic_path = next(base.rglob('native_full_semantic_dump.json'))
    historical_dump = read(semantic_path)
    assert historical['frozen_decoded_nms_normalization_contract']['schema_version'] == 2
    assert historical_dump['frozen_decoded_nms_normalization_contract']['schema_version'] == 1
    assert historical['frozen_decoded_nms_normalization_contract_sha256'] != historical_dump['frozen_decoded_nms_normalization_contract_sha256']
    # The actual merge gate still rejects that old disagreement.
    old_row = {'ok': True, 'deepx_prepared_feed_benchmark': historical,
               'normalization_frozen': True,
               'frozen_decoded_nms_normalization_contract_sha256': historical['frozen_decoded_nms_normalization_contract_sha256']}
    old_result = runner._attach_semantic_dump(old_row, base, 'yolo26s', 'native_full_deepx', 'deepx_m1_full', SimpleNamespace(dump_outputs=True), precomputed_result=historical_dump)
    assert old_result['ok'] is False
    assert 'normalization_contract_mismatch' in str(old_result)
    outputs = np.array([[[10, 150, 120, 250, .9, 0], [11, 151, 121, 251, .8, 0]]], np.float32)
    case = prepare_runner_case(tmp_path, monkeypatch, model='yolo26s', task='detection', output=outputs)
    _set_endpoint(case, 'decoded_nms')
    declaration = read(case.root / 'output_contracts.json')
    declaration['contracts'][0]['endpoint_mode'] = 'decoded'
    write(case.root / 'output_contracts.json', declaration)
    dump, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None, blocked
    perf = runner._generic_full_via_suite(case.root, case.model, 'native_full_deepx', 'deepx_m1_full', case.ns, prepared_input_manifest=Path(dump['input_manifest']))
    perf['runtime_success'] = True
    before = owned(perf)
    merged = attach(case, perf, dump)
    assert merged['ok'] is True, merged
    assert owned(merged) == before
    assert merged['frozen_decoded_nms_normalization_contract_sha256'] == dump['frozen_decoded_nms_normalization_contract_sha256']
    assert merged['frozen_decoded_nms_normalization_contract']['schema_version'] == 2
    assert merged['frozen_decoded_nms_normalization_result']['detection_count'] == 2
    assert merged['postprocess_completed_frames'] == 3
    assert merged['request_latency']['task_complete'] is True
    for field, value in [('confidence_threshold', .001), ('iou_threshold', .7), ('original_wh', [640, 640]), ('nms_implementation', 'other')]:
        contract = deepcopy(dump['frozen_decoded_nms_normalization_contract'])
        contract[field] = value
        contract['contract_sha256'] = pp.canonical_json_sha256({k: v for k, v in contract.items() if k != 'contract_sha256'})
        with pytest.raises(pp.FrozenPostprocessError):
            pp.verify_frozen_decoded_nms_normalization_contract(contract)
    changed = deepcopy(dump)
    changed['frozen_decoded_nms_normalization_contract_sha256'] = historical_dump['frozen_decoded_nms_normalization_contract_sha256']
    assert attach(case, deepcopy(perf), changed)['ok'] is False


def replay32(tmp_path, monkeypatch):
    """Real image/GT/export path; four saved SDK outputs, no inference."""
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
    import onnx_splitpoint_tool.runners.task_completion as completion
    suite = _suite()
    root = tmp_path / 'replay_run'
    artifact = candidate_suite(root, model='yolo26s')
    (root / 'models').mkdir()
    (root / 'models/source.onnx').symlink_to(MODEL)
    profile = load_evaluation_profile(TASK / 'profiles/A.yaml').raw_profile
    manifest = A2 / 'models/yolo26s/benchmark_set/legacy_suite/resources/validation/detection/images_n32_s20260710'
    run = {'id': 'deepx_m1_full', 'model_id': 'yolo26s', 'backend': 'deepx_m1', 'variant': 'full', 'benchmark_task': 'detection',
           'validation_images': str(manifest), 'validation_max_images': 32,
           'contract_path': 'deepx/deepx_m1/full/output_contract.json', 'task_quality_gate': profile['quality_gate']}
    samples = suite._deepx_collect_validation_samples(str(manifest), root, 32)
    raw = saved('000000052891')
    _install_runtime(monkeypatch, raw)
    monkeypatch.setitem(sys.modules, 'splitpoint_runners.task_completion', completion)
    image = next(Path(x['path']) for x in samples if Path(x['path']).stem == '000000052891')
    monkeypatch.setattr(suite, '_deepx_find_prepared_feed_image', lambda *a: (image, 'saved_output_replay'))
    dxnn = Path(artifact['artifact_path'])
    perf_dir = root / 'results/performance'
    perf = suite._run_deepx_prepared_feed_benchmark(root, dxnn, run, SimpleNamespace(warmup=1, runs=2, benchmark_task='detection'), perf_dir)
    assert perf['status'] == 'ok', perf
    energy = suite._run_deepx_prepared_feed_benchmark(root, dxnn, run,
        SimpleNamespace(warmup=0, runs=3, benchmark_task='detection', energy_measurement_only=True,
                        throughput_frames=3, throughput_warmup_frames=0), root / 'results/energy_work_replay')
    assert energy['status'] == 'ok', energy
    assert energy['completed_frames'] == energy['postprocess_completed_frames'] == 3
    assert energy['frozen_decoded_nms_normalization_contract'] == perf['frozen_decoded_nms_normalization_contract']
    observed = []
    class SavedSDK:
        def __init__(self, path):
            assert Path(path) == dxnn
        def run(self, feeds):
            assert feeds[0].shape == (640, 640, 3)
            name = Path(samples[len(observed)]['path']).stem
            observed.append(name)
            control = '000000000139' if len(observed) % 2 else '000000000285'
            return [saved(name if name in {'000000052891', '000000395801'} else control)]
    monkeypatch.setattr(sys.modules['dx_engine'], 'InferenceEngine', SavedSDK)
    result_dir = root / 'results/semantic'
    semantic = suite._run_deepx_semantic_validation(root, dxnn, run, SimpleNamespace(), result_dir)
    assert semantic['status'] == 'ok', semantic.get('errors')
    assert semantic['image_count'] == semantic['validated_image_count'] == 32
    assert len(set(observed)) == 32 and {'000000052891', '000000395801'} <= set(observed)
    exported = suite._deepx_export_central_quality_request(root, dxnn, run, semantic, result_dir, completed_task_evidence=perf)
    request = read(result_dir / 'task_quality_inputs/full_request.json')
    endpoint = request['producer_identity']['endpoint']
    assert endpoint['identity']['stage'] == 'decoded_nms'
    assert endpoint['sha256'] == perf['source_endpoint_contract_hash'] == energy['source_endpoint_contract_hash']
    return root, exported, semantic


def test_real_32_image_export_preserves_ground_truth_and_candidate_provenance(tmp_path, monkeypatch):
    root, exported, semantic = replay32(tmp_path, monkeypatch)
    requests = list(root.rglob('full_request.json'))
    assert len(requests) == 1, exported
    request = read(requests[0])
    candidate_path = requests[0].parent / request['candidate']['path']
    candidate = read(candidate_path)
    assert request['record_count'] == len(candidate['records']) == 32
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == request['candidate']['sha256']
    assert len({r['image_id'] for r in candidate['records']}) == 32
    assert all('ground_truth' in r for r in candidate['records'])
    from onnx_splitpoint_tool.quality_service import quality_request_from_manifest, prepare_evaluation, _evaluate_payload
    historical = next(r for r in read(A2 / 'quality_management/central_quality_summary.json')['results'] if r.get('task') == 'detection')
    reference_path = next((A2 / 'quality_management/references/yolo26s').rglob('canonical_cpu_reference.json'))
    reference_before = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    loaded = quality_request_from_manifest(requests[0], reference_artifact=reference_path)
    from dataclasses import replace
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    manager = object.__new__(EvaluationWorkflowRunner)
    manager.run_dir = root
    manager._central_quality_reference_store = None
    status = read(A2 / 'quality_management/references/yolo26s/management_cpu_reference_status.json')
    identity, stored = manager._central_reference_contract('yolo26s', read(reference_path)['records'], status)
    loaded = replace(loaded, reference_identity=identity.fingerprint())
    _, payload = prepare_evaluation(loaded)
    evaluated = _evaluate_payload(payload)
    assert evaluated['evaluated_images'] == 32
    assert evaluated['primary']['metric'] == 'coco_ap_50_95'
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == reference_before
    assert evaluated['reference_identity'] == historical['reference_identity']
    from onnx_splitpoint_tool.workflow.scientific_reporting import project_central_quality_status, _write_reports
    from onnx_splitpoint_tool.accuracy_reporting import observed_coverage
    evaluated.update(model_id='yolo26s', backend=request['producer_identity']['backend'],
        task='detection', variant='full', case_id='full',
        source_request=str(requests[0].relative_to(root)),
        source_request_sha256=hashlib.sha256(requests[0].read_bytes()).hexdigest(),
        collection_eval_run_id=root.name, technical_status='completed',
        cpu_reference_store_manifest=str(Path(stored['manifest_path'])))
    projected = project_central_quality_status({'status': 'ok', 'request_count': 1, 'results': [evaluated]})
    qrows = projected['results']
    _write_reports(root / 'reports/scientific', {'run_id': root.name, 'rows': [],
        'central_quality_results': qrows, 'central_quality_reporting': projected,
        'quality_reporting_policy': request['metric_gate_config']['reporting_policy']})
    proof = read(root / 'reports/scientific/sentinel_coverage.json')
    assert proof['evaluated_images'] == 32 and len(proof['consumers']) == 1, proof
    consumer = proof['consumers'][0]
    assert consumer['variant'] == 'full' and consumer['backend'] == 'deepx_m1'
    assert consumer['candidate_record_count'] == 32 and consumer['reference_identity'] == historical['reference_identity']
    assert consumer['metric'] == 'coco_ap_50_95'
    write(root / 'reports/run_status_summary.json', {'blocking_reasons': []})
    counts = {'central_quality_uncertainty_inconclusive': sum(q['accuracy_assessment']['uncertainty'] == 'inconclusive' for q in qrows)}
    assert guard().continuation_checks(root, 'A', qrows, counts) == []
    for key, value in [('backend', 'tensorrt'), ('collection_eval_run_id', 'another_run'),
                       ('reference_identity', '0' * 64),
                       ('observed_image_ids', []), ('evaluated_images', 30),
                       ('observed_image_ids', qrows[0]['observed_image_ids'][:-1] + [qrows[0]['observed_image_ids'][0]])]:
        damaged = [{**qrows[0], key: value}]
        assert not observed_coverage(damaged, run_id=root.name, run_root=root)['consumers'], key
    original = candidate_path.read_bytes()
    candidate_path.write_bytes(original + b' ')
    assert not observed_coverage(qrows, run_id=root.name, run_root=root)['consumers']
    assert 'sentinel_deepx_full_per_consumer_evidence_incomplete' in guard().continuation_checks(root, 'A', qrows, counts)
    candidate_path.write_bytes(original)
    # Even internally rehashed records cannot replace observed unique IDs.
    for records in (candidate['records'][:30], candidate['records'][:-1] + [candidate['records'][0]]):
        write(candidate_path, {**candidate, 'records': records})
        changed_request = deepcopy(request)
        changed_request['candidate']['sha256'] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        write(requests[0], changed_request)
        damaged = [{**qrows[0], 'source_request_sha256': hashlib.sha256(requests[0].read_bytes()).hexdigest()}]
        assert not observed_coverage(damaged, run_id=root.name, run_root=root)['consumers']
        assert 'sentinel_deepx_full_per_consumer_evidence_incomplete' in guard().continuation_checks(root, 'A', damaged, counts)


@pytest.mark.parametrize('threshold,bad', [(.25, False), (.001, True)])
def test_existing_low_ap_threshold_is_never_raised_to_hide_retained_errors(tmp_path, monkeypatch, threshold, bad):
    array = saved('000000052891')
    _install_runtime(monkeypatch, array)
    artifact = candidate_suite(tmp_path)
    suite = _suite()
    bound = suite._deepx_bind_authoritative_endpoint_contract(tmp_path, {'model_id': 'renamed_detector', 'benchmark_task': 'detection'}, artifact)
    result = suite._deepx_bn6_runtime_semantic_attestation(array, bound, confidence_threshold=threshold)
    assert result['pass'] is not bad
    assert result['confidence_threshold'] == threshold
    if bad:
        assert 'ordered_xyxy_fraction' in result['reason']


@pytest.mark.parametrize('n,k,classes', [(37, 17, 3), (101, 23, 7), (2, 1, 2)])
def test_topk_contract_is_independent_of_candidate_count_classes_and_names(tmp_path, n, k, classes):
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    nodes = []
    def op(kind, inputs, outputs, **attrs):
        nodes.append(h.make_node(kind, inputs, outputs, name='operation_' + str(len(nodes)), **attrs))
    constants = {'stride': np.ones((1, 1, n), np.float32), 'split': np.array([4, classes], np.int64),
        'axis': np.array([-1], np.int64), 'axis2': np.array([2], np.int64), 'k': np.array([k], np.int64),
        'nc': np.array(classes, np.int64), 'tile_classes': np.array([1, 1, classes], np.int64),
        'tile_boxes': np.array([1, 1, 4], np.int64)}
    op('Sub', ['anchors', 'lo'], ['a'])
    op('Add', ['anchors', 'hi'], ['b'])
    op('Concat', ['a', 'b'], ['corners'], axis=1)
    op('Mul', ['corners', 'stride'], ['boxes'])
    op('Sigmoid', ['logits'], ['scores'])
    op('Concat', ['boxes', 'scores'], ['decoded'], axis=1)
    op('Transpose', ['decoded'], ['matrix'], perm=[0, 2, 1])
    op('Split', ['matrix', 'split'], ['xyxy', 'probs'], axis=-1)
    op('ReduceMax', ['probs', 'axis'], ['best'], keepdims=0)
    op('TopK', ['best', 'k'], ['first_values', 'first_indices'], axis=-1)
    op('Unsqueeze', ['first_indices', 'axis'], ['indices'])
    op('Tile', ['indices', 'tile_classes'], ['score_indices'])
    op('GatherElements', ['probs', 'score_indices'], ['candidate_scores'], axis=1)
    op('Flatten', ['candidate_scores'], ['flat_scores'], axis=1)
    op('TopK', ['flat_scores', 'k'], ['values', 'labels'], axis=-1)
    op('Div', ['labels', 'nc'], ['box_indices'])
    op('Mod', ['labels', 'nc'], ['class_indices'], fmod=0)
    op('Flatten', ['indices'], ['flat_indices'], axis=2)
    op('Gather', ['flat_indices', 'box_indices'], ['selected_indices'], axis=0)
    op('Tile', ['selected_indices', 'tile_boxes'], ['expanded_indices'])
    op('GatherElements', ['xyxy', 'expanded_indices'], ['selected_boxes'], axis=1)
    op('Unsqueeze', ['values', 'axis2'], ['selected_scores'])
    op('Unsqueeze', ['class_indices', 'axis2'], ['selected_classes'])
    op('Cast', ['selected_classes'], ['classes_float'], to=T.FLOAT)
    op('Concat', ['selected_boxes', 'selected_scores', 'classes_float'], ['out'], axis=-1)
    inputs = [h.make_tensor_value_info(name, T.FLOAT, [1, 2, n]) for name in ('anchors', 'lo', 'hi')]
    inputs.append(h.make_tensor_value_info('logits', T.FLOAT, [1, classes, n]))
    graph = h.make_graph(nodes, 'unrelated', inputs, [h.make_tensor_value_info('out', T.FLOAT, [1, k, 6])],
        [nh.from_array(value, name) for name, value in constants.items()])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 18)])
    onnx.checker.check_model(model)
    path = tmp_path / 'unrelated.onnx'
    onnx.save(model, path)
    proof = ep.inspect_bn6_candidate_graph(path)
    assert proof['candidate_count'] == k and proof['class_count'] == classes
    candidate_suite(tmp_path, proof=proof)
    array = np.tile(np.array([1, 150, 20, 170, .8, classes - 1], np.float32), (1, k, 1))
    runtime, outputs, _, _ = processor(tmp_path, array)
    assert runtime.process(outputs, original_wh=[640, 480])['detection_count'] == k
