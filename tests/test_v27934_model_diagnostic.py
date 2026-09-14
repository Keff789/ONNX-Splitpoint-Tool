"""Software validation of private diagnosis, never a simulated hardware gate."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_model_diagnostics_v27934 as diagnostic
from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract, preprocessing_contract_sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('model_probe_v34', ROOT / 'scripts/hailo_model_build_probe_v27934.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    # External vendor version discovery is the only mocked metadata boundary.
    monkeypatch.setattr(backend, '_hailo_sdk_version_token', lambda: 'hailo-dataflow-compiler:5.3.0')
    monkeypatch.delenv('ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST', raising=False)
    monkeypatch.setenv('ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY', 'relaxed')
    model = tmp_path / 'mobilenet_v3_large.onnx'
    import onnx
    from onnx import helper as h, TensorProto as T, numpy_helper as nh
    graph = h.make_graph([h.make_node('Identity', ['constant'], ['logits'])], 'synthetic_identity_fixture',
        [h.make_tensor_value_info('images', T.FLOAT, [1, 3, 224, 224])],
        [h.make_tensor_value_info('logits', T.FLOAT, [1, 10])],
        [nh.from_array(np.arange(10, dtype=np.float32)[None], 'constant')])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid('', 18)], ir_version=10), model)
    calibration = tmp_path / 'calibration'
    calibration.mkdir()
    for index in range(500):
        (calibration / f'fixture_{index:03d}.jpg').write_bytes(f'not-real-calibration-{index}'.encode())
    contract = canonical_image_preprocessing_contract('classification', (224, 224))
    key, payload = backend._hailo_cache_key(model_path=model, activation_part1=None,
        hw_arch='hailo10h', opt_level=1, calib_dir=calibration, calib_count=500,
        calib_batch_size=8, extra_model_script='', start_nodes=None, end_nodes=None,
        preprocessing_contract=contract, effective_calib_count=500,
        calibration_storage='memmap', calibration_memory_cap_bytes=256 * 1024 * 1024,
        net_name='mobilenet_v3_large_full', net_input_shapes={'images': [1, 3, 224, 224]},
        disable_rt_metadata_extraction=True)
    hef = tmp_path / 'cpu' / 'compiled.hef'
    hef.parent.mkdir()
    hef.write_bytes(b'synthetic-byte-identity-not-a-runnable-HEF')
    backend._write_hailo_receipt(hef_path=hef, source_onnx=model, compiler_onnx=model,
        hw_arch='hailo10h', net_name='mobilenet_v3_large_full', preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract), cache_key=key,
        cache_payload=payload, calibration_identity=payload['calibration_identity'], calibration_count=500)
    images = []
    for i in range(16):
        image = tmp_path / f'image_{i}.jpg'
        image.write_bytes(('fixed-development-image-' + str(i)).encode())
        images.append({'id': str(i), 'path': str(image), 'label': i % 5})
    image_manifest = tmp_path / 'images.json'
    image_manifest.write_text(json.dumps(images))
    args = dict(cpu_hef=hef, source_onnx=model, compiler_onnx=model,
        calibration_dir=calibration, images_json=image_manifest,
        venv=sys.executable, timeout_s=3600)
    return args, diagnostic.prepare_request(**args)


def test_prepare_is_receipt_bound_and_no_implicit_force_or_publication(baseline, tmp_path):
    args, request = baseline
    original = {Path(args[k]): Path(args[k]).read_bytes() for k in ('cpu_hef', 'source_onnx', 'images_json')}
    kwargs = diagnostic.builder_arguments(request, tmp_path / 'private')
    assert kwargs['force'] is True and kwargs['publish_artifacts'] is False
    assert kwargs['fixup'] is False and kwargs['add_conv_defaults'] is False
    assert kwargs['onnx_path'] == request['compiler_onnx']['path']
    assert (kwargs['opt_level'], kwargs['calib_count'], kwargs['calib_batch_size']) == (1, 500, 8)
    assert kwargs['compute_device'] == 'gpu' and kwargs['wsl_timeout_s'] == 3600
    assert {p: p.read_bytes() for p in original} == original


@pytest.mark.parametrize('field', ['source_onnx', 'compiler_onnx', 'cpu_hef', 'cpu_receipt'])
def test_frozen_file_changes_block_before_dispatch(baseline, field):
    _, request = baseline
    Path(request[field]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='frozen_input_changed'):
        diagnostic.validate_request(request)


@pytest.mark.parametrize('field,value', [('role', 'part1'), ('family', 'hailo8'), ('model', 'resnet50'), ('preset', 'fast')])
def test_request_cannot_relabel_baseline(baseline, field, value):
    _, request = baseline
    request[field] = value
    with pytest.raises(ValueError, match='baseline_recipe_or_graph_mismatch'):
        diagnostic.validate_request(request)


def test_calibration_change_is_not_ignored(baseline):
    args, request = baseline
    (args['calibration_dir'] / 'new.jpg').write_bytes(b'new')
    with pytest.raises(ValueError, match='calibration_identity_changed'):
        diagnostic.validate_request(request)


def test_changing_directory_and_request_identity_cannot_override_cpu_receipt(baseline, tmp_path):
    _, request = baseline
    other = tmp_path / 'unrelated_calibration'
    other.mkdir()
    (other / 'wrong.jpg').write_bytes(b'wrong')
    request['calibration_dir'] = str(other)
    request['calibration_identity'] = backend._calibration_identity(other, strict=False)
    with pytest.raises(ValueError, match='calibration_identity_changed'):
        diagnostic.validate_request(request)


def test_manifest_cannot_alias_an_unrelated_calibration_directory(baseline, tmp_path):
    args, request = baseline
    manifest = tmp_path / 'manifest.json'
    items = [{'relative_path': f'different_{i:03d}.jpg'} for i in range(500)]
    manifest.write_text(json.dumps({'items': items, 'item_count': 500}))
    with pytest.raises(ValueError, match='manifest_does_not_bind_selected'):
        diagnostic.calibration_records(args['calibration_dir'], diagnostic.file_identity(manifest))


def test_calibration_bytes_frozen_even_when_relaxed_metadata_matches(baseline):
    args, request = baseline
    import os
    image = args['calibration_dir'] / 'fixture_000.jpg'
    stat = image.stat()
    image.write_bytes(b'x' * stat.st_size)
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='selected_calibration_files_changed'):
        diagnostic.validate_request(request)


def test_random_fallback_does_not_become_same_recipe_model_pass(baseline, tmp_path):
    _, request = baseline
    import shutil
    work = tmp_path / 'private'
    work.mkdir()
    hef = work / 'compiled.hef'
    shutil.copyfile(request['cpu_hef']['path'], hef)
    receipt = copy.deepcopy(request['cpu_build_receipt'])
    receipt.update(diagnostic_only=True, publish_artifacts=False)
    backend._atomic_write_json(backend._hailo_receipt_path(hef), receipt)
    events = [{'phase': name, 'event': 'completed', 'monotonic_s': 1} for name in ('translate', 'optimize', 'compile', 'publication')]
    result = {'ok': True, 'hef_path': str(hef), 'calib_info': {'source': 'random', 'used_count': 500},
        'details': {'phase_events': events}}
    summary = probe.summarize_result(result, request, work, [])
    assert summary['model_build_status'] == 'recipe_mismatch'
    assert summary['recipe_matches_cpu_baseline'] is False
    assert summary['gpu_execution_status'] == 'gpu_execution_unproven'


def test_different_original_and_compiler_graph_use_bound_compiler_identity(baseline, tmp_path):
    args, request = baseline
    import onnx
    original = tmp_path / 'original_mobilenet.onnx'
    graph = onnx.load(args['source_onnx'])
    graph.doc_string = 'Original before preparation; fixture with distinct serialized graph bytes.'
    onnx.save(graph, original)
    receipt = copy.deepcopy(request['cpu_build_receipt'])
    receipt['source_onnx_sha256'] = backend._bare_file_sha256(original)
    backend._atomic_write_json(backend._hailo_receipt_path(args['cpu_hef']), receipt)
    frozen = diagnostic.prepare_request(**{**args, 'source_onnx': original})
    assert frozen['source_onnx']['sha256'] != frozen['compiler_onnx']['sha256']
    assert receipt['cache_payload']['model_sha256'] == frozen['compiler_onnx']['sha256']
    work = tmp_path / 'private'
    work.mkdir()
    hef = work / 'compiled.hef'
    hef.write_bytes(Path(args['cpu_hef']).read_bytes())
    private = copy.deepcopy(receipt)
    private.update(source_onnx_sha256=frozen['compiler_onnx']['sha256'], diagnostic_only=True, publish_artifacts=False)
    backend._atomic_write_json(backend._hailo_receipt_path(hef), private)
    events = [{'phase': 'optimize', 'event': 'started', 'monotonic_s': 10},
              {'phase': 'optimize', 'event': 'completed', 'monotonic_s': 20}]
    summary = probe.summarize_result({'ok': True, 'hef_path': str(hef),
        'calib_info': {'source': frozen['calibration_dir'], 'used_count': 500},
        'details': {'phase_events': events, 'compiler_context': {'device': 'gpu', 'gpu_index': '0'}}},
        frozen, work, [{'gpu_index': 0, 'sm_utilization_percent': 20, 'owned_compiler_descendant': True,
            'interval_start_monotonic_s': 12, 'observed_monotonic_s': 13}])
    assert summary['recipe_matches_cpu_baseline'] is True
    # Actual resolver's string GPU index is matched to numeric pmon index.
    assert summary['gpu_execution_status'] == 'pass'
    # The incomplete synthetic phase record is never a hardware build PASS.
    assert summary['model_build_status'] == 'build_completion_unproven'
    assert summary['claim_eligible'] is False


def test_wrong_named_graph_never_substitutes_receipt_bound_graph(baseline, tmp_path):
    args, _ = baseline
    other = tmp_path / 'other' / 'mobilenet_v3_large.onnx'
    other.parent.mkdir()
    other.write_bytes(b'wrong-download-same-name')
    with pytest.raises(ValueError, match='compiler_graph_does_not_match'):
        diagnostic.prepare_request(**{**args, 'compiler_onnx': other})


@pytest.mark.parametrize('timeout', [0, 180, -1, 100000, True, 3600.0])
def test_model_build_never_inherits_compute_smoke_or_infinite_budget(baseline, timeout):
    args, _ = baseline
    with pytest.raises(ValueError, match='budget'):
        diagnostic.prepare_request(**{**args, 'timeout_s': timeout})


def test_private_environment_preserves_parent_and_recipe(baseline, tmp_path):
    _, request = baseline
    parent = {'LD_LIBRARY_PATH': '/vendor/unchanged', 'HOME': '/original-home', 'TF_NUM_INTRAOP_THREADS': '4'}
    before = dict(parent)
    env = diagnostic.private_environment(request, tmp_path / 'diagnostic', parent)
    assert parent == before
    assert env['LD_LIBRARY_PATH'] == parent['LD_LIBRARY_PATH']
    assert env['HOME'] == '/original-home'
    assert env['TF_NUM_INTRAOP_THREADS'] == '4'
    assert env['ONNX_SPLITPOINT_HAILO_CACHE_ENABLED'] == env['ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED'] == '0'
    assert env['ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE'] == 'memmap'
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCH_HOME', 'TFHUB_CACHE_DIR', 'ONNX_SPLITPOINT_HAILO_CACHE_ROOT', 'ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT'):
        assert Path(env[key]).is_relative_to(tmp_path / 'diagnostic')


@pytest.mark.parametrize('change', [{}, {'sm_utilization_percent': 0}, {'owned_compiler_descendant': False}, {'gpu_index': 1}, {'interval_start_monotonic_s': 9.9}, {'observed_monotonic_s': 21}, {'sm_utilization_percent': float('nan')}])
def test_gpu_requires_own_SM_work_strictly_inside_optimization(change):
    events = [{'phase': 'optimize', 'event': 'started', 'monotonic_s': 10}, {'phase': 'optimize', 'event': 'completed', 'monotonic_s': 20}]
    sample = {'pid': 10, 'gpu_index': 0, 'sm_utilization_percent': 15,
        'owned_compiler_descendant': True, 'interval_start_monotonic_s': 12, 'observed_monotonic_s': 13, **change}
    evidence = diagnostic.attributable_gpu_evidence([sample], events, gpu_index=0)
    assert evidence['status'] == ('pass' if not change else 'gpu_execution_unproven')


def arrays_fixture():
    ids = [str(i) for i in range(16)]
    logits = np.zeros((16, 10), dtype=np.float32)
    logits[:, 0] = 2
    feed = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
    rows = {name: {'ids': ids, 'logits': logits.copy(), 'feed': feed.copy(), 'origin': 'synthetic_unit_fixture'}
        for name in ('source_float', 'build_float', 'cpu_hef', 'gpu_hef')}
    return ids, rows


def test_equal_bad_accuracy_does_not_become_quality_or_claim_pass():
    ids, stages = arrays_fixture()
    report = diagnostic.compare_classification_arrays(stages, labels=[1] * 16, ids=ids)
    assert report['stages']['cpu_hef']['top1_correct'] == report['stages']['gpu_hef']['top1_correct'] == 0
    assert report['quality_status'] == 'not_evaluated_against_campaign_gate'
    assert report['claim_eligible'] is False and report['original_quality_decision_preserved'] is True
    assert report['stages']['quantized_emulation']['status'] == 'not_available'
    assert report['comparisons']['cpu_hef_vs_gpu_hef']['max_abs_difference'] == 0


def test_additional_gpu_quality_loss_visible_separately():
    ids, stages = arrays_fixture()
    stages['gpu_hef']['logits'][:, 1] = 3
    report = diagnostic.compare_classification_arrays(stages, labels=[0] * 16, ids=ids)
    assert report['comparisons']['cpu_hef_vs_gpu_hef']['correct_count_delta'] == -16
    assert report['comparisons']['cpu_hef_vs_gpu_hef']['top1_changed_count'] == 16
    assert report['thresholds_changed'] is False


@pytest.mark.parametrize('problem', ['nonfinite', 'wrong_feed', 'wrong_ids', 'wrong_axis'])
def test_numeric_and_binding_errors_remain_visible(problem):
    ids, stages = arrays_fixture()
    if problem == 'nonfinite':
        stages['gpu_hef']['logits'][0, 0] = np.nan
        report = diagnostic.compare_classification_arrays(stages, labels=[0] * 16, ids=ids)
        assert report['stages']['gpu_hef']['status'] == 'invalid_numeric'
        assert 'cpu_hef_vs_gpu_hef' not in report['comparisons']
        return
    if problem == 'wrong_feed':
        stages['gpu_hef']['feed'][0] += 1
    elif problem == 'wrong_ids':
        stages['gpu_hef']['ids'] = ids[::-1]
    else:
        stages['gpu_hef']['logits'] = stages['gpu_hef']['logits'].T
    with pytest.raises(ValueError):
        diagnostic.compare_classification_arrays(stages, labels=[0] * 16, ids=ids)


def test_export_keeps_evidence_and_excludes_binary_builds_and_raw_arrays(tmp_path):
    output = tmp_path / 'report'
    output.mkdir()
    (output / 'summary.json').write_text('{"model_build_status":"not_run"}')
    for name in ('model.onnx', 'compiled.hef', 'quantized.har', 'raw_tensors.npz', 'image.jpg', 'unknown.json'):
        (output / name).write_bytes(b'private')
    archive = probe.export_evidence(output)
    import zipfile
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ['report/summary.json']
        assert z.testzip() is None


def test_supervisor_entry_is_a_standalone_executable_not_import_side_effect():
    import subprocess
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/hailo_model_build_probe_v27934.py'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'prepare' in result.stdout and 'execute' in result.stdout and 'compare' in result.stdout


def test_yolo_bridge_same_actual_feed_exposes_part2_runtime_difference(tmp_path):
    import shutil
    from PIL import Image
    from test_v27931_hailo26_boundary_diagnostics import synthetic_reference_graphs
    import hailo10_yolo26_boundary_probe_v27931 as boundary
    import hailo_boundary_diagnostics_v27931 as packet
    files, bridge, expected = synthetic_reference_graphs(tmp_path / 'graphs')
    output = tmp_path / 'result'
    output.mkdir()
    shutil.copyfile(bridge, output / 'bound_bridge.onnx')
    raw = np.round(expected.transpose(0, 2, 1) / .1).astype(np.uint8)
    wrong = expected.transpose(0, 2, 1).copy()
    wrong[..., 4] = -3
    packet.dump_packet(output, {'A_000': np.array(Image.open(files['image']).convert('RGB')),
        'B_000': raw, 'C_000': raw.reshape(1, 6, 2), 'E_000': wrong},
        {'status': 'captured', 'stages': {'A': {'names': ['images']}, 'B': {'names': ['raw_name']},
        'C': {'names': ['cut']}, 'E': {'names': ['output0']}},
        'runtime_io': {'runtime_output_quantization': {'raw_name': {'scale': .1, 'zero_point': 0.}},
        'outputs_dequantized': False}})
    request = {'remote_artifacts': {'bridge': {'sha256': packet.sha256(bridge)}},
        'reference_part1_sha256': packet.sha256(files['part1']),
        'reference_part2_sha256': packet.sha256(files['part2']),
        'engine_io': {'inputs': [{'name': 'cut'}]}}
    boundary.offline_reference(request, files, output)
    report, arrays = packet.load_packet(output)
    comparison = report['comparisons']['E_vs_bound_bridge_P2_same_C']
    assert comparison['tensors']['output0']['max_abs'] > 3
    assert comparison['semantic_pass'] is False
    np.testing.assert_allclose(arrays['P2_same_C_000'], expected.transpose(0, 2, 1), atol=1e-6)
    np.testing.assert_array_equal(arrays['E_000'], wrong)
    assert report['part2_same_feed_comparison']['raw_outputs_corrected'] is False


def test_runtime_controller_reads_existing_setup_without_registry_migration(tmp_path):
    import hailo_model_runtime_probe_v27934 as runtime
    import yaml
    registry = tmp_path / 'hardware.yaml'
    registry.write_text(yaml.safe_dump({'schema': 'onnx-splitpoint/hardware-setups', 'schema_version': 2,
        'hardware_setups': [{'id': 'diagnostic_h10', 'accelerator': 'hailo10h',
            'host': {'address': '192.0.2.5', 'user': 'nx', 'port': 22}}]}))
    before = registry.read_bytes()
    remote = runtime.hardware_setup(registry, 'diagnostic_h10')
    assert remote == {'host': '192.0.2.5', 'user': 'nx', 'port': 22}
    assert registry.read_bytes() == before


def test_runtime_plan_binds_exact_pair_and_source_graph_without_upload(baseline, tmp_path):
    import hailo_model_runtime_probe_v27934 as runtime
    import shutil
    _, request = baseline
    directory = tmp_path / 'private_build'
    directory.mkdir()
    hef = directory / 'compiled.hef'
    shutil.copyfile(request['cpu_hef']['path'], hef)
    receipt = copy.deepcopy(request['cpu_build_receipt'])
    receipt.update(diagnostic_only=True, publish_artifacts=False)
    backend._atomic_write_json(backend._hailo_receipt_path(hef), receipt)
    diagnostic.write_json(directory / 'request.json', request)
    summary = {'model_build_status': 'pass', 'recipe_matches_cpu_baseline': True,
        'private_hef': diagnostic.file_identity(hef), 'fixture': 'synthetic_metadata_not_hardware_evidence'}
    diagnostic.write_json(directory / 'summary.json', summary)
    resolved, observed, bound = runtime.prepare_runtime_request(directory, 'diagnostic_h10')
    assert bound['input_shape'] == [1, 3, 224, 224] and bound['class_count'] == 10
    assert bound['expected_source_onnx_sha256'] == request['source_onnx']['sha256']
    assert bound['expected_compiler_sha256'] == request['compiler_onnx']['sha256']
    assert len(bound['images']) == 16 and bound['cpu_hef']['path'] == 'cpu.hef'
    assert bound['compiler_invoked'] is False and bound['energy_invoked'] is False
    hef.write_bytes(b'changed_private_gpu_artifact')
    with pytest.raises(ValueError, match='frozen_input_changed'):
        runtime.prepare_runtime_request(directory, 'diagnostic_h10')


def test_runtime_collector_uses_actual_feed_for_real_original_and_build_ORT(baseline, tmp_path):
    import hailo_model_runtime_probe_v27934 as runtime
    _, request = baseline
    output = tmp_path / 'runtime_result'
    output.mkdir()
    ids = [row['id'] for row in request['images']]
    remote = {'model': request['model'], 'setup_id': 'diagnostic_h10', 'input_shape': [1, 3, 224, 224],
        'class_count': 10, 'cpu_hef': request['cpu_hef'], 'gpu_hef': request['cpu_hef'],
        'expected_source_onnx_sha256': request['source_onnx']['sha256'],
        'expected_compiler_sha256': request['compiler_onnx']['sha256']}
    feed = np.zeros((16, 224, 224, 3), np.float32)
    logits = np.tile(np.arange(10, dtype=np.float32), (16, 1))
    arrays_path = output / 'runtime_arrays.npz'
    np.savez_compressed(arrays_path, cpu_hef_logits=logits, gpu_hef_logits=logits + 1,
        cpu_hef_logical_feed=feed, gpu_hef_logical_feed=feed)
    metadata = {'runtime_status': 'pass', 'family': 'hailo10h', 'model': request['model'], 'setup_id': 'diagnostic_h10',
        'expected_source_onnx_sha256': remote['expected_source_onnx_sha256'],
        'expected_compiler_sha256': remote['expected_compiler_sha256'],
        'logical_feed_equal': True, 'compiler_invoked': False, 'frozen_inputs_unchanged': True,
        'arrays': diagnostic.file_identity(arrays_path),
        'stages': {role: {'ids': ids, 'inference_count': 16, 'artifact_sha256': request['cpu_hef']['sha256'],
                         'runtime_status': 'pass', 'hef_readability_status': 'pass',
                         'fixture': 'synthetic_external_device_boundary'} for role in ('cpu_hef', 'gpu_hef')}}
    diagnostic.write_json(output / 'runtime_result.json', metadata)
    report = runtime.float_comparison(request, remote, output)
    assert report['comparisons']['source_float_vs_build_float']['max_abs_difference'] == 0
    assert report['comparisons']['build_float_vs_cpu_hef']['max_abs_difference'] == 0
    assert report['comparisons']['cpu_hef_vs_gpu_hef']['max_abs_difference'] == 1
    assert report['claim_eligible'] is False and report['quality_status'] == 'not_evaluated_against_campaign_gate'
    metadata['stages']['gpu_hef']['ids'] = ids[::-1]
    diagnostic.write_json(output / 'runtime_result.json', metadata)
    with pytest.raises(ValueError, match='wrong_image_ids'):
        runtime.float_comparison(request, remote, output)


def test_yolo_four_image_manifest_requires_distinct_fixed_ids(tmp_path):
    import hailo10_yolo26_boundary_probe_v27931 as boundary
    files = []
    for index in range(4):
        path = tmp_path / f'image_{index}.jpg'
        path.write_bytes(f'fixed-source-{index}'.encode())
        files.append({'id': str(index), 'path': str(path)})
    manifest = tmp_path / 'four.json'
    manifest.write_text(json.dumps(files))
    rows = boundary.fixed_four_images(manifest)
    assert len(rows) == 4 and len({r['sha256'] for r in rows}) == 4
    files[3]['id'] = files[0]['id']
    manifest.write_text(json.dumps(files))
    with pytest.raises(ValueError, match='unique'):
        boundary.fixed_four_images(manifest)
