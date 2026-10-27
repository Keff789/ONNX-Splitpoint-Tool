"""Actual normal completion snapshots, never synthetic production head values."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import native_dataset_quality as capture
from onnx_splitpoint_tool.native_three_stage import FastDetectionCompletionRuntime
from test_v279_native_three_stage import _decoded_completion_contract
from test_native_dataset_quality_workflow import prepared
from onnx_splitpoint_tool.workflow import native_dataset_quality as workflow


def _save(path, value):
    path.write_text(json.dumps(value))
    return capture.file_descriptor(path)


@pytest.fixture
def probe(tmp_path):
    outputs, contract = _decoded_completion_contract()
    runtime = FastDetectionCompletionRuntime(contract)
    runtime.process(outputs)
    attestation = runtime.attestation(completed_work_units=1)
    identity = attestation['sentinel_identity']
    timed = {'completion_execution_contract': contract, 'completion_execution_attestation': attestation,
             'completion_sentinel_identity': identity}
    for projected, bound in [('semantic_evidence_repetition_index', 'process_local_repetition_index'),
                             ('semantic_evidence_repetition_id', 'repetition_id'),
                             ('semantic_evidence_runtime_instance_id', 'runtime_instance_id')]:
        if bound in identity:
            timed[projected] = identity[bound]
    rows = []
    for index, (name, array) in enumerate(outputs.items()):
        path = tmp_path / f'output_{index}.bin'
        array.tofile(path)
        descriptor = capture.file_descriptor(path)
        rows.append({**descriptor, 'file': path.name, 'name': name, 'index': index,
                     'shape': list(array.shape), 'dtype': str(array.dtype)})
    manifest = {'schema': 'onnx-splitpoint/native-output-dump', 'schema_version': 4,
                'dump_inference_scope': 'last_measured_completion_source_outputs',
                'completion_sentinel_identity': identity,
                'completion_attestation_sha256': attestation['attestation_sha256'],
                'input_image_sha256': identity.get('input_image_sha256'), 'outputs': rows}
    path = tmp_path / 'native_fifo_output_manifest.json'
    config = {'completion_source_probe_manifest': _save(path, manifest)}
    return config, timed, manifest, path, outputs


def test_actual_normal_dump_and_fast_attestation_supply_readonly_contract_samples(probe):
    config, timed, _, _, original = probe
    outputs = capture._load_completion_source_probe(config, timed)
    for name, array in outputs.items():
        np.testing.assert_array_equal(array, original[name])
        assert not array.flags.writeable
    signature = {name: {'shape': list(array.shape), 'dtype': str(array.dtype)} for name, array in outputs.items()}
    assert capture._completion_source_outputs({'_completion_source_probe_outputs': outputs}, signature) is outputs
    bad = deepcopy(signature)
    bad[next(iter(bad))]['shape'][0] += 1
    with pytest.raises(ValueError, match='actual engine signature'):
        capture._completion_source_outputs({'_completion_source_probe_outputs': outputs}, bad)


@pytest.mark.parametrize('fault', ['bytes', 'sentinel', 'attestation', 'scope', 'file_escape', 'resealed_wrong_values'])
def test_contract_probe_rejects_different_measured_source(probe, fault):
    config, timed, manifest, path, _ = probe
    row = manifest['outputs'][0]
    raw = path.parent / row['file']
    if fault == 'bytes':
        raw.write_bytes(raw.read_bytes() + b'changed')
    elif fault == 'sentinel':
        manifest['completion_sentinel_identity'] = {**manifest['completion_sentinel_identity'], 'completed_work_unit_index': 999}
    elif fault == 'attestation':
        manifest['completion_attestation_sha256'] = '0' * 64
    elif fault == 'scope':
        manifest['dump_inference_scope'] = 'unrelated_probe'
    elif fault == 'file_escape':
        row['file'] = '../output.bin'
    else:
        values = np.fromfile(raw, dtype=np.float32)
        values[0] += 1
        values.tofile(raw)
        row.update(capture.file_descriptor(raw))
    config['completion_source_probe_manifest'] = _save(path, manifest)
    with pytest.raises((ValueError, RuntimeError)):
        capture._load_completion_source_probe(config, timed)


def test_probe_stage_keeps_original_manifest_and_exact_relative_binary(prepared, probe):
    config, _, manifest, source, _ = probe
    prepared.config['completion_source_probe_manifest'] = config['completion_source_probe_manifest']
    _save(prepared.path, prepared.config)
    action = workflow.prepare_native_quality_action(prepared.path, **prepared.kwargs)
    stage = prepared.output / 'stage'
    target = stage / 'inputs/completion_source_probe' / source.name
    assert target.read_bytes() == source.read_bytes()
    for row in manifest['outputs']:
        assert (target.parent / row['file']).read_bytes() == (source.parent / row['file']).read_bytes()
    staged = json.loads((stage / 'capture_config.json').read_text())
    assert staged['completion_source_probe_manifest']['path'] == action['remote_root'] + '/' + target.relative_to(stage).as_posix()
    assert staged['completion_source_probe_manifest']['sha256'] == config['completion_source_probe_manifest']['sha256']
