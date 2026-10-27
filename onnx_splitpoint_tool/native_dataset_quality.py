"""Untimed dataset observations of the existing Native FIFO completion path.

The Native runners own inference, preprocessing and completion.  This module
only selects the frozen images, records their actual completed detections and
exports the ordinary central-management request.  It never evaluates AP.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from . import native_detection_postprocess as postprocess
from .native_output_endpoint import load_authoritative_output_contract, runtime_output_contract
from .native_three_stage import verify_fast_completion_attestation
from .native_dataset_quality_export import export_native_quality_request
from .native_command_contract import canonical_json_sha256, verify_native_command_contract
from .quality_cache import image_ids_fingerprint
from .quality_service import _validate_detection_quality_contract


IMAGE_COUNT = 5000


def file_descriptor(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def checked_json(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(descriptor['path'])).expanduser().resolve(strict=True)
    actual = file_descriptor(path)
    if any(actual[key] != descriptor[key] for key in ('sha256', 'size_bytes')):
        raise ValueError('Native dataset Quality input descriptor changed')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Native dataset Quality JSON object required')
    return value


def _timed_model_identity(timed, identity):
    """Read an absent H8 model alias only from its verified Native command."""
    aliases = [timed.get(name) for name in ('model_id', 'model') if timed.get(name)]
    raw = timed.get('native_command_contract')
    if raw is not None:
        command, status = verify_native_command_contract(raw, expected_identity={
            'model': identity['model_id'], 'case': identity['case_id'],
            'setup_id': identity['setup_id'], 'backend': identity['backend'],
            'precision': identity['runtime_precision_identity'],
        })
        if command is None:
            raise ValueError('Native dataset Quality timed Native command invalid: ' + status)
        model = command['model']
        if any(alias != model for alias in aliases):
            raise ValueError('Native dataset Quality timed model aliases conflict')
        return model
    if not aliases or any(alias != aliases[0] for alias in aliases):
        raise ValueError('Native dataset Quality timed model authority missing or conflicting')
    return aliases[0]


def load_capture_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    config = json.loads(path.read_text())
    if (config.get('schema') != 'onnx-splitpoint/native-dataset-quality-config'
            or config.get('schema_version') != 1):
        raise ValueError('Native dataset Quality configuration schema invalid')
    identity = config.get('identity') or {}
    if (identity.get('backend') not in {'hailo8_to_trt', 'hailo10h_to_trt'}
            or any(not identity.get(key) for key in (
                'eval_run_id', 'model_id', 'case_id', 'setup_id',
                'source_run_id', 'runtime_precision_identity'))):
        raise ValueError('Native dataset Quality identity incomplete')
    if config.get('performance_claims_emitted') is not False:
        raise ValueError('Native dataset Quality must not emit performance claims')
    native = config.get('native') or {}
    if native.get('engine_build_forbidden') is not True:
        raise ValueError('Native dataset Quality requires explicit no-build')
    if native.get('task', 'detection') != 'detection':
        raise ValueError('Native dataset Quality supports detection only')
    ground_truth = checked_json(config['ground_truth'])
    rows = ground_truth.get('records')
    if not isinstance(rows, list) or len(rows) != IMAGE_COUNT:
        raise ValueError('Native dataset Quality requires exactly 5000 frozen images')
    ids = [row.get('image_id') for row in rows]
    if (len(set(ids)) != IMAGE_COUNT or any(not isinstance(x, str) or Path(x).name != x for x in ids)
            or ids != sorted(ids)):
        raise ValueError('Native dataset Quality image identities invalid')
    if any(not isinstance(row.get('ground_truth'), list)
           or not isinstance(row.get('image_sha256'), str) or len(row['image_sha256']) != 64
           for row in rows):
        raise ValueError('Native dataset Quality ground truth/image hashes missing')
    if int(config['quality_contract']['dataset']['image_count']) != IMAGE_COUNT:
        raise ValueError('Native dataset Quality contract population differs')
    quality, quality_sha = _validate_detection_quality_contract(config['quality_contract'], role='Native capture')
    if quality_sha != config.get('quality_contract_sha256'):
        raise ValueError('Native dataset Quality contract hash differs')
    if image_ids_fingerprint(ids) != quality['dataset']['image_ids_sha256']:
        raise ValueError('Native dataset Quality frozen image identities differ')
    truth = [{'image_id': row['image_id'], 'ground_truth': row['ground_truth']} for row in rows]
    if canonical_json_sha256(truth) != quality['dataset']['ground_truth_sha256']:
        raise ValueError('Native dataset Quality frozen ground truth differs')
    statistics = config.get('statistics') or {}
    if statistics.get('bootstrap_repetitions') != 1000 or statistics.get('seed') != 20260710:
        raise ValueError('Native dataset Quality requires unchanged B1000/seed20260710')
    prior = checked_json(config['prior_artifact_selection_file'])
    if config['prior_artifact_selection'] != prior:
        raise ValueError('Native dataset Quality prior artifact authority differs')
    if Path(native['native_split_quality_binding']).resolve() != Path(config['prior_artifact_selection_file']['path']).resolve():
        raise ValueError('Native dataset Quality runtime artifact selection differs')
    # This is the original measured Native result, not a new performance run.
    timed = checked_json(config['timed_fps_authority'])
    if (str(timed.get('case') or timed.get('case_id') or '') != identity['case_id']
            or _timed_model_identity(timed, identity) != identity['model_id']):
        raise ValueError('Native dataset Quality timed source case/model differs')
    if timed.get('completion_execution_contract') != config['timed_completion_contract']:
        raise ValueError('Native dataset Quality timed completion authority differs')
    if timed.get('native_split_quality_binding') != prior:
        raise ValueError('Native dataset Quality timed artifact authority differs')
    if timed.get('setup_id') != identity['setup_id'] or timed.get('precision') != identity['runtime_precision_identity']:
        raise ValueError('Native dataset Quality timed setup/precision authority differs')
    if timed.get('completion_runtime_mode') != config.get('completion_runtime_mode'):
        raise ValueError('Native dataset Quality timed runtime mode differs')
    if not (type(timed.get('completed_frames')) is int and timed['completed_frames'] > 0):
        raise ValueError('Native dataset Quality timed completed source missing')
    config['_ground_truth_records'] = rows
    config['_config_file'] = file_descriptor(path)
    return config


def _runtime_factory(config: Mapping[str, Any]):
    if config['identity']['backend'] == 'hailo10h_to_trt':
        from .native_dataset_quality_h10 import H10NativeQualityRuntime
        return H10NativeQualityRuntime
    from .native_dataset_quality_h8 import H8NativeQualityRuntime
    return H8NativeQualityRuntime


def _sources(paths) -> list[dict[str, Any]]:
    result = []
    for path in sorted(set(map(str, paths))):
        record = file_descriptor(path)
        parts = Path(path).parts
        markers = [index for index, part in enumerate(parts)
                   if part in {'onnx_splitpoint_tool', 'scripts'}]
        if not markers:
            raise ValueError('Native dataset Quality source outside normal Product closure')
        record['relative_path'] = Path(*parts[markers[-1]:]).as_posix()
        result.append(record)
    return result


def _completion_contract(config, signature, image):
    dummy = {name: np.zeros(spec['shape'], dtype=spec['dtype']) for name, spec in signature.items()}
    declaration = load_authoritative_output_contract(
        Path(config['native']['benchmark_set']), backend='tensorrt',
        model_id=config['identity']['model_id'], variant='full', task='detection')
    source = runtime_output_contract('detection', dummy, raw_fallback=False, declared_contract=declaration)
    with Image.open(image) as opened:
        wh = list(opened.size)
    policy = config['timed_completion_contract']['processor_contract']
    contract = postprocess.build_detection_completion_execution_contract(
        model_id=config['identity']['model_id'], outputs=dummy,
        input_hw=policy['input_hw'], original_wh=wh, source_endpoint_contract=source,
        preprocess={'mode': config['native']['preprocess_mode'], 'rgb': True,
                    'pad_value': config['native']['letterbox_pad_value']},
        confidence_threshold=policy['confidence_threshold'],
        iou_threshold=policy['iou_threshold'], max_detections=policy['max_detections'])
    # Geometry and implementation provenance may be freshly bound; numeric
    # source/completion policy is selected by the measured Native endpoint.
    for key in ('completion_mode', 'task', 'model_id', 'model_family', 'execution_policy', 'yolov7_head_mapping'):
        if contract[key] != config['timed_completion_contract'][key]:
            raise ValueError('Native dataset Quality timed completion policy differs: ' + key)
    for key in ('confidence_threshold', 'iou_threshold', 'max_detections',
                'input_hw', 'source_output_format', 'output_coordinate_space',
                'host_nms_applied', 'source_nms_attested'):
        if contract['processor_contract'].get(key) != policy.get(key):
            raise ValueError('Native dataset Quality timed processor policy differs: ' + key)
    # Compare the complete semantic policy, including candidate selection and
    # coordinate normalization.  Image geometry and truthful implementation /
    # declaration provenance are freshly attested for this capture.
    geometry_and_provenance = {
        'contract_sha256', 'implementation_artifacts', 'original_wh',
        'letterbox_geometry_contract', 'letterbox_geometry_contract_sha256',
        'source_output_endpoint_attestation_sha256',
    }
    semantic = lambda value: {key: item for key, item in value.items()
                              if key not in geometry_and_provenance}
    if semantic(contract['processor_contract']) != semantic(policy):
        raise ValueError('Native dataset Quality complete timed processor semantics differ')
    source_semantic = lambda value: {key: item for key, item in value.items()
                                     if key != 'output_endpoint_attestation_sha256'}
    if source_semantic(contract['source_endpoint']) != source_semantic(config['timed_completion_contract']['source_endpoint']):
        raise ValueError('Native dataset Quality timed source endpoint differs')
    return contract


def capture_native_quality(config_path: str | Path, out_dir: str | Path, *, runtime_factory=None):
    """Record exactly one real Native completion for each frozen dataset image."""
    config = load_capture_config(config_path)
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    factory = runtime_factory or _runtime_factory(config)
    records, evidence = [], []
    before = None
    try:
        with factory(config['native'], out / 'runtime') as native:
            from . import native_three_stage
            source_paths = [__file__, inspect.getsourcefile(postprocess),
                            inspect.getsourcefile(native_three_stage),
                            inspect.getsourcefile(type(native)),
                            inspect.getsourcefile(export_native_quality_request), *native.source_paths]
            before = _sources(source_paths)
            for ordinal, expected in enumerate(config['_ground_truth_records']):
                image = Path(config['dataset_root']) / expected['image_id']
                image_file = file_descriptor(image)
                if image_file['sha256'] != expected['image_sha256']:
                    raise ValueError('Native dataset Quality source image changed: ' + expected['image_id'])
                contract = _completion_contract(config, native.output_signature, image)
                if not records:
                    config['base_completion_contract'] = deepcopy(contract)
                runtime = postprocess.create_detection_completion_runtime(
                    contract, runtime_mode=config.get('completion_runtime_mode', 'auto'),
                    observation_relation='same_hotloop_sentinel', source_paths=tuple(native.source_paths),
                    sentinel_context={'image_id': expected['image_id'],
                                      'input_image_sha256': image_file['sha256']})
                observed = native.run_image(image, runtime)
                if observed['native_completed_count'] != 1 or runtime.completed_count != 1:
                    raise ValueError('Native dataset Quality image did not complete exactly once')
                attestation = observed['completion_execution_attestation']
                if observed['completion_execution_contract'] != contract:
                    raise ValueError('Native dataset Quality FIFO completion contract changed')
                if attestation.get('observation_relation') == 'postflight_oracle_sentinel':
                    verified = verify_fast_completion_attestation(attestation, execution_contract=contract)
                else:
                    verified = postprocess.verify_detection_completion_execution_attestation(
                        attestation, execution_contract=contract)
                if verified['completed_work_units'] != 1:
                    raise ValueError('Native dataset Quality attested completion count differs')
                detections = deepcopy(verified['last_result']['detections'])
                records.append({'image_id': expected['image_id'], 'candidate': detections,
                                'ground_truth': deepcopy(expected['ground_truth'])})
                snapshot = observed['prepared_input_snapshot']
                if isinstance(snapshot, list):
                    snapshot = {'inputs': snapshot}
                evidence.append({'image_id': expected['image_id'], 'image_sha256': image_file['sha256'],
                    'execution_contract': contract, 'execution_attestation': verified,
                    'prepared_input_snapshot': snapshot,
                    'timing_input_binding': observed['timing_input_binding'],
                    'boundary_proof': observed['runtime_boundary_evidence'], 'native_completed_count': 1})
                if (ordinal + 1) % 100 == 0:
                    print(json.dumps({'native_quality_completed_images': ordinal + 1,
                                      'expected_images': IMAGE_COUNT}), flush=True)
            if before != _sources(source_paths):
                raise ValueError('Native dataset Quality implementation changed during capture')
        return export_native_quality_request(config, records, evidence, before, out)
    except Exception as error:
        (out / 'native_quality_capture_failure.json').write_text(json.dumps({
            'valid_capture': False, 'expected_images': IMAGE_COUNT,
            'completed_images': len(records), 'error': repr(error),
            'performance_claims_emitted': False}, indent=2) + '\n')
        raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = capture_native_quality(args.config, args.out)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
