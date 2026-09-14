"""S6: four fixed YOLO26s images through real Full and quality record paths.

Shares the classification probe's launcher, lock, staging, supervisor, SSH and
collection. The second processing pass consumes copied raw outputs; it performs
no additional inference and never serves as a four-image COCO-AP acceptance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_common import SCHEMA, compare, digest, exact_artifact, isolated_product, read_json, save_tensors, sha_token, stats, write_json

MODEL = 'yolo26s'
SAMPLE_COUNT = 4
MAX_INPUT_BYTES = 32 * 1024 * 1024
EXTRA_PRODUCT_SOURCES = (
    ('onnx_splitpoint_tool/native_detection_postprocess.py', 'splitpoint_runners/native_detection_postprocess.py'),
    ('onnx_splitpoint_tool/native_output_endpoint.py', 'splitpoint_runners/native_output_endpoint.py'),
    ('onnx_splitpoint_tool/runners/harness/yolo.py', 'splitpoint_runners/harness/yolo.py'),
)


def select_four_samples(candidates, explicit_ids=None):
    """Freeze ordered identities using only image metadata before any inference."""
    from PIL import Image
    by_id = {item['image_id']: item for item in candidates}
    if len(by_id) != len(candidates):
        raise ValueError('detection_duplicate_image_id')
    if explicit_ids is not None:
        if len(explicit_ids) != SAMPLE_COUNT or len(set(explicit_ids)) != SAMPLE_COUNT:
            raise ValueError('detection_requires_four_unique_fixed_image_ids')
        if any(image_id not in by_id for image_id in explicit_ids):
            raise ValueError('detection_fixed_image_id_not_in_original_request')
        candidates = [by_id[image_id] for image_id in explicit_ids]
    selected, ratios = [], set()
    for item in candidates:
        with Image.open(item['path']) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError('detection_invalid_image_geometry')
        ratio = Fraction(width, height)
        if ratio in ratios:
            if explicit_ids is not None:
                raise ValueError('detection_four_distinct_aspect_ratios_required')
            continue
        ratios.add(ratio)
        selected.append({**item, 'original_wh': [width, height], 'aspect_ratio': str(ratio)})
        if len(selected) == SAMPLE_COUNT:
            break
    if len(selected) != SAMPLE_COUNT:
        raise ValueError('detection_four_distinct_aspect_ratios_not_available_in_bounded_selection')
    if sum(Path(item['path']).stat().st_size for item in selected) > MAX_INPUT_BYTES:
        raise ValueError('detection_selected_image_byte_budget_exceeded')
    return selected


def prepare_detection_model(run, target, remote, args, output):
    from collect_smokes import BUNDLE, copy_source_snapshot, select_samples
    if args.models != [MODEL]:
        raise ValueError('detection_control_requires_exact_yolo26s_model')
    suite = run / f'models/{MODEL}/benchmark_set/legacy_suite'
    plan = read_json(suite / 'benchmark_plan.json')
    matches = [row for row in plan.get('runs', []) if row.get('id') == 'deepx_m1_full']
    if len(matches) != 1 or matches[0].get('setup_id') != target['id']:
        raise ValueError('detection_full_plan_setup_not_unique')
    runrow = dict(matches[0])
    contract_path = suite / 'deepx/deepx_m1/full/output_contract.json'
    contract = read_json(contract_path)
    if contract.get('model_id') != MODEL:
        raise ValueError('detection_model_contract_mismatch')
    inp = contract.get('input') or {}
    if (inp.get('shape') != [640, 640, 3] or inp.get('layout') != 'HWC'
            or inp.get('dtype') != 'uint8' or inp.get('color_space') != 'RGB'
            or inp.get('preprocess_mode') != 'letterbox' or inp.get('letterbox_pad_value') != 114):
        raise ValueError('detection_requires_bound_640_rgb_uint8_letterbox_contract')
    stage_kind = str(contract.get('contract_family') or contract.get('endpoint_mode') or '')
    if stage_kind not in {'raw_head', 'raw_detection_head', 'decoded_pre_nms', 'decoded_nms'}:
        raise ValueError('detection_physical_output_stage_not_explicit')
    meta = read_json(run / f'models/{MODEL}/model_manifest.json')
    onnx, onnx_checked = exact_artifact([suite / f'models/{MODEL}.onnx', meta.get('resolved_path'), meta.get('file', {}).get('path')], contract.get('source_onnx_sha256'))
    dxnn, dx_checked = exact_artifact([contract_path.parent / 'model.dxnn', contract.get('suite_artifact_path'), contract.get('artifact_path')], contract.get('suite_artifact_sha256') or contract.get('artifact_sha256'))
    old_request_path = run / f'models/{MODEL}/benchmark_results/quality_inputs/{target["id"]}/results/deepx_m1_full/task_quality_inputs/full_request.json'
    historical = read_json(old_request_path)
    ids = list(historical.get('expected_image_ids') or [])
    if len(ids) < SAMPLE_COUNT or len(ids) != len(set(ids)):
        raise ValueError('detection_original_quality_image_ids_invalid')
    explicit = getattr(args, 'detection_image_ids', None)
    wanted = list(explicit) if explicit is not None else ids[:32]
    if any(image_id not in ids for image_id in wanted):
        raise ValueError('detection_fixed_image_id_not_in_original_request')
    validation_source = Path(str(runrow.get('validation_images') or 'validation'))
    if not validation_source.is_absolute():
        validation_source = suite / validation_source
    manifest_paths = [validation_source if validation_source.suffix == '.json' else validation_source / 'manifest.json',
                      run / 'campaign/inputs/dataset_detection_validation.json', runrow.get('validation_manifest'),
                      plan.get('campaign', {}).get('dataset_manifests', {}).get('detection', {}).get('validation')]
    candidates, manifest = select_samples(manifest_paths, wanted, require_labels=False)
    selected = select_four_samples(candidates, explicit)
    stage = output / '_stage'
    staged_suite = stage / 'suite'
    staged_suite.mkdir(parents=True)
    provenance = copy_source_snapshot(suite, args.tool_dir, staged_suite)
    for source, relative in EXTRA_PRODUCT_SOURCES:
        src, dest = args.tool_dir / source, staged_suite / relative
        if not src.is_file():
            raise FileNotFoundError('detection_product_source_missing:' + str(src))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        provenance.append({'relative_path': relative, 'origin': 'installed_tool', 'source_path': str(src), 'sha256': digest(dest)})
    for source, relative in ((contract_path, 'deepx/deepx_m1/full/output_contract.json'),
                             (suite / 'output_contracts.json', 'output_contracts.json'),
                             (dxnn, 'deepx/deepx_m1/full/model.dxnn')):
        dest = staged_suite / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    for sample in selected:
        sample['image'] = 'images/' + sample['image_id']
        destination = staged_suite / sample['image']
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(sample['path'], destination)
        retained = output / 'selected_inputs' / sample['image']
        retained.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination, retained)
    # No annotations are copied into this four-image parity observation; the
    # original policy and full request are retained, and no COCO AP is assessed.
    write_json(staged_suite / 'images/manifest.json', {'samples': [{'image': s['image_id'], 'annotations': []} for s in selected]})
    runrow.update(model_id=MODEL, benchmark_task='detection', validation_images='images', validation_max_images=SAMPLE_COUNT,
                  dxnn_path='deepx/deepx_m1/full/model.dxnn')
    request = {'schema': SCHEMA, 'task': 'detection', 'stage': 'detection-control', 'nonce': uuid.uuid4().hex,
               'model_id': MODEL, 'setup_id': target['id'], 'runtime_venv': remote['runtime_venv'],
               'samples': selected, 'hardware_samples': SAMPLE_COUNT, 'input_hw': [640, 640],
               'original_full_run': runrow, 'expected_dxnn_sha256': digest(dxnn), 'expected_onnx_sha256': digest(onnx),
               'selection_frozen_before_inference': True, 'selection_policy': 'explicit_four_ids' if explicit else 'first_four_distinct_aspect_ratios_in_first_32_original_quality_ids',
               'declared_physical_output_stage': stage_kind, 'diagnostic_only': True, 'counts_as_benchmark': False,
               'coco_ap_evaluated': False, 'quality_guardrails_modified': False}
    shutil.copytree(BUNDLE / 'lib', stage / 'lib', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('deepx_full_workflow_smoke_worker_v27930.py', 'deepx_full_output_probe_worker_v27930.py'):
        shutil.copyfile(args.tool_dir / 'scripts' / name, stage / 'lib' / name)
    write_json(stage / 'request.json', request)
    write_json(output / 'evidence_request.json', request)
    write_json(output / 'resolution.json', {'source_contract': contract, 'source_provenance': provenance,
               'onnx_candidates': onnx_checked, 'dxnn_candidates': dx_checked, 'validation_manifest': manifest})
    shutil.copyfile(old_request_path, output / 'original_full_quality_request.json')
    for row in provenance:
        dest = output / 'product_sources' / row['relative_path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged_suite / row['relative_path'], dest)
    return stage, request


def run_detection(request, stage, out):
    import numpy as np
    import dx_engine
    from PIL import Image
    suite = Path(stage) / 'suite'
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if request.get('model_id') != MODEL or request.get('selection_frozen_before_inference') is not True:
        raise ValueError('detection_request_not_fixed_yolo26s')
    samples = list(request.get('samples') or [])
    if len(samples) != SAMPLE_COUNT or len({s['image_id'] for s in samples}) != SAMPLE_COUNT:
        raise ValueError('detection_requires_exact_four_samples')
    bench = isolated_product(suite)
    from splitpoint_runners import native_full_input as nfi
    from splitpoint_runners.native_detection_postprocess import DetectionCompletionRuntime, build_detection_completion_execution_contract
    from splitpoint_runners.native_output_endpoint import load_authoritative_output_contract, runtime_output_contract
    model = suite / 'deepx/deepx_m1/full/model.dxnn'
    if digest(model) != request['expected_dxnn_sha256']:
        raise ValueError('detection_staged_dxnn_digest_mismatch')
    runrow = dict(request['original_full_run'])
    _, contract = bench._deepx_input_size_from_contract(suite, runrow, default=640)
    if contract.get('endpoint_contract_binding_status') != 'attested':
        raise ValueError('detection_runtime_endpoint_not_attested')
    declared = load_authoritative_output_contract(suite, backend='deepx_m1', model_id=MODEL, variant='full', task='detection')
    names = [str(item.get('name') or '') for item in contract.get('outputs', [])]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError('detection_output_names_not_unique')
    prepared = []
    for index, sample in enumerate(samples):
        path = suite / sample['image']
        if digest(path) != sample['sha256']:
            raise ValueError('detection_image_digest_mismatch')
        with Image.open(path) as image:
            if list(image.size) != sample['original_wh']:
                raise ValueError('detection_original_geometry_mismatch')
        sealed = nfi.prepare_and_seal_deepx_native_full_input(image_path=path, input_contract=contract, task='detection',
                     out_dir=out / f'sealed_{index:03d}', model=MODEL, setup_id=request['setup_id'], comparison_backend='deepx')
        loaded = nfi.load_sealed_deepx_native_full_input(sealed['manifest_path'], image_path=path, input_contract=contract,
                     task='detection', expected_model=MODEL, expected_setup_id=request['setup_id'], expected_comparison_backend='deepx')
        prepared.append(np.asarray(loaded['runtime_input']).copy())
    summary = {'schema': SCHEMA, 'task': 'detection', 'stage': 'S6', 'model_id': MODEL, 'request_nonce': request['nonce'],
               'status': 'started', 'collection_status': 'pending', 'pipeline_consistency': 'not_evaluated',
               'hardware_executed': False, 'counts_as_benchmark': False, 'diagnostic_only': True, 'compiler_invoked': False,
               'quality_acceptance': 'NOT_EVALUATED', 'coco_ap_evaluated': False, 'quality_guardrails_modified': False,
               'rows': [], 'errors': [], 'inference_budget': SAMPLE_COUNT, 'quality_record_scope': 'actual_production_semantic_detections_json'}
    real_engine = dx_engine.InferenceEngine
    class CaptureEngine:
        def __init__(self, path, *args, **kwargs):
            if Path(path).resolve() != model.resolve():
                raise ValueError('detection_unexpected_engine_path')
            self.engine = real_engine(path, *args, **kwargs)
        def run(self, feeds):
            index = len(summary['rows'])
            if index >= SAMPLE_COUNT:
                raise ValueError('detection_inference_budget_exceeded')
            sample = samples[index]
            row = {'ordinal': index, 'image_id': sample['image_id'], 'original_wh': sample['original_wh'], 'status': 'attempted'}
            summary['rows'].append(row)
            try:
                if len(feeds) != 1:
                    raise ValueError('detection_feed_count_invalid')
                feed = np.asarray(feeds[0]).copy()
                row['native_vs_quality_feed'] = compare(prepared[index], feed)
                if not row['native_vs_quality_feed'].get('exact_equal') or not row['native_vs_quality_feed'].get('dtype_equal'):
                    raise ValueError('detection_native_quality_input_mismatch')
                raw = self.engine.run(feeds)
                summary['hardware_executed'] = True
                arrays = [np.asarray(item).copy() for item in (raw if isinstance(raw, (list, tuple)) else [raw])]
                save_tensors(out / f'raw_{index:03d}.npz', feed=feed, **{f'output_{i:02d}': item for i, item in enumerate(arrays)})
                if len(arrays) != len(names):
                    raise ValueError('detection_runtime_output_count_mismatch')
                row['raw_outputs'] = [{'name': name, **stats(value)} for name, value in zip(names, arrays)]
                if any(not np.issubdtype(value.dtype, np.floating) for value in arrays):
                    raise ValueError('detection_runtime_output_requires_declared_float_or_explicit_dequantization')
                if not all(np.isfinite(value).all() for value in arrays):
                    raise ValueError('detection_runtime_output_nonfinite')
                named = dict(zip(names, arrays))
                source = runtime_output_contract('detection', named, raw_fallback=True, declared_contract=declared)
                execution = build_detection_completion_execution_contract(model_id=MODEL, outputs=named,
                            input_hw=request['input_hw'], original_wh=sample['original_wh'], source_endpoint_contract=source,
                            preprocess=contract['input'])
                runtime = DetectionCompletionRuntime(execution, observation_relation='independent_replay')
                result = runtime.process(named)
                row.update(status='observed', physical_output_stage=source['stage'], host_completion_mode=execution['completion_mode'],
                           host_nms_applied=result['host_nms_applied'], host_completion_count=runtime.completed_count,
                           host_detections=list(runtime.last_detections), completion_result=result,
                           input_modified_by_engine=not np.array_equal(feed, feeds[0]),
                           raw_outputs_unchanged_by_host=all(stats(value)['sha256_bytes'] == observed['sha256_bytes'] for value, observed in zip(arrays, row['raw_outputs'])))
                write_json(out / f'completion_{index:03d}.json', {'execution_contract': execution, 'result': result})
                return raw
            except Exception as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                write_json(out / 'remote_result.json', summary)
        def __getattr__(self, name):
            return getattr(self.engine, name)
    quality_dir = suite / 'results/s6_quality'
    quality_dir.mkdir(parents=True, exist_ok=True)
    try:
        dx_engine.InferenceEngine = CaptureEngine
        semantic = bench._run_deepx_semantic_validation(suite, model, runrow,
                    SimpleNamespace(validation_images='', benchmark_task='detection', validation_max_images=SAMPLE_COUNT), quality_dir)
        summary['product_quality_semantic'] = semantic
        records_path = quality_dir / 'detections.json'
        records = read_json(records_path).get('images', []) if records_path.is_file() else []
        if records_path.is_file():
            shutil.copyfile(records_path, out / 'product_quality_detections.json')
        by_id = {Path(record['image']).name: record for record in records}
        if len(by_id) != SAMPLE_COUNT or len(records) != SAMPLE_COUNT:
            summary['errors'].append('detection_quality_record_set_incomplete')
        for row in summary['rows']:
            record = by_id.get(row['image_id'])
            if record is None:
                row['quality_record_missing'] = True
                continue
            decoder = record.get('decoder_contract') or {}
            audit = record.get('preprocessing_audit') or {}
            row['quality_preprocessing_audit'] = audit
            row['quality_decoder_contract'] = decoder
            row['quality_detections'] = record.get('detections')
            row['same_raw_record_parity'] = row.get('host_detections') == record.get('detections')
            expected_nms = row.get('physical_output_stage') in {'raw_head', 'decoded_pre_nms'}
            row['exact_once_completion'] = (row.get('host_completion_count') == 1
                         and row.get('host_nms_applied') is expected_nms and decoder.get('host_nms_applied') is expected_nms)
            row['original_geometry_binding'] = list(audit.get('source_shape_hw') or []) == list(reversed(row['original_wh']))
            row['classes_valid'] = all(isinstance(d.get('class_id'), int) and 0 <= d['class_id'] < 80 for d in row.get('host_detections', []))
            if not all(row.get(key) is True for key in ('same_raw_record_parity', 'exact_once_completion', 'original_geometry_binding', 'classes_valid', 'raw_outputs_unchanged_by_host')) or row.get('input_modified_by_engine') is not False:
                summary['errors'].append('detection_pipeline_mismatch:' + row['image_id'])
        if semantic.get('status') != 'ok' or semantic.get('error_count') != 0 or semantic.get('validated_image_count') != SAMPLE_COUNT:
            summary['errors'].append('detection_product_quality_semantic_incomplete')
    except Exception as exc:
        summary['errors'].append(f'{type(exc).__name__}: {exc}')
        summary['traceback'] = traceback.format_exc()
    finally:
        dx_engine.InferenceEngine = real_engine
        gc.collect()
    complete = len(summary['rows']) == SAMPLE_COUNT and all(row.get('status') == 'observed' for row in summary['rows'])
    summary['collection_status'] = 'complete' if complete else 'incomplete'
    summary['pipeline_consistency'] = 'consistent_same_raw_output_scope' if complete and not summary['errors'] else 'failed_or_incomplete'
    summary['status'] = 'complete' if complete and not summary['errors'] else 'partial'
    summary['root_cause_status'] = 'quality_cause_not_assessed_by_four_image_parity'
    write_json(out / 'remote_result.json', summary)
    return summary


def collect_detection(args):
    from collect_smokes import archive_output, run_remote, select_target, workflow_gate
    args.run_dir = args.run_dir.expanduser().resolve()
    args.tool_dir = args.tool_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.is_relative_to(args.run_dir) or args.output_dir.is_relative_to(args.tool_dir):
        print('diagnostic output must be outside the original run and installed source', file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='deepx_detection_s6_v27931_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_', dir=args.output_dir))
    summary = {'schema': SCHEMA, 'stage': 'S6', 'model_id': MODEL, 'status': 'started', 'hardware_executed': False,
               'collection_status': 'pending', 'pipeline_consistency': 'not_evaluated', 'compiler_invoked': False,
               'counts_as_benchmark': False, 'quality_acceptance': 'NOT_EVALUATED', 'coco_ap_evaluated': False,
               'quality_guardrails_modified': False, 'run_dir': str(args.run_dir), 'output_dir': str(output)}
    try:
        with workflow_gate(Path.home() / '.onnx_splitpoint_tool/locks/workflow_platform_interlock.lock'):
            target, remote = select_target(args.run_dir)
            stage, request = prepare_detection_model(args.run_dir, target, remote, args, output)
            if args.plan_only or args.offline_only:
                summary.update(status='planned', collection_status='complete', hardware_executed=False,
                               pipeline_consistency='not_evaluated', root_cause_status='hardware_pending')
            else:
                for executable in ('ssh', 'scp'):
                    if not shutil.which(executable):
                        raise ValueError('required_executable_missing:' + executable)
                transport = run_remote(stage, remote, output)
                summary['transport'] = transport
                rp = output / 'remote/remote_result.json'
                result = read_json(rp) if rp.is_file() else {}
                summary['hardware_executed'] = result.get('hardware_executed') is True
                summary['collection_status'] = result.get('collection_status', 'incomplete')
                summary['pipeline_consistency'] = result.get('pipeline_consistency', 'not_evaluated')
                valid = (transport.get('request_binding_verified') is True and transport.get('cleanup_confirmed') is True
                         and result.get('request_nonce') == request['nonce'] and result.get('status') == 'complete')
                summary['status'] = 'evidence_collected' if valid else 'partial_evidence'
    except Exception as exc:
        summary.update(status='failed', collection_status='incomplete', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        stage = output / '_stage'
        if stage.is_dir():
            shutil.rmtree(stage)
        write_json(output / 'collection_summary.json', summary)
        archive = archive_output(output)
        print('EVIDENCE_DIRECTORY=' + str(output), flush=True)
        print('S6_STATUS=' + summary['status'], flush=True)
        print('QUALITY_ACCEPTANCE=NOT_EVALUATED; COCO_AP_NOT_EVALUATED', flush=True)
        print('DIAGNOSTIC_ZIP=' + str(archive), flush=True)
    return 0 if summary['status'] in {'planned', 'evidence_collected'} else 2
