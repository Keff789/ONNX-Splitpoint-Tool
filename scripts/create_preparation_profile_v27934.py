#!/usr/bin/env python3
"""Create a separate bounded preparation copy; never overwrite a user profile.

Preview (no files written):
  .venv/bin/python -B scripts/create_preparation_profile_v27934.py --source profiles/CompleteSetDev.yaml
Save a NEW file, then select it in the normal GUI/CLI:
  .venv/bin/python -B scripts/create_preparation_profile_v27934.py --source profiles/CompleteSetDev.yaml --out profiles/CompleteSetDev_preparation_v27934.yaml

The copy stops after the first selected model's build_backend_artifacts stage.
It does not establish a warm full matrix, Quality-FIRST/TRT runtime bindings,
GPU acceptance, final quality, or Native FS energy results. Existing source
profiles with disabled cache/store (including old v2771 canaries) remain
unchanged; cache/store are enabled only in the new preparation copy.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from onnx_splitpoint_tool import run_modes as rm
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile, validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.config_values import parse_config_bool


FORCE_KEYS = {'force_build', 'force_rebuild_engines',
              'native_force_rebuild_engines', 'force_rebuild_native_engines'}
CPU_FAMILIES = {'hailo8': {'device': 'cpu'}, 'hailo10h': {'device': 'cpu'}}
SCOPE_KEYS = ('model_suite', 'selection_policy', 'run_profiles')
STOP_STAGE = 'build_backend_artifacts'


def _force_off(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORCE_KEYS:
                value[key] = False
            else:
                _force_off(item)
    elif isinstance(value, list):
        for item in value:
            _force_off(item)


def _check_preserved_scope(before, after):
    for key in SCOPE_KEYS:
        if before.get(key) != after.get(key):
            raise ValueError(f'preparation_scope_changed:{key}')


def _verify_recipe(profile):
    hailo, deepx, store = (profile[key] for key in ('hailo_build', 'deepx_build', 'artifact_store'))
    expected = {
        'hailo_mode': (hailo.get('mode'), 'reuse_and_build_missing'),
        'deepx_mode': (deepx.get('mode'), 'reuse_and_build_missing'),
        'hailo_force': (hailo.get('force_build'), False),
        'deepx_force': (deepx.get('force_build'), False),
        'hailo_recipe': ((hailo.get('preset'), hailo.get('optimization_level'), hailo.get('calib_count'), hailo.get('calib_batch_size')), ('balanced', 1, 500, 8)),
        'hailo_integrity': (hailo.get('cache_integrity'), 'relaxed'),
        'hailo_compute': (hailo.get('compute_by_family'), CPU_FAMILIES),
        'deepx_recipe': ((deepx.get('calib_count'), deepx.get('calibration_method'), deepx.get('opt_level')), (500, 'ema', 0)),
        'deepx_classification': (deepx.get('classification_preprocessing'), 'imagenet_mean_std'),
        'hailo_cache': (hailo.get('cache_enabled'), True),
        'artifact_store': (store.get('enabled'), True),
        'bounded_stop': ((profile.get('workflow') or {}).get('stop_after'), STOP_STAGE),
        'execution_mode': ((profile.get('workflow') or {}).get('execution_mode'), 'generate_benchmarksets'),
    }
    for field, (actual, wanted) in expected.items():
        if actual != wanted:
            raise ValueError(f'preparation_effective_mismatch:{field}:{actual!r}')
    if rm.profile_build_summary(profile)['native_force_fields']:
        raise ValueError('preparation_effective_native_force_remaining')


def _candidate(source, name):
    result = copy.deepcopy(source)
    result['name'] = name
    result['purpose'] = (
        'v34 Vorbereitung – erstes Modell. Vorhandene passende Artefakte wiederverwenden; '
        'fehlende bauen. Stop nach vollständiger Build-Publikation des ersten ausgewählten '
        'Modells; keine vollständige Matrix-, Quality-FIRST/TRT-, GPU- oder Finalfreigabe.'
    )
    _force_off(result)
    hailo = result.setdefault('hailo_build', {})
    hailo.update(mode='reuse_and_build_missing', force_build=False, preset='balanced',
                 optimization_level=1, calib_count=500, calib_batch_size=8,
                 cache_integrity='relaxed', cache_enabled=True,
                 compute_by_family=copy.deepcopy(CPU_FAMILIES))
    deepx = result.setdefault('deepx_build', {})
    deepx.update(mode='reuse_and_build_missing', force_build=False, calib_count=500,
                 calibration_method='ema', opt_level=0,
                 classification_preprocessing='imagenet_mean_std')
    result.setdefault('artifact_store', {}).update(enabled=True, register_hailo=True, register_deepx=True)
    result.setdefault('workflow', {}).update(execution_mode='generate_benchmarksets',
                                            skip_runtime_benchmarks=True, stop_after=STOP_STAGE)
    preset = result.get('execution_preset')
    if isinstance(preset, dict):
        # The normal loader has already supplied the effective mode. Bind that
        # exact snapshot; do not look up a new default or change final-quality
        # budgets while correcting the separate calibration/build contract.
        mode = preset['snapshot']
        mode['label'] = 'v34 Vorbereitung – erstes Modell'
        mode['description'] = result['purpose']
        mode['recommended_for'] = 'Begrenzter CPU-Build-/Reuse-Test vor der weiteren Artefaktvorbereitung.'
        build = mode.setdefault('build', {})
        build.setdefault('hailo', {}).update(mode='reuse_and_build_missing', force_build=False,
            preset='balanced', optimization_level=1, calibration_items=500,
            calibration_batch_size=8, cache_integrity='relaxed', cache_enabled=True,
            compute_by_family=copy.deepcopy(CPU_FAMILIES))
        build.setdefault('deepx', {}).update(mode='reuse_and_build_missing', force_build=False,
            calibration_items=500, calibration_method='ema', optimization_level=0,
            classification_preprocessing='imagenet_mean_std')
        build.setdefault('artifact_store', {}).update(enabled=True, register_hailo=True, register_deepx=True)
        mode.setdefault('data', {})['calibration_items'] = {'classification': 500, 'detection': 500}
        mode.setdefault('runtime', {}).update(execution_mode='generate_benchmarksets', skip_runtime_benchmarks=True)
        preset['follow_tool_config'] = False
        preset['snapshot_sha256'] = rm._json_hash(mode)
        # Remove only the old provenance mirror, so the newly explicit
        # classification/device choices are reported as belonging to this copy.
        preset.pop('build_provenance', None)
        result, _ = rm.apply_run_mode(result)
    validate_evaluation_profile_payload(result, source='v34_preparation_copy')
    _check_preserved_scope(source, result)
    _verify_recipe(result)
    return result


def prepare(source_path: Path, output_path: Path | None = None):
    source_path = Path(source_path).expanduser().resolve(strict=True)
    if source_path.suffix.lower() not in {'.yaml', '.yml'} or not source_path.is_file():
        raise ValueError('source_must_be_an_existing_local_yaml')
    output = Path(output_path).expanduser().absolute() if output_path is not None else None
    if output is not None:
        if output.exists() or output.is_symlink():
            raise ValueError('output_must_be_a_new_file')
        if not output.parent.is_dir() or output.suffix.lower() not in {'.yaml', '.yml'}:
            raise ValueError('output_requires_existing_parent_and_yaml_extension')
    source_bytes = source_path.read_bytes()
    raw = yaml.safe_load(source_bytes)
    if not isinstance(raw, dict):
        raise ValueError('source_profile_must_be_a_mapping')
    registry = None
    preset = raw.get('execution_preset')
    if isinstance(preset, dict) and parse_config_bool(preset.get('follow_tool_config', True), field='execution_preset.follow_tool_config'):
        registry_path = Path(preset.get('config_path') or rm.default_run_modes_path()).expanduser().resolve()
        # load_run_modes_config can create a missing default. This diagnostic
        # must never do that merely to preview a profile.
        if not registry_path.is_file():
            raise ValueError(f'current_source_registry_missing:{registry_path}')
        registry = registry_path, registry_path.read_bytes()
    loaded = load_evaluation_profile(source_path, validate=True)
    if loaded is None or isinstance(loaded, tuple):
        raise ValueError('source_profile_did_not_resolve')
    effective = loaded.raw_profile
    models = [row for tier in ('primary', 'reserve')
              for row in (effective.get('model_suite') or {}).get(tier, [])]
    ids = [str(row.get('id') or '') for row in models]
    if any(name.lower().replace('-', '_') in {'yolov7_ultralytics', 'yolo7_ultralytics'} for name in ids):
        raise ValueError('wrong_yolov7_ultralytics_mapping:select_the_existing_yolov7_paper_graph_explicitly;no_model_id_or_boundary_was_replaced')
    name = (output.stem if output is not None else source_path.stem + '_preparation_v27934')
    if name == loaded.profile_id:
        name += '_preparation_v27934'
    candidate = _candidate(effective, name)
    warnings = []
    if (effective.get('deepx_build') or {}).get('classification_preprocessing', 'current_scale_only') == 'current_scale_only':
        warnings.append('Classification changes from current_scale_only to imagenet_mean_std in the new copy only; the first matching MeanStd artifact may legitimately require a build. Detection preprocessing is unchanged.')
    for field, enabled in (
        ('hailo_build.cache_enabled', (effective.get('hailo_build') or {}).get('cache_enabled', True)),
        ('artifact_store.enabled', (effective.get('artifact_store') or {}).get('enabled', True)),
    ):
        if enabled is False:
            warnings.append(f'SOURCE_CACHE_DISABLED:{field}; enabled in the new copy only. The original canary/profile still has disabled reuse storage.')
    if any(name.lower() in {'yolov7', 'yolo7'} for name in ids):
        warnings.append('The existing yolov7 ID and graph path were preserved. Verify that this is the intended paper export; the helper cannot infer graph provenance from a name.')
    report = {
        'status': 'PREVIEW', 'source': str(source_path), 'source_profile_id': loaded.profile_id,
        'source_profile_sha256': (loaded.start_snapshot or {}).get('source_profile_sha256', ''),
        'source_values': rm.profile_build_summary(effective),
        'output': str(output) if output is not None else None, 'profile_id': name,
        'label': 'v34 Vorbereitung – erstes Modell',
        'binding': 'bound_snapshot' if candidate.get('execution_preset') else 'explicit_legacy_profile',
        'effective_values': rm.profile_build_summary(candidate), 'model_ids_preserved': ids,
        'selection_policy_preserved': True, 'model_paths_preserved': True,
        'stop_after': STOP_STAGE, 'prepared_model_scope': 'first_selected_model_only',
        'full_planned_matrix_ready': False, 'quality_first_trt_binding_ready': False,
        'hardware_test_run': False, 'final_quality_budgets_changed': False,
        'source_and_registry_unchanged': True, 'warnings': warnings,
    }

    def check_inputs():
        if source_path.read_bytes() != source_bytes:
            raise ValueError('source_changed_during_preparation')
        if registry is not None and registry[0].read_bytes() != registry[1]:
            raise ValueError('registry_changed_during_preparation')

    check_inputs()
    if output is not None:
        data = (
            '# v34 Vorbereitung – erstes Modell; separate copy, original unchanged.\n'
            '# Source profile: ' + json.dumps(str(source_path), ensure_ascii=False) + '\n'
            '# Existing source-profile identity: ' + report['source_profile_sha256'] + '\n'
            + yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True)
        )
        # Publish only a complete, successfully reloaded YAML and never replace
        # a concurrently created destination. No original profile is rewritten.
        fd, temporary = tempfile.mkstemp(prefix='.v27934_preparation_', suffix='.yaml', dir=output.parent)
        temp = Path(temporary)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            reloaded = load_evaluation_profile(temp, validate=True).raw_profile
            _verify_recipe(reloaded)
            _check_preserved_scope(effective, reloaded)
            check_inputs()
            os.link(temp, output)
            report['status'] = 'CREATED'
        finally:
            temp.unlink(missing_ok=True)
    return candidate, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True, help='Existing local profile YAML; never modified.')
    parser.add_argument('--out', type=Path, help='Explicit NEW YAML path; omit for read-only JSON preview.')
    args = parser.parse_args(argv)
    try:
        _, report = prepare(args.source, args.out)
    except Exception as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': f'{type(exc).__name__}:{exc}'}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
