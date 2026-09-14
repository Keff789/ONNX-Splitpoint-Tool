"""Preparation copies use the real loader and preserve their source contract."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool import run_modes as rm
from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v34_preparation_helper', ROOT / 'scripts/create_preparation_profile_v27934.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def _source(tmp_path, *, binding='central'):
    profile = yaml.safe_load((ROOT / 'profiles/resnet50_v2772_hailo_parallel_build_canary.yaml').read_text())
    profile['name'] = 'original_user_profile'
    profile['hailo_build'].update(force_build=True, cache_enabled=False)
    profile['artifact_store']['enabled'] = False
    profile['deepx_build'] = {'classification_preprocessing': 'current_scale_only', 'force_build': True}
    # Preserve graph paths, paper identity and selected non-anchor coverage.
    model = copy.deepcopy(profile['model_suite']['primary'][0])
    model.update(id='regnet_x_1_6gf', onnx='/unchanged/model roots/regnet.onnx')
    profile['model_suite']['primary'].append(model)
    paper = copy.deepcopy(model)
    paper.update(id='yolov7_paper', onnx='/unchanged/model roots/yolov7_paper.onnx', task='detection')
    profile['model_suite']['primary'].append(paper)
    profile['selection_policy']['forced_cases'].update(regnet_x_1_6gf=['b132'], yolov7_paper=['b070', 'b082'])
    profile['native_producers']['force_rebuild_engines'] = True
    registry = tmp_path / 'registry.yaml'
    config = rm.default_run_modes_config()
    config['modes']['final']['data']['calibration_items'] = {'classification': 8, 'detection': 32}
    config['modes']['final']['build']['hailo'].update(force_build=True, cache_enabled=False)
    config['modes']['final']['build']['deepx']['force_build'] = True
    config['modes']['final']['build']['artifact_store']['enabled'] = False
    if binding != 'legacy':
        profile['execution_preset'] = {'id': 'final', 'follow_tool_config': binding == 'central', 'config_path': str(registry)}
        if binding == 'central':
            registry.write_text(yaml.safe_dump(config))
        else:
            profile['execution_preset'].update(snapshot=config['modes']['final'], config_sha256='historical-registry-identity')
    path = tmp_path / 'user.yaml'
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    return path, registry


@pytest.mark.parametrize('binding', ['central', 'snapshot', 'legacy'])
def test_create_copy_preserves_original_models_boundaries_and_quality(tmp_path, binding):
    source, registry = _source(tmp_path, binding=binding)
    before = source.read_bytes()
    registry_before = registry.read_bytes() if registry.exists() else None
    effective = load_evaluation_profile(source).raw_profile
    assert effective['hailo_build']['calib_count'] != 500
    target = tmp_path / 'new_preparation_v27934.yaml'
    candidate, report = helper.prepare(source, target)
    loaded = load_evaluation_profile(target).raw_profile
    assert report['status'] == 'CREATED'
    assert report['label'] == 'v34 Vorbereitung – erstes Modell'
    assert report['prepared_model_scope'] == 'first_selected_model_only'
    assert report['full_planned_matrix_ready'] is False
    assert report['quality_first_trt_binding_ready'] is False
    assert report['hardware_test_run'] is False
    assert report['source_profile_sha256']
    assert source.read_bytes() == before
    assert (registry.read_bytes() if registry.exists() else None) == registry_before
    for key in helper.SCOPE_KEYS:
        assert loaded.get(key) == effective.get(key)
    helper._verify_recipe(loaded)
    assert loaded['deepx_build']['classification_preprocessing'] == 'imagenet_mean_std'
    assert effective['deepx_build']['classification_preprocessing'] == 'current_scale_only'
    assert 'SOURCE_CACHE_DISABLED:' in '\n'.join(report['warnings'])
    assert 'Detection preprocessing is unchanged' in '\n'.join(report['warnings'])
    assert not list(tmp_path.glob('.v27934_preparation_*'))
    if binding != 'legacy':
        preset = loaded['execution_preset']
        assert preset['follow_tool_config'] is False
        assert preset['label'] == 'v34 Vorbereitung – erstes Modell'
        assert preset['snapshot_sha256'] == rm._json_hash(preset['snapshot'])
        assert preset['config_sha256'] == effective['execution_preset']['config_sha256']
        assert preset['snapshot']['quality'] == effective['execution_preset']['snapshot']['quality']
        assert preset['snapshot']['data']['validation_items'] == effective['execution_preset']['snapshot']['data']['validation_items']
        assert preset['snapshot']['quality']['bootstrap_repetitions'] == 5000
        assert preset['snapshot']['ranking'] == effective['execution_preset']['snapshot']['ranking']
    else:
        assert 'execution_preset' not in candidate


def test_preview_writes_nothing_and_defaults_to_new_name(tmp_path):
    source, registry = _source(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    candidate, report = helper.prepare(source)
    assert report['status'] == 'PREVIEW'
    assert report['output'] is None
    assert candidate['name'] == 'user_preparation_v27934'
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize('destination', ['source', 'existing', 'symlink'])
def test_existing_destinations_cannot_be_overwritten(tmp_path, destination):
    source, _ = _source(tmp_path)
    target = source if destination == 'source' else tmp_path / 'occupied.yaml'
    if destination == 'existing':
        target.write_text('already here')
    elif destination == 'symlink':
        target.symlink_to(source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match='output_must_be_a_new_file'):
        helper.prepare(source, target)
    assert source.read_bytes() == before
    if destination == 'existing':
        assert target.read_text() == 'already here'


def test_missing_followed_registry_is_never_created(tmp_path):
    source, registry = _source(tmp_path)
    registry.unlink()
    with pytest.raises(ValueError, match='current_source_registry_missing'):
        helper.prepare(source, tmp_path / 'new.yaml')
    assert not registry.exists()
    assert not (tmp_path / 'new.yaml').exists()


def test_ultralytics_identity_is_rejected_without_silent_paper_substitution(tmp_path):
    source, _ = _source(tmp_path)
    data = yaml.safe_load(source.read_text())
    data['model_suite']['primary'][-1]['id'] = 'yolov7_ultralytics'
    source.write_text(yaml.safe_dump(data))
    before = source.read_bytes()
    with pytest.raises(ValueError, match='wrong_yolov7_ultralytics_mapping'):
        helper.prepare(source, tmp_path / 'new.yaml')
    assert source.read_bytes() == before
    assert not (tmp_path / 'new.yaml').exists()


def test_racing_destination_is_preserved_and_temporary_file_removed(tmp_path, monkeypatch):
    source, _ = _source(tmp_path)
    target = tmp_path / 'new.yaml'
    real_link = helper.os.link
    def race(src, dst):
        Path(dst).write_text('other writer won')
        return real_link(src, dst)
    monkeypatch.setattr(helper.os, 'link', race)
    with pytest.raises(FileExistsError):
        helper.prepare(source, target)
    assert target.read_text() == 'other writer won'
    assert not list(tmp_path.glob('.v27934_preparation_*'))


def test_registry_change_aborts_before_publication(tmp_path, monkeypatch):
    source, registry = _source(tmp_path)
    real_candidate = helper._candidate
    def changing_registry(*args):
        result = real_candidate(*args)
        registry.write_text('external writer changed this')
        return result
    monkeypatch.setattr(helper, '_candidate', changing_registry)
    with pytest.raises(ValueError, match='registry_changed_during_preparation'):
        helper.prepare(source, tmp_path / 'new.yaml')
    assert not (tmp_path / 'new.yaml').exists()
