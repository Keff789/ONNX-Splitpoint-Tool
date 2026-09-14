"""Shipped profile policy and retirement of the former uncached Force starter."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIRECTORIES = (
    ROOT / 'profiles',
    ROOT / 'onnx_splitpoint_tool/resources/evaluation_profiles',
)
FORCE_KEYS = {
    'force_build', 'force_rebuild_engines',
    'native_force_rebuild_engines', 'force_rebuild_native_engines',
}


def _force_settings(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            name = f'{prefix}.{key}' if prefix else str(key)
            if key in FORCE_KEYS:
                yield name, item
            yield from _force_settings(item, name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _force_settings(item, f'{prefix}[{index}]')


def test_all_shipped_profiles_and_bound_snapshots_have_force_off():
    found = 0
    for directory in PROFILE_DIRECTORIES:
        for path in sorted(directory.glob('*.yaml')):
            settings = yaml.safe_load(path.read_text(encoding='utf-8'))
            for field, value in _force_settings(settings):
                found += 1
                assert value is False, f'{path.relative_to(ROOT)}:{field}={value!r}'
    assert found > 0, 'No shipped Force settings were inspected'


@pytest.mark.parametrize('relative', [
    'profiles/hailo10h_native_bringup.yaml',
    'profiles/native_hailo10h_bringup.yaml',
    'profiles/native_resnet_yolo26s_hailo10h_bringup_v1.yaml',
    'profiles/resnet_yolo26s_hailo10h_native.yaml',
    'onnx_splitpoint_tool/resources/evaluation_profiles/native_resnet_yolo26s_hailo10h_bringup_v1.yaml',
    'onnx_splitpoint_tool/resources/evaluation_profiles/native_resnet_yolo26s_hailo8_smoke_v1.yaml',
])
def test_native_bringup_profile_loads_without_force(relative):
    profile = load_evaluation_profile(ROOT / relative, validate=True)
    assert profile is not None and not isinstance(profile, tuple)
    force = list(_force_settings(profile.raw_profile))
    assert any(name.endswith('force_rebuild_engines') for name, _ in force)
    assert all(value is False for _, value in force)


def test_existing_canary_enables_reuse_without_changing_build_recipe():
    path = ROOT / 'profiles/resnet50_v2772_hailo_parallel_build_canary.yaml'
    profile = load_evaluation_profile(path, validate=True)
    assert profile is not None and not isinstance(profile, tuple)
    hailo = profile.raw_profile['hailo_build']
    store = profile.raw_profile['artifact_store']
    assert hailo['force_build'] is False
    assert hailo['mode'] == 'reuse_and_build_missing'
    assert hailo['cache_enabled'] is True
    assert store['enabled'] is True
    assert store['register_hailo'] is True
    assert hailo['targets'] == ['hailo8', 'hailo10']
    assert (hailo['build_full'], hailo['build_part1'], hailo['build_part2']) == (False, True, False)
    assert (hailo['preset'], hailo['optimization_level'], hailo['calib_count'], hailo['calib_batch_size']) == ('smoke', 0, 8, 8)


def test_retired_force_starter_performs_no_build_or_cache_writes(tmp_path):
    marker = tmp_path / 'compiler_called'
    fake_python = tmp_path / 'fake_python'
    fake_python.write_text('#!/bin/sh\n: > "$FORCE_TEST_MARKER"\nexit 97\n', encoding='utf-8')
    fake_python.chmod(0o700)
    env = dict(os.environ, PYTHON_BIN=str(fake_python), FORCE_TEST_MARKER=str(marker),
               CANARY_BASE_ROOT=str(tmp_path / 'runs'), MODELS_ROOT=str(tmp_path / 'models'))
    script = ROOT / 'scripts/run_v2772_hailo_parallel_build_canary.sh'
    completed = subprocess.run(['bash', str(script)], env=env, cwd=tmp_path,
                               capture_output=True, text=True, timeout=5)
    assert completed.returncode == 2
    assert 'CANARY_RESULT=NOT_RUN' in completed.stdout
    assert 'legacy_force_canary_retired' in completed.stdout
    assert 'CANARY_RESULT=PASS' not in completed.stdout
    assert not marker.exists()
    assert not (tmp_path / 'runs').exists()
    assert not (tmp_path / 'models').exists()
