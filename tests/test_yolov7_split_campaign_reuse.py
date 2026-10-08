"""The normal campaign entry reuses bound bytes and admits one process interval."""
from pathlib import Path

import pytest

from test_yolo_concurrent_campaign_runtime import campaign_config, load_canary


@pytest.mark.parametrize('repetitions', [1, 3])
def test_regular_count_contract_accepts_one_or_three_intervals(monkeypatch, tmp_path, repetitions):
    canary = load_canary(monkeypatch)
    config = campaign_config(tmp_path / 'inputs', canary)
    config['repetitions'] = repetitions
    assert canary._campaign_configuration(config)['repetitions'] == repetitions
    config['frames'] = 100
    with pytest.raises(RuntimeError, match='performance_budget'):
        canary._campaign_configuration(config)


@pytest.mark.parametrize('damage', ['', 'sha256', 'size_bytes', 'missing'])
def test_no_build_reuses_only_verified_library(monkeypatch, tmp_path, damage):
    canary = load_canary(monkeypatch)
    config = campaign_config(tmp_path / 'inputs', canary)
    config.update(repetitions=1, build=False)
    library = tmp_path / 'existing.so'
    library.write_bytes(b'existing library bytes; no SDK or model build')
    config['native_executable'] = dict(path=str(library), sha256=canary.sha256_file(library), size_bytes=library.stat().st_size)
    config['expected_library_sha256'] = config['native_executable']['sha256']
    if damage == 'sha256':
        config['native_executable']['sha256'] = '0' * 64
    elif damage == 'size_bytes':
        config['native_executable']['size_bytes'] += 1
    elif damage == 'missing':
        del config['native_executable']
    monkeypatch.setattr(canary, 'build_cpp', lambda *_: pytest.fail('unexpected build'))
    reached = []
    def checked_library(path):
        reached.append(Path(path))
        raise RuntimeError('verified_existing_library_reached')
    monkeypatch.setattr(canary, 'NativeLibrary', checked_library)
    with pytest.raises(RuntimeError, match='verified_existing_library_reached' if not damage else 'artifact_|no_build'):
        canary.run_campaign(config, out_root=tmp_path / 'output')
    assert reached == ([library] if not damage else [])
