"""AP0: bounded original projections are required and replay writes separately."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from scripts import run_complete_set_replay_v27931 as replay_cli

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'tests/fixtures/v27931_complete_set'
REQUIRED = ('original_identity_projection.json', 'original_energy_projection.json',
            'original_source_profile.yaml', 'original_resolved_profile.yaml',
            'ROOT_PROJECTION_PROVENANCE.json', 'PROVENANCE.json')


def _copied_fixture(tmp_path, monkeypatch):
    root = tmp_path/'isolated_source'
    folder = root/'tests/fixtures/v27931_complete_set'
    folder.mkdir(parents=True)
    for name in REQUIRED:
        shutil.copyfile(FIXTURE/name, folder/name)
    monkeypatch.setattr(replay_cli, 'ROOT', root)
    return folder


@pytest.mark.parametrize('name', REQUIRED)
def test_T00_1_required_original_fixture_missing_is_error(tmp_path, monkeypatch, name):
    folder = _copied_fixture(tmp_path, monkeypatch)
    (folder/name).unlink()
    output = tmp_path/'output'
    with pytest.raises(ValueError, match='historical_root_fixture_provenance_invalid'):
        replay_cli.replay_energy(None, output)
    assert not output.exists()


@pytest.mark.parametrize('name', REQUIRED[:4])
def test_T00_1_modified_original_projection_or_profile_is_error(tmp_path, monkeypatch, name):
    folder = _copied_fixture(tmp_path, monkeypatch)
    # Whitespace preserves parseability but changes original fixture bytes.
    path = folder/name
    path.write_bytes(path.read_bytes()+b' ')
    output = tmp_path/'output'
    with pytest.raises(ValueError, match='fixture_source_binding_invalid'):
        replay_cli.replay_energy(None, output)
    assert not output.exists()


@pytest.mark.parametrize('mutation', ('synthetic', 'original_hash', 'archive', 'symlink'))
def test_T00_1_invalid_or_foreign_fixture_origin_rejected(tmp_path, monkeypatch, mutation):
    folder = _copied_fixture(tmp_path, monkeypatch)
    path = folder/'ROOT_PROJECTION_PROVENANCE.json'
    data = json.loads(path.read_text())
    if mutation == 'synthetic':
        data['synthetic'] = True
    elif mutation == 'original_hash':
        key = 'reports/native_energy_measurements/native_producer_energy_results.json'
        data['original_files'][key]['sha256'] = '0'*64
    elif mutation == 'archive':
        data['source_archive']['path'] = 'another_run.zip'
    else:
        original = folder/'original_energy_projection.json'
        original.unlink()
        original.symlink_to(FIXTURE/original.name)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='historical_root_fixture_provenance_invalid'):
        replay_cli.replay_energy(None, tmp_path/'output')


def test_T00_1_verified_fixture_scope_retains_original_numbers_and_profiles(tmp_path):
    provenance, payloads = replay_cli._root_fixture_provenance()
    assert provenance['synthetic'] is False
    assert provenance['source_archive']['sha256'] == 'df0b40b33b29f47355f6b705c6b7957da9a114ed4b3bdc88c3b386edd0933a7d'
    assert provenance['source_archive']['size_bytes'] == 44293643
    for name in REQUIRED[2:4]:
        entry = provenance['fixtures'][name]
        assert entry['representation'] == 'byte_exact_original'
        assert entry['sha256'] == provenance['original_files'][entry['source_files'][0]]['sha256']
    before = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
    report = replay_cli.replay_energy(None, tmp_path/'replay')
    assert report['original_inputs_byte_unchanged'] is True
    assert report['fixture_provenance_verification']['status'] == 'verified_shipped_projection_hashes'
    assert 'omitted payloads are not verified' in report['fixture_provenance_verification']['scope']
    assert report['measurement_rows'] == 55 and report['valid_repetitions'] == 165
    assert report['energy_evidence']['scientific_ready'] is False
    assert report['projection']['energy_accounting']['campaign_comparison_released'] is False
    assert {name: hashlib.sha256((FIXTURE/name).read_bytes()).hexdigest() for name in before} == before


def _run_main(monkeypatch, output, input_run=None):
    args = ['complete-set-replay', '--output-dir', str(output)]
    if input_run is not None:
        args.extend(['--input-run', str(input_run)])
    monkeypatch.setattr(sys, 'argv', args)
    return replay_cli.main()


@pytest.mark.parametrize('existing', ('nonempty', 'file', 'symlink'))
def test_T00_1_cli_refuses_existing_output_without_overwrite(tmp_path, monkeypatch, existing):
    output = tmp_path/'output'
    protected = tmp_path/'protected'; protected.mkdir()
    marker = protected/'keep.json'; marker.write_text('original artifact bytes')
    if existing == 'nonempty':
        output.mkdir(); (output/'keep.json').write_text('original artifact bytes')
    elif existing == 'file':
        output.write_text('original artifact bytes')
    else:
        output.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError, match='historical_replay_output_must_be_new_or_empty_directory'):
        _run_main(monkeypatch, output)
    assert marker.read_text() == 'original artifact bytes'
    if existing == 'nonempty':
        assert list(output.iterdir()) == [output/'keep.json']
    elif existing == 'file':
        assert output.read_text() == 'original artifact bytes'


@pytest.mark.parametrize('inside_kind', ('same', 'child', 'symlink_child'))
def test_T00_1_cli_never_writes_into_input_run(tmp_path, monkeypatch, inside_kind):
    original = tmp_path/'original'; original.mkdir()
    marker = original/'original.json'; marker.write_text('unchanged original')
    output = original if inside_kind == 'same' else original/'replay'
    if inside_kind == 'symlink_child':
        alias = tmp_path/'alias'; alias.symlink_to(original, target_is_directory=True)
        output = alias/'replay'
    with pytest.raises(ValueError, match='historical_replay_output_must_be_outside_original_run'):
        _run_main(monkeypatch, output, original)
    assert list(original.iterdir()) == [marker]
    assert marker.read_text() == 'unchanged original'


def test_T00_1_cli_separate_output_preserves_every_input_run_file(tmp_path, monkeypatch):
    # Explicit projection-backed input-run fixture; no synthetic hardware success.
    original = tmp_path/'original'
    reports = original/'reports'; reports.mkdir(parents=True)
    identity = json.loads((FIXTURE/'original_identity_projection.json').read_text())
    (reports/'native_producer_stage.json').write_text(json.dumps({key: value for key, value in identity.items() if key != 'rows'}))
    (reports/'native_producer_summary.json').write_text(json.dumps({'rows': identity['rows']}))
    energy = reports/'native_energy_measurements/native_producer_energy_results.json'
    energy.parent.mkdir(); shutil.copyfile(FIXTURE/'original_energy_projection.json', energy)
    (original/'keep.txt').write_text('historical file outside replay inputs')
    before = {str(path.relative_to(original)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in original.rglob('*') if path.is_file()}
    output = tmp_path/'output'; output.mkdir()
    assert _run_main(monkeypatch, output, original) == 0
    after = {str(path.relative_to(original)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in original.rglob('*') if path.is_file()}
    assert before == after
    assert (output/'complete_set_identity_replay.json').is_file()
    result = json.loads((output/'complete_set_energy_replay.json').read_text())
    assert result['original_inputs_byte_unchanged'] is True
    assert len(result['input_sha256']) == 3
    assert result['measurement_rows'] == 55 and result['valid_repetitions'] == 165
    with pytest.raises(ValueError, match='historical_replay_output_must_be_new_or_empty_directory'):
        _run_main(monkeypatch, output, original)
