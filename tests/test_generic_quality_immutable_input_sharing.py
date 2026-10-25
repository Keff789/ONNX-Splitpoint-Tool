"""Local immutable input reuse over fresh OWN fixtures; no campaign data."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import shutil

import pytest

from onnx_splitpoint_tool.workflow import generator_binding
from test_generic_quality_selected_local_import import normal_import, prepare, selection
from test_generic_quality_selected_transport import write


def test_cross_device_links_fall_back_to_private_input_copies(tmp_path, monkeypatch):
    state = prepare(tmp_path)
    image = state['dataset'] / 'images/own.JPEG'
    write(image, b'OWN JPEG input')
    model = state['suite'] / 'models/model.onnx'
    model.chmod(0o640)
    os.utime(model, ns=(1700000000000000000, 1700000000123456789))
    before = {p.relative_to(state['suite']): p.read_bytes()
              for p in state['suite'].rglob('*') if p.is_file()}
    attempts = []

    def cross_device(source, destination, **kwargs):
        attempts.append(Path(source).relative_to(state['suite']).as_posix())
        raise OSError(errno.EXDEV, 'OWN different filesystem')

    monkeypatch.setattr(generator_binding.os, 'link', cross_device)
    result = normal_import(state)
    assert set(attempts) == {'models/model.onnx', 'b001/part1.onnx', 'b001/part2.onnx',
                             image.relative_to(state['suite']).as_posix()}
    for relative in attempts:
        source_stat = (state['suite'] / relative).stat()
        copied_stat = (result.suite_dir / relative).stat()
        assert copied_stat.st_mode == source_stat.st_mode
        assert copied_stat.st_mtime_ns == source_stat.st_mtime_ns
        assert (result.suite_dir / relative).read_bytes() == before[Path(relative)]
        assert not (result.suite_dir / relative).samefile(state['suite'] / relative)
    sharing = result.metrics['local_immutable_input_sharing']
    assert sharing == {'enabled': True, 'hardlinked_files': 0, 'hardlinked_bytes': 0,
                       'fallback_copied_files': 4,
                       'fallback_copied_bytes': sum(len(before[Path(p)]) for p in attempts)}
    assert {p.relative_to(state['suite']): p.read_bytes()
            for p in state['suite'].rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('kind', ['file', 'parent'])
def test_validation_symlinks_are_copied_without_sharing(tmp_path, kind):
    state = prepare(tmp_path)
    external = tmp_path / 'OWN_external_images'
    external.mkdir()
    original = external / 'own.jpg'
    write(original, b'OWN external image')
    images = state['dataset'] / 'images'
    if kind == 'parent':
        images.symlink_to(external, target_is_directory=True)
    else:
        images.mkdir()
        (images / 'own.jpg').symlink_to(original)
    assert selection(state)
    result = normal_import(state)
    copied = result.suite_dir / (images / 'own.jpg').relative_to(state['suite'])
    assert not copied.is_symlink()
    assert copied.read_bytes() == original.read_bytes() == b'OWN external image'
    assert not copied.samefile(original)
    assert result.metrics['local_immutable_input_sharing']['hardlinked_files'] == 3


def test_unproved_model_symlink_keeps_complete_private_copy(tmp_path):
    state = prepare(tmp_path)
    model = state['suite'] / 'models/model.onnx'
    original = tmp_path / 'OWN_external_model.onnx'
    original.write_bytes(model.read_bytes())
    model.unlink()
    model.symlink_to(original)
    assert selection(state) == ()
    result = normal_import(state)
    assert result.metrics['local_immutable_input_sharing']['enabled'] is False
    copied = result.suite_dir / 'models/model.onnx'
    assert not copied.is_symlink() and not copied.samefile(original)
    assert copied.read_bytes() == original.read_bytes()
    assert (result.suite_dir / 'b020/part1.onnx').is_file()


def test_mutable_metadata_and_noninput_binaries_remain_private(tmp_path):
    state = prepare(tmp_path)
    paths = ['models/OWN_metadata.json', 'resources/validation/OWN_manifest.json',
             'resources/validation/OWN_annotations.json', 'resources/validation/OWN_helper.py',
             'resources/validation/OWN_payload.bin', 'b001/OWN_weights.bin']
    for relative in paths:
        write(state['suite'] / relative, b'OWN private bytes')
    result = normal_import(state)
    for relative in paths:
        source, copied = state['suite'] / relative, result.suite_dir / relative
        assert copied.read_bytes() == source.read_bytes()
        assert not copied.samefile(source)
    metadata = result.suite_dir / paths[0]
    metadata.write_bytes(b'OWN current metadata replacement')
    assert (state['suite'] / paths[0]).read_bytes() == b'OWN private bytes'
    inputs = json.loads(result.artifacts['generator_input_json'].read_text())
    assert inputs['local_runtime_binary_excludes']
    assert inputs['local_sidecar_publication_targets'] == ['quality_runtime/b001.json']


def test_source_replacement_during_link_is_rejected(tmp_path, monkeypatch):
    state = prepare(tmp_path)
    original_link = generator_binding.os.link
    model = state['suite'] / 'models/model.onnx'
    expected = model.read_bytes()
    linked = []

    def replaced(source, destination, **kwargs):
        source = Path(source)
        if source == model:
            replacement = source.with_suffix('.replacement')
            replacement.write_bytes(expected)
            replacement.replace(source)
            original_link(source, destination, **kwargs)
            linked.append(Path(destination))
        else:
            original_link(source, destination, **kwargs)

    monkeypatch.setattr(generator_binding.os, 'link', replaced)
    with pytest.raises(ValueError, match='generic_quality_suite_import_shared_input_changed'):
        normal_import(state)
    assert len(linked) == 1 and linked[0].exists()
    assert model.read_bytes() == expected


def test_destination_replacement_after_link_is_preserved_on_rejection(tmp_path, monkeypatch):
    state = prepare(tmp_path)
    original_link = generator_binding.os.link
    model = state['suite'] / 'models/model.onnx'
    expected = model.read_bytes()
    replaced = []

    def replace_destination(source, destination, **kwargs):
        original_link(source, destination, **kwargs)
        if Path(source) == model:
            destination = Path(destination)
            destination.unlink()
            destination.write_bytes(b'OWN foreign destination replacement')
            replaced.append(destination)

    monkeypatch.setattr(generator_binding.os, 'link', replace_destination)
    with pytest.raises(ValueError, match='generic_quality_suite_import_shared_input_changed'):
        normal_import(state)
    assert len(replaced) == 1
    assert replaced[0].read_bytes() == b'OWN foreign destination replacement'
    assert model.read_bytes() == expected


def test_existing_destination_symlink_cannot_overwrite_another_donor(tmp_path, monkeypatch):
    state = prepare(tmp_path)
    original_link = generator_binding.os.link
    model = state['suite'] / 'models/model.onnx'
    other_donor = tmp_path / 'OWN_other_donor.onnx'
    other_donor.write_bytes(b'OWN unrelated immutable donor')
    destinations = []

    def conflicting_target(source, destination, **kwargs):
        if Path(source) == model:
            Path(destination).symlink_to(other_donor)
            destinations.append(Path(destination))
            raise FileExistsError(errno.EEXIST, 'OWN concurrently occupied target')
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(generator_binding.os, 'link', conflicting_target)
    with pytest.raises(ValueError, match='generic_quality_suite_import_shared_input_target_conflict'):
        normal_import(state)
    assert len(destinations) == 1 and destinations[0].is_symlink()
    assert other_donor.read_bytes() == b'OWN unrelated immutable donor'


def test_fallback_exclusive_open_rejects_new_destination_symlink(tmp_path, monkeypatch):
    state = prepare(tmp_path)
    original_link, original_open = generator_binding.os.link, generator_binding.os.open
    model = state['suite'] / 'models/model.onnx'
    target = state['run_root'] / 'models/own_model/benchmark_set/legacy_suite/models/model.onnx'
    other_donor = tmp_path / 'OWN_other_donor.onnx'
    other_donor.write_bytes(b'OWN unrelated immutable donor')
    attempted = []

    def cross_device(source, destination, **kwargs):
        if Path(source) == model:
            raise OSError(errno.EXDEV, 'OWN different filesystem')
        original_link(source, destination, **kwargs)

    def replaced_before_open(path, flags, *args, **kwargs):
        if str(path) == str(target) and flags & os.O_EXCL:
            target.symlink_to(other_donor)
            attempted.append(target)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(generator_binding.os, 'link', cross_device)
    monkeypatch.setattr(generator_binding.os, 'open', replaced_before_open)
    with pytest.raises(shutil.Error):
        normal_import(state)
    assert attempted == [target] and target.is_symlink()
    assert other_donor.read_bytes() == b'OWN unrelated immutable donor'
