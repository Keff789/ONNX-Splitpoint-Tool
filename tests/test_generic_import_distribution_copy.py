"""Only a known obsolete transport output is omitted from Generic import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from onnx_splitpoint_tool.management_reference import _source_contract
from onnx_splitpoint_tool.remote.bundle import build_suite_bundle, remote_minimal_bundle_patterns
from onnx_splitpoint_tool.gui.controller import _copy_runner_lib
from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime
from test_generic_quality_suite_import import _fixture, _import, _write


def _inputs(tmp_path):
    data = _fixture(tmp_path)
    source = data["source"]
    _copy_runner_lib(source)
    for case in ("b001", "b002"):
        write_runner_skeleton_onnxruntime(str(source / case))
    files = {
        "dist/suite_bundle.tar.gz": b"old transport archive; not a scientific input",
        "dist/suite_bundle.tar.gz.manifest.json": b'{"old_transport_manifest":true}',
        "dist/suite_bundle.tar.gz.lock": b"preserved historical transport metadata",
        "dist/suite_bundle.tar.gz.other": b"preserved neighbouring file",
        "suite_bundle.tar.gz": b"preserved different root address",
        "b001/dist/suite_bundle.tar.gz": b"preserved different nested address",
        "models/model.onnx": b"unchanged canonical source ONNX",
        "models/model.onnx.data": b"unchanged external weights",
        "resources/validation/frozen/manifest.json": b'{"frozen_validation":true}',
        "resources/validation/frozen/image.jpg": b"unchanged validation image",
        "quality_management/canonical_cpu_reference.json": b'{"reference_role":"canonical_cpu_ort","records":[]}',
        "quality_management/status.json": b'{"source_contract_sha256":"unchanged historical binding"}',
    }
    for index, case in enumerate(data["catalogue"], 1):
        name = case["case_id"]
        _write(source / name / "split_manifest.json", {
            "boundary": index, "full_model": "../models/model.onnx",
            "part1_model": "part1.onnx", "part2_model": "part2.onnx"})
        for relative in ("part1.onnx", "part2.onnx", "hailo/hailo8/compiled.hef",
                         "deepx/model.dxnn", "native_trt/part2.engine"):
            files[f"{name}/{relative}"] = f"unchanged {name}/{relative}".encode()
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for run in data["profile"]["run_profiles"]:
        run.update(validation_images="resources/validation/frozen", validation_max_images=2)
    return data


def _snapshot(source):
    return {path.relative_to(source).as_posix(): path.read_bytes()
            for path in source.rglob("*") if path.is_file()}


def test_exact_generic_import_omits_old_archive_without_reading_it_or_changing_inputs(tmp_path, monkeypatch):
    data = _inputs(tmp_path)
    before = _snapshot(data["source"])
    old_archive = data["source"] / "dist/suite_bundle.tar.gz"
    original = shutil.copyfile

    def copyfile(source, destination, **kwargs):
        assert Path(source) != old_archive, "obsolete transport payload was copied"
        return original(source, destination, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", copyfile)
    result = _import(data)
    assert not (result.suite_dir / "dist/suite_bundle.tar.gz").exists()
    # The established importer still refreshes the three ordinary plan/harness
    # files. Every other original input, including all twenty cases, survives.
    rewritten = {"benchmark_plan.json", "benchmark_set.json", "benchmark_suite.py"}
    for relative, content in before.items():
        if relative not in rewritten | {"dist/suite_bundle.tar.gz"}:
            assert (result.suite_dir / relative).read_bytes() == content, relative
    assert _snapshot(data["source"]) == before
    assert len(json.loads((result.suite_dir / "benchmark_set.json").read_text())["cases"]) == 20


def test_normal_bundle_rebuilds_absent_archive_without_changing_cpu_source_identity(tmp_path):
    data = _inputs(tmp_path)
    result = _import(data)
    suite = result.suite_dir
    archive = suite / "dist/suite_bundle.tar.gz"
    assert not archive.exists()
    assert archive.with_name(archive.name + ".manifest.json").exists()
    plan = json.loads((suite / "benchmark_plan.json").read_text())
    contract = json.loads((suite / "benchmark_set.json").read_text())
    source_identity = _source_contract(suite, plan, contract)
    reference = suite / "quality_management/canonical_cpu_reference.json"
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    includes, excludes = remote_minimal_bundle_patterns()
    stats = build_suite_bundle(suite, archive, includes=includes, excludes=excludes)
    assert archive.is_file() and stats.reused is False
    with tarfile.open(archive) as bundle:
        assert "dist/suite_bundle.tar.gz" not in bundle.getnames()
        assert bundle.extractfile("models/model.onnx").read() == (suite / "models/model.onnx").read_bytes()
        assert bundle.extractfile("b020/part1.onnx").read() == (suite / "b020/part1.onnx").read_bytes()
    assert _source_contract(suite, plan, contract) == source_identity
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == reference_hash


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_unknown_node_at_transport_address_is_preserved(tmp_path, kind):
    data = _inputs(tmp_path)
    archive = data["source"] / "dist/suite_bundle.tar.gz"
    archive.unlink()
    if kind == "directory":
        archive.mkdir()
        (archive / "scientific_input.bin").write_bytes(b"preserve unexpected directory input")
    else:
        archive.symlink_to("suite_bundle.tar.gz.other")
    result = _import(data)
    target = result.suite_dir / "dist/suite_bundle.tar.gz"
    if kind == "directory":
        assert (target / "scientific_input.bin").read_bytes() == b"preserve unexpected directory input"
    else:
        assert target.read_bytes() == (archive.parent / "suite_bundle.tar.gz.other").read_bytes()
        assert archive.is_symlink()


@pytest.mark.parametrize("kind", ["legacy", "mixed_legacy"])
def test_non_generic_import_keeps_existing_transport_copy(tmp_path, kind):
    data = _inputs(tmp_path)
    profile = {} if kind == "legacy" else {"run_profiles": [
        {"id": "ort_cpu", "type": "onnxruntime", "provider": "cpu"},
        {"id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt"}]}
    before = (data["source"] / "dist/suite_bundle.tar.gz").read_bytes()
    result = _import(data, profile_payload=profile)
    assert (result.suite_dir / "dist/suite_bundle.tar.gz").read_bytes() == before
    assert (data["source"] / "dist/suite_bundle.tar.gz").read_bytes() == before


def test_mixed_generic_runtime_stays_rejected_before_copy(tmp_path):
    data = _inputs(tmp_path)
    data["profile"]["run_profiles"].append({"id": "native", "type": "onnxruntime", "provider": "cpu"})
    with pytest.raises(ValueError, match="generic_quality_suite_import_runtime_scope_invalid"):
        _import(data)
    assert not (data["run_root"] / "models/model/benchmark_set/legacy_suite").exists()
    assert (data["source"] / "dist/suite_bundle.tar.gz").is_file()


@pytest.mark.parametrize("fault", ["no_central_authority", "duplicate_id", "unknown_stage"])
def test_unresolved_generic_authority_fails_before_copy(tmp_path, fault):
    data = _inputs(tmp_path)
    if fault == "no_central_authority":
        data["profile"]["quality_gate"]["statistics"]["execution_location"] = "remote"
    elif fault == "duplicate_id":
        data["profile"]["run_profiles"].append(dict(data["profile"]["run_profiles"][0]))
    else:
        data["profile"]["run_profiles"][0]["stage2"] = "unknown_backend"
    with pytest.raises(ValueError):
        _import(data)
    assert not (data["run_root"] / "models/model/benchmark_set/legacy_suite").exists()
    assert (data["source"] / "dist/suite_bundle.tar.gz").is_file()
