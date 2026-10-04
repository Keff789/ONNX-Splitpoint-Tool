from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.workflow import debug_pack as packs


def _suite(tmp_path, declarations):
    case = tmp_path / "suite/b064"
    case.mkdir(parents=True)
    for name in ("model_part1_b64.onnx", "model_part1_hailo_identity_b64.onnx",
                 "model_part2_b64.onnx", "model_part2_host_tail_b64.onnx"):
        (case / name).write_bytes(name.encode())
    (case / "split_manifest.json").write_text(json.dumps(declarations))
    return case.parent


@pytest.mark.parametrize("role", ["part1", "part2"])
@pytest.mark.parametrize("alias", ["legacy", "model", "path", "nested"])
def test_manifest_regular_sources_ignore_backend_helpers(tmp_path, role, alias):
    name = f"model_{role}_b64.onnx"
    declaration = {
        "legacy": {role: name}, "model": {role + "_model": name},
        "path": {role + "_path": name}, "nested": {"models": {role: {"path": name}}},
    }[alias]
    suite = _suite(tmp_path, declaration)
    assert remote_run._trt_preflight_split_source(suite, "b064", role=role) == suite / "b064" / name


def test_manifest_missing_declared_source_does_not_fall_back(tmp_path):
    suite = _suite(tmp_path, {"part1": "missing_part1.onnx"})
    with pytest.raises(FileNotFoundError, match="declared source missing"):
        remote_run._trt_preflight_part1_source(suite, "b064")


def test_conflicting_regular_manifest_declarations_are_rejected(tmp_path):
    suite = _suite(tmp_path, {
        "part1": "model_part1_b64.onnx",
        "models": {"part1": {"path": "model_part1_hailo_identity_b64.onnx"}},
    })
    with pytest.raises(RuntimeError, match="declarations conflict"):
        remote_run._trt_preflight_part1_source(suite, "b064")


def test_matching_regular_manifest_aliases_are_accepted(tmp_path):
    suite = _suite(tmp_path, {
        "part1": "model_part1_b64.onnx",
        "models": {"part1": {"path": "./model_part1_b64.onnx"}},
    })
    assert remote_run._trt_preflight_part1_source(suite, "b064").name == "model_part1_b64.onnx"


def test_legacy_undeclared_ambiguity_is_still_rejected(tmp_path):
    suite = _suite(tmp_path, {})
    with pytest.raises(RuntimeError, match="source ambiguous"):
        remote_run._trt_preflight_part1_source(suite, "b064")


def _run(tmp_path):
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": "original", "tool_version": "2.91.0", "status": "failed",
    }))
    (run / "profile.yaml").write_text("model_suite: {}\n")
    (run / "evaluation_workflow.log").write_text("original log\n")
    return run


def test_large_required_metadata_uses_normal_streaming_export(tmp_path, monkeypatch):
    run = _run(tmp_path)
    summary = run / "reports/native_producer_summary.json"
    payload = b'{"rows":[],"diagnostics":"' + b'x' * (2 * 1024 * 1024) + b'"}\n'
    summary.write_bytes(payload)
    before = summary.stat()
    monkeypatch.setattr(packs, "EXACT_METADATA_MAX_FILE_BYTES", 1024 * 1024)
    # Whole-file reads of the source would recreate the very memory problem
    # the streaming exception must avoid. ZIP payload verification is separate.
    read_text, read_bytes = Path.read_text, Path.read_bytes
    def bounded_text(path, *args, **kwargs):
        assert path != summary
        return read_text(path, *args, **kwargs)
    def bounded_bytes(path, *args, **kwargs):
        assert path != summary
        return read_bytes(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", bounded_text)
    monkeypatch.setattr(Path, "read_bytes", bounded_bytes)
    result = packs.create_evaluation_debug_pack(run, tmp_path / "pack.zip")
    with zipfile.ZipFile(result["out_zip"]) as archive:
        name = summary.relative_to(run).as_posix()
        assert archive.read(name) == payload
        assert archive.getinfo(name).compress_type == zipfile.ZIP_DEFLATED
        manifest = json.loads(archive.read("debug_pack_manifest.json"))
        row = next(row for row in manifest["files"] if row["path"] == name)
        assert row["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
        assert manifest["offline_replay_core"]["copy_mode"] == "streamed_zip_deflate_exact_bytes"
    assert summary.stat().st_mtime_ns == before.st_mtime_ns
    assert summary.stat().st_size == before.st_size


def test_streaming_exception_remains_exact_and_bounded(tmp_path, monkeypatch):
    run = _run(tmp_path)
    core = run / "reports/native_producer_summary.json"
    other = run / "reports/native_producer_stage.json"
    for path in (core, other):
        path.write_text(" " * 80)
    monkeypatch.setattr(packs, "EXACT_METADATA_MAX_FILE_BYTES", 64)
    monkeypatch.setattr(packs, "OFFLINE_REPLAY_CORE_MAX_FILE_BYTES", 96)
    assert packs.should_include_debug_file(run, core, 64)[0]
    assert not packs.should_include_debug_file(run, other, 64)[0]
    core.write_text(" " * 97)
    assert not packs.should_include_debug_file(run, core, 64)[0]
    with pytest.raises(RuntimeError, match="required_debug_member_not_publishable"):
        packs.create_evaluation_debug_pack(run, tmp_path / "too-large.zip")
    assert not (tmp_path / "too-large.zip").exists()
