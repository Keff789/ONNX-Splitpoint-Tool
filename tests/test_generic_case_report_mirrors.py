"""Local collector snapshots; no inference, network or historic run mutation."""
import errno
import json
import os
from pathlib import Path

from onnx_splitpoint_tool import artifact_store
from onnx_splitpoint_tool.workflow import execution_binding as binding


def paths(tmp_path):
    raw = tmp_path / "transport/results/b001/results_hailo8_to_trt/validation_report.json"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b'{"test_report": "first"}')
    dst = tmp_path / "evaluation"
    rel = raw.relative_to(tmp_path / "transport")
    canonical = dst / "remote_diagnostics/case_reports" / rel
    alias = dst / "remote_diagnostics/setup/case_reports" / rel
    return raw, dst, canonical, alias


def collect(raw, dst, *, share=True):
    binding._copy_remote_result_files(raw.parents[3], dst, flat_prefix="setup",
                                     share_generic_case_reports=share)
    return json.loads((dst / "remote_diagnostics/setup/result_copy_manifest.json").read_text())


def test_report_mirrors_share_only_independent_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_store, "_owned_reflink_copy", lambda *args: 1)
    raw, dst, canonical, alias = paths(tmp_path)
    payload = raw.read_bytes()
    manifest = collect(raw, dst)
    assert not manifest["errors"]
    assert os.path.samefile(canonical, alias)
    assert not os.path.samefile(raw, canonical)
    rows = [r for r in manifest["copied"] if r["kind"] == "case_validation_report"]
    assert len(rows) == 2 and {r["source"] for r in rows} == {str(raw)}
    assert len({r["sha256"] for r in rows}) == 1
    assert rows[0]["materialization"] == "copy" and rows[1]["materialization"] == "hardlink"
    # A later normal download may rewrite its transport inode in place.
    raw.write_bytes(b'{"test_report": "download replaced raw"}')
    assert canonical.read_bytes() == alias.read_bytes() == payload


def test_repeated_collect_replaces_both_reports_without_truncating_old_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_store, "_owned_reflink_copy", lambda *args: 1)
    raw, dst, canonical, alias = paths(tmp_path)
    collect(raw, dst)
    old = alias.read_bytes()
    new = b'{"test_report": "second"}'
    raw.write_bytes(new)
    real_link = os.link
    witnessed = []

    def link(source, destination, *args, **kwargs):
        if Path(source) == canonical:
            witnessed.append((canonical.read_bytes(), alias.read_bytes()))
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", link)
    with alias.open("rb") as previous_inode:
        assert not collect(raw, dst)["errors"]
        assert previous_inode.read() == old
    assert witnessed == [(new, old)]
    assert os.path.samefile(canonical, alias)
    assert canonical.read_bytes() == alias.read_bytes() == raw.read_bytes() == new
    # A later ordinary/non-sharing collection must also replace old hardlinks.
    raw.write_bytes(old)
    with alias.open("rb") as previous_inode:
        assert not collect(raw, dst, share=False)["errors"]
        assert previous_inode.read() == new
    assert not os.path.samefile(canonical, alias)
    assert canonical.read_bytes() == alias.read_bytes() == old


def test_link_fallback_and_corrupt_snapshot_fail_without_overwriting_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_store, "_owned_reflink_copy", lambda *args: 1)
    raw, dst, canonical, alias = paths(tmp_path)

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "test cross-device link")

    monkeypatch.setattr(os, "link", cross_device)
    manifest = collect(raw, dst)
    assert not manifest["errors"]
    assert not os.path.samefile(canonical, alias)
    assert canonical.read_bytes() == alias.read_bytes() == raw.read_bytes()
    assert {r["materialization"] for r in manifest["copied"]
            if r["kind"] == "case_validation_report"} == {"copy"}
    previous = canonical.read_bytes()
    raw.write_bytes(b'{"test_report": "third"}')

    def corrupt(source, destination, **kwargs):
        assert kwargs["allow_hardlink"] is False
        Path(destination).write_bytes(b"incorrect test copy")
        return "copy"

    monkeypatch.setattr(artifact_store, "materialize_file", corrupt)
    manifest = collect(raw, dst)
    assert any("case_report_snapshot_bytes_changed" in r["reason"] for r in manifest["errors"])
    assert not [r for r in manifest["copied"] if r["kind"] == "case_validation_report"]
    assert canonical.read_bytes() == alias.read_bytes() == previous
    assert not list(canonical.parent.glob(".*.tmp"))
