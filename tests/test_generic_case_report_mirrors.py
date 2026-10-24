"""Local collector snapshots; no inference, network or historic run mutation."""
import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

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
    assert {r["sha256"] for r in rows} == {"sha256:" + hashlib.sha256(payload).hexdigest()}
    assert {r["size_bytes"] for r in rows} == {len(payload)}
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


def downloaded_collection_error():
    return {
        "status": "partial",
        "remote_rc": 0,
        "remote_dispatched": True,
        "error": "Remote result collection failed (rc=1)",
        "remote_cleanup": {
            "reason": "result_collection_not_successful",
            "policy": {"results_downloaded": True, "result_collection_rc": 1},
        },
    }


def test_closed_capture_report_eligibility_requires_no_defer_metadata():
    output = {"status": "ok", "remote_rc": 0, "remote_dispatched": True}
    assert binding._closed_generic_report_snapshots(True, output)
    output["workflow_execution"] = {"defer_central_quality": False}
    assert binding._closed_generic_report_snapshots(True, output)


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_downloaded_collector_rc1_keeps_closed_capture_report_eligibility(status):
    output = downloaded_collection_error()
    output["status"] = status
    before = json.loads(json.dumps(output))
    assert binding._closed_generic_report_snapshots(True, output)
    assert output == before


@pytest.mark.parametrize("rejection", [
    "scope", "cancelled", "active", "workload_rc", "bool_rc", "terminal",
    "primary", "failure_kind", "error", "not_downloaded", "not_dispatched",
])
def test_unclosed_capture_reports_are_not_eligible_for_sharing(rejection):
    output = downloaded_collection_error()
    scope, cancelled = True, False
    if rejection == "scope":
        scope = False
    elif rejection == "cancelled":
        cancelled = True
    elif rejection == "active":
        output["status"] = "running"
    elif rejection == "workload_rc":
        output["remote_rc"] = 7
    elif rejection == "bool_rc":
        output["remote_rc"] = False
    elif rejection == "terminal":
        output["terminal_remote_failure"] = True
    elif rejection == "primary":
        output["primary_failure"] = {"reason": "test workload failed"}
    elif rejection == "failure_kind":
        output["failure_kind"] = "test_workload_failed"
    elif rejection == "error":
        output["error"] = "Unrelated test service failure"
    elif rejection == "not_downloaded":
        output["remote_cleanup"]["policy"]["results_downloaded"] = False
    elif rejection == "not_dispatched":
        output["remote_dispatched"] = False
    assert binding._closed_generic_report_snapshots(scope, output, cancelled=cancelled) is False


@pytest.mark.parametrize("change_at", ["materialization", "raw_hash"])
def test_unsettled_raw_version_prevents_both_snapshot_publications(tmp_path, monkeypatch, change_at):
    monkeypatch.setattr(artifact_store, "_owned_reflink_copy", lambda *args: 1)
    raw, dst, canonical, alias = paths(tmp_path)
    payload, initial_mtime = raw.read_bytes(), raw.stat().st_mtime_ns
    real_materialize = artifact_store.materialize_file
    real_hash = binding.sha256_file_uncached
    changed = []

    def rewrite_raw():
        raw.write_bytes(payload)
        current = raw.stat()
        os.utime(raw, ns=(current.st_atime_ns, initial_mtime + 1_000_000_000))
        changed.append(True)

    def materialize(source, destination, **kwargs):
        result = real_materialize(source, destination, **kwargs)
        if change_at == "materialization" and Path(source) == raw and not changed:
            rewrite_raw()
        return result

    def hash_file(path, *args, **kwargs):
        result = real_hash(path, *args, **kwargs)
        if change_at == "raw_hash" and Path(path) == raw and not changed:
            rewrite_raw()
        return result

    monkeypatch.setattr(artifact_store, "materialize_file", materialize)
    monkeypatch.setattr(binding, "sha256_file_uncached", hash_file)
    manifest = collect(raw, dst)
    assert changed == [True]
    assert any(r["kind"] == "case_validation_report" for r in manifest["errors"])
    assert not [r for r in manifest["copied"] if r["kind"] == "case_validation_report"]
    assert not canonical.exists() and not alias.exists()
    assert raw.read_bytes() == payload and raw.stat().st_nlink == 1
    assert not list(canonical.parent.glob(".*.tmp"))


@pytest.mark.parametrize("bad_snapshot", ["raw_hardlink", "raw_symlink"])
def test_raw_alias_cannot_be_published_as_first_snapshot(tmp_path, monkeypatch, bad_snapshot):
    raw, dst, canonical, alias = paths(tmp_path)
    payload = raw.read_bytes()

    def materialize(source, destination, **kwargs):
        assert Path(source) == raw
        assert kwargs["allow_hardlink"] is False
        if bad_snapshot == "raw_hardlink":
            os.link(source, destination)
        else:
            Path(destination).symlink_to(source)
        return "copy"

    monkeypatch.setattr(artifact_store, "materialize_file", materialize)
    manifest = collect(raw, dst)
    assert any(r["kind"] == "case_validation_report" for r in manifest["errors"])
    assert not [r for r in manifest["copied"] if r["kind"] == "case_validation_report"]
    assert not canonical.exists() and not alias.exists()
    assert raw.read_bytes() == payload and raw.stat().st_nlink == 1
    assert not list(canonical.parent.glob(".*.tmp"))


def test_full_catalogue_shares_only_selected_generic_scope_and_target_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_store, "_owned_reflink_copy", lambda *args: 1)
    rows = [
        {"id": "  Generic / H8 :: source-only  ", "hardware_setup_id": "setup", "case_ids": ["b007"]},
        {"id": "other-target", "hardware_setup_id": "other", "case_ids": ["b008"]},
    ]
    scopes = binding._generic_report_snapshot_scopes(rows, "setup")
    folder = "results_Generic_H8_source-only"
    assert scopes == {("b007", folder)}
    remote, dst = tmp_path / "transport", tmp_path / "evaluation"
    reports = []
    for index in range(1, 21):
        report = remote / "results" / f"b{index:03}" / folder / "validation_report.json"
        report.parent.mkdir(parents=True)
        report.write_bytes(json.dumps({"own_case": index, "kind": "generic"}).encode())
        reports.append(report)
    native = remote / "results/b007/results_native_fp16/validation_report.json"
    native.parent.mkdir(parents=True)
    native.write_bytes(b'{"own_case": 7, "kind": "native"}')
    reports.append(native)
    binding._copy_remote_result_files(
        remote, dst, flat_prefix="setup",
        share_generic_case_reports=binding._closed_generic_report_snapshots(
            bool(scopes), {"status": "ok", "remote_rc": 0, "remote_dispatched": True}),
        generic_case_report_scopes=scopes,
    )
    manifest = json.loads((dst / "remote_diagnostics/setup/result_copy_manifest.json").read_text())
    assert not manifest["errors"]
    copied = [r for r in manifest["copied"] if r["kind"] == "case_validation_report"]
    assert len(copied) == 42
    for raw in reports:
        relative = raw.relative_to(remote)
        first = dst / "remote_diagnostics/case_reports" / relative
        second = dst / "remote_diagnostics/setup/case_reports" / relative
        selected = tuple(relative.parts[1:3]) in scopes
        assert os.path.samefile(first, second) is selected
        assert not os.path.samefile(raw, first)
        assert not os.path.samefile(raw, second)
        assert raw.stat().st_nlink == 1
        assert first.read_bytes() == second.read_bytes() == raw.read_bytes()
        matches = [row for row in copied if row["source"] == str(raw)]
        assert len(matches) == 2
        assert {row["sha256"] for row in matches} == {"sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest()}


def test_late_first_snapshot_mutation_cannot_publish_second(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_store, "_owned_reflink_copy", lambda *args: 1)
    raw, dst, canonical, alias = paths(tmp_path)
    payload = raw.read_bytes()
    unknown = payload + b"\nOWN late FIRST mutation must survive\n"
    real_hash = binding.sha256_file_uncached
    witnessed = []

    def hash_file(path, *args, **kwargs):
        result = real_hash(path, *args, **kwargs)
        path = Path(path)
        if path.parent == alias.parent and path.name.startswith("." + alias.name + "."):
            assert os.path.samefile(canonical, path)
            assert canonical.read_bytes() == payload
            witnessed.append(result)
            before = canonical.stat()
            canonical.write_bytes(unknown)
            current = canonical.stat()
            os.utime(canonical, ns=(current.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(binding, "sha256_file_uncached", hash_file)
    manifest = collect(raw, dst)
    assert witnessed == ["sha256:" + hashlib.sha256(payload).hexdigest()]
    assert any(r["destination"] == str(alias)
               and "case_report_snapshot_bytes_changed" in r["reason"]
               for r in manifest["errors"])
    rows = [r for r in manifest["copied"] if r["kind"] == "case_validation_report"]
    assert len(rows) == 1 and rows[0]["destination"] == str(canonical)
    assert not alias.exists()
    assert canonical.read_bytes() == unknown and canonical.stat().st_nlink == 1
    assert raw.read_bytes() == payload and raw.stat().st_nlink == 1
    assert not os.path.samefile(raw, canonical)
    assert not list(alias.parent.glob(".*.tmp"))
