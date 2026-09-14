"""T04: real ZIP export of original v2.80 CPU-reference failure diagnostics."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import debug_pack as packs


FIXTURE = Path(__file__).parent / "fixtures" / "v2801_cpu_reference_details"
PREFIX = "quality_management/references/"
STATUS = "management_cpu_reference_status.json"
STDOUT = "management_cpu_reference_stdout.txt"


def _write(run: Path, relative: str, payload: bytes = b"{}") -> Path:
    path = run / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "new_export_run"
    _write(run, "evaluation_workflow.log", b"Original failed run diagnostics.\n")
    return run


def _index(run: Path, paths: list[str]) -> None:
    _write(run, "artifact_index.json", json.dumps({
        "schema": "onnx-splitpoint/artifact-index", "schema_version": 1,
        "artifacts": [{"path": path} for path in paths],
    }).encode())


def _pack(tmp_path: Path, run: Path, **kwargs) -> tuple[dict, dict[str, bytes]]:
    result = packs.create_evaluation_debug_pack(run, tmp_path / "debug.zip", **kwargs)
    assert result["archive_verification"] == "verified"
    with zipfile.ZipFile(result["out_zip"]) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        members = {name: archive.read(name) for name in archive.namelist()}
    return json.loads(members["debug_pack_manifest.json"]), members


def _snapshot(run: Path) -> dict[str, tuple[bytes, int]]:
    return {path.relative_to(run).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in run.rglob("*") if path.is_file() and not path.is_symlink()}


def test_t0401_all_twelve_original_diagnostics_are_exported_byte_identically(tmp_path):
    run = _run(tmp_path)
    shutil.copytree(FIXTURE / "quality_management", run / "quality_management")
    records = json.loads((FIXTURE / "original_index_records.json").read_text())
    _write(run, "artifact_index.json", json.dumps({"artifacts": records}).encode())
    before = _snapshot(run)
    manifest, members = _pack(tmp_path, run)
    assert len(records) == 12
    assert sum(record["size_bytes"] for record in records) == 11890
    files = {row["path"]: row for row in manifest["files"]}
    for record in records:
        path = record["path"]
        original = (FIXTURE / path).read_bytes()
        assert members[path] == original
        assert len(original) == record["size_bytes"]
        assert "sha256:" + hashlib.sha256(original).hexdigest() == record["sha256"]
        assert files[path]["sha256"] == record["sha256"]
        assert files[path]["diagnostic_kind"] == "management_cpu_reference_diagnostic"
        if path.endswith(STATUS):
            payload = json.loads(original)
            assert payload["return_code"] == 1
            assert payload["errors"] == ["quality_reference_not_emitted"]
        else:
            assert b"generic Full central-quality dispatch identity is incomplete" in original
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["complete"] and manifest["complete"]
    assert inventory["archived_total_bytes"] == 11890
    assert set(inventory["archived_members"]) == {r["path"] for r in records}
    assert not inventory["omitted_source_members"]
    assert _snapshot(run) == before


@pytest.mark.parametrize("relative", [
    PREFIX + "model/by_source_contract/sha/canonical_cpu_reference.json",
    PREFIX + "model/canonical_cpu_reference.json",
    PREFIX + "model/reference.json",
    PREFIX + "model/candidate.json",
    "quality_management/quality_inputs/candidate.json",
    "quality_management/quality_inputs/reference_predictions.json",
    "quality_management/quality_inputs/annotations.json",
    PREFIX + "model/workspaces/" + STATUS,
    PREFIX + "model/workspaces/" + STDOUT,
    PREFIX + "model/deeper/" + STATUS,
    PREFIX + "workspaces/" + STATUS,
    PREFIX + "model/" + STATUS + ".backup.json",
])
def test_t0402_t0403_small_bodies_and_noncanonical_names_stay_excluded(tmp_path, relative):
    run = _run(tmp_path)
    path = _write(run, relative, b'{"small":true}')
    assert not packs.is_management_reference_diagnostic(relative)
    assert not packs.should_include_debug_file(run, path)[0]
    _, members = _pack(tmp_path, run)
    assert relative not in members


@pytest.mark.parametrize("relative", [
    PREFIX + "../" + STATUS,
    PREFIX + "./" + STATUS,
    PREFIX + "" + "/" + STATUS,
    "/" + PREFIX + "model/" + STATUS,
    PREFIX + "model\\..\\external/" + STATUS,
])
def test_t0403_unsafe_lexical_paths_cannot_enable_exception(relative):
    assert not packs.is_management_reference_diagnostic(relative)


@pytest.mark.parametrize("directory_link", [False, True])
def test_t0403_external_symlink_is_not_read_and_is_inventoried(tmp_path, directory_link):
    run = _run(tmp_path)
    relative = PREFIX + "model/" + STDOUT
    external = _write(tmp_path, "external/" + STDOUT, b"EXTERNAL PRIVATE BYTES")
    destination = run / relative
    if directory_link:
        destination.parent.parent.mkdir(parents=True)
        destination.parent.symlink_to(external.parent, target_is_directory=True)
        _index(run, [relative])
    else:
        destination.parent.mkdir(parents=True)
        destination.symlink_to(external)
        # A direct unsafe file is inventoried even without a runindex record.
    assert not packs.should_include_debug_file(run, destination)[0]
    manifest, members = _pack(tmp_path, run)
    assert relative not in members
    assert not any(b"EXTERNAL PRIVATE BYTES" in payload for payload in members.values())
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["expected_members"] == [relative]
    assert inventory["omitted_source_members"] == [{"path": relative, "reason": "source_unsafe:ValueError"}]
    assert not inventory["complete"] and not manifest["complete"]
    assert external.read_bytes() == b"EXTERNAL PRIVATE BYTES"


@pytest.mark.parametrize("declared", [
    PREFIX + "model/../../external/" + STDOUT,
    "/outside/" + PREFIX + "model/" + STDOUT,
    PREFIX + "model/workspaces/" + STDOUT,
])
def test_t0403_index_does_not_authorize_external_or_noncanonical_diagnostics(tmp_path, declared):
    run = _run(tmp_path)
    _index(run, [declared])
    manifest, members = _pack(tmp_path, run)
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert not inventory["expected_members"]
    assert inventory["reference_failures"] == [{
        "path": declared, "reason": "reference_diagnostic_index_path_not_canonical",
    }]
    assert declared not in members
    assert not manifest["complete"]


def test_t0404_missing_registered_files_are_reported_without_inventing_other_jobs(tmp_path):
    run = _run(tmp_path)
    present = PREFIX + "model/" + STATUS
    missing = PREFIX + "model/" + STDOUT
    _write(run, present)
    _write(run, "models/no_reference_job/model_manifest.json", b'{"model_id":"no_reference_job"}')
    _index(run, [present, missing, missing])
    manifest, members = _pack(tmp_path, run)
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["expected_members"] == sorted([present, missing])
    assert inventory["missing_source_members"] == [missing]
    assert inventory["omitted_source_members"] == [{"path": missing, "reason": "source_missing"}]
    assert present in members and missing not in members
    assert not manifest["complete"]


def test_t0404_unregistered_absent_counterpart_is_not_a_missing_job(tmp_path):
    run = _run(tmp_path)
    present = PREFIX + "model/" + STATUS
    _write(run, present)
    manifest, _ = _pack(tmp_path, run)
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["expected_members"] == [present]
    assert not inventory["missing_source_members"]
    # v2.80.3 separates copied visible diagnostics from unknown index coverage.
    assert inventory["all_admitted_members_archived"]
    assert inventory["index_validation"]["status"] == "missing"
    assert not inventory["complete"] and not manifest["complete"]


@pytest.mark.parametrize("empty_index", [False, True])
def test_t0404_run_without_references_or_index_remains_complete(tmp_path, empty_index):
    run = _run(tmp_path)
    if empty_index:
        _index(run, [])
    manifest, _ = _pack(tmp_path, run)
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["status"] == "not_present"
    assert not inventory["expected_members"] and not inventory["reference_failures"]
    assert inventory["complete"] and manifest["complete"]


def test_t0404_hard_file_ceiling_still_applies_to_large_caller_allowance(tmp_path, monkeypatch):
    run = _run(tmp_path)
    relative = PREFIX + "model/" + STDOUT
    _write(run, relative, b"12345")
    monkeypatch.setattr(packs, "MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_FILE_BYTES", 4)
    manifest, members = _pack(tmp_path, run, max_small_file_bytes=128)
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["max_file_bytes"] == 4 and relative not in members
    assert inventory["omitted_source_members"][0]["reason"] == "management reference diagnostic exceeds compact file limit"
    assert not inventory["complete"] and not manifest["complete"]


@pytest.mark.parametrize("filename", [STATUS, STDOUT])
def test_t0404_file_limit_omission_does_not_masquerade_as_original_or_tail(tmp_path, filename, monkeypatch):
    monkeypatch.setattr(packs, "MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_FILE_BYTES", 32)
    run = _run(tmp_path)
    relative = PREFIX + "model/" + filename
    _write(run, relative, b"x" * 33)
    manifest, members = _pack(tmp_path, run, max_small_file_bytes=32, tail_bytes=16)
    assert relative not in members
    assert "debug_tails/" + relative + ".tail" not in members
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["omitted_source_members"] == [{
        "path": relative, "reason": "management reference diagnostic exceeds compact file limit", "size_bytes": 33,
    }]
    assert inventory["status"] == "partial" and not manifest["complete"]


def test_t0404_total_budget_is_deterministic_and_does_not_evict_other_diagnostics(tmp_path, monkeypatch):
    run = _run(tmp_path)
    paths = [PREFIX + model + "/" + STATUS for model in ("a", "b", "c")]
    for path in reversed(paths):
        _write(run, path, b"1234")
    _write(run, "reports/native_producer_stage.json", b'{"other":true}')
    monkeypatch.setattr(packs, "MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_TOTAL_BYTES", 8)
    manifest, members = _pack(tmp_path, run)
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["archived_members"] == paths[:2]
    assert inventory["archived_total_bytes"] == 8
    assert paths[2] not in members
    assert inventory["omitted_source_members"] == [{
        "path": paths[2], "reason": "management reference diagnostic total limit exceeded", "size_bytes": 4, "limit_bytes": 8,
    }]
    assert "reports/native_producer_stage.json" in members
    assert not manifest["complete"]


def test_t0404_unreadable_original_is_visible_in_inventory(tmp_path, monkeypatch):
    run = _run(tmp_path)
    relative = PREFIX + "model/" + STDOUT
    source = _write(run, relative, b"source diagnostics")
    real_open = Path.open

    def unreadable(path, *args, **kwargs):
        if path == source:
            raise PermissionError("controlled unreadable diagnostic")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    manifest, members = _pack(tmp_path, run)
    assert relative not in members
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["omitted_source_members"] == [{"path": relative, "reason": "source_unreadable:PermissionError"}]
    assert inventory["status"] == "partial" and not manifest["complete"]


def test_t0404_late_read_failure_does_not_publish_false_complete_zip(tmp_path, monkeypatch):
    run = _run(tmp_path)
    relative = PREFIX + "model/" + STDOUT
    _write(run, relative, b"source diagnostics")
    write_member = packs._write_source_member

    def fail(archive, source, member, **kwargs):
        if member == relative:
            raise PermissionError("read failed after admission")
        return write_member(archive, source, member, **kwargs)

    monkeypatch.setattr(packs, "_write_source_member", fail)
    with pytest.raises(RuntimeError, match="required_debug_member_write_failed"):
        _pack(tmp_path, run)
    assert not (tmp_path / "debug.zip").exists()
    assert not (tmp_path / "debug.zip.manifest.json").exists()


@pytest.mark.parametrize("payload", [b'{"status":', b"not JSON\xff", b"{}"])
def test_t0405_malformed_status_is_raw_diagnostic_not_semantic_success(tmp_path, payload):
    run = _run(tmp_path)
    relative = PREFIX + "model/" + STATUS
    _write(run, relative, payload)
    _index(run, [relative])  # Archive completeness includes verified index coverage.
    manifest, members = _pack(tmp_path, run)
    assert members[relative] == payload
    inventory = manifest["management_cpu_reference_diagnostics"]
    assert inventory["complete"]  # Archive completeness, never model/quality PASS.
    assert inventory["content_policy"] == "original_bytes_only; diagnostic_status_is_not_semantic_success"
    assert "quality" not in inventory and "reference_status" not in inventory


def test_t0406_excluded_data_and_external_stdout_pointer_never_enter_zip(tmp_path):
    run = _run(tmp_path)
    excluded = [PREFIX + "model/" + name for name in [
        "model.onnx", "model.hef", "model.har", "model.dxnn", "full.engine",
        "output.npz", "output.npy", "output.bin", "image.jpg", "archive.zip",
        "workspaces/note.json", ".venv/metadata.json", "venv/metadata.json",
    ]]
    for relative in excluded:
        _write(run, relative, b"excluded payload")
    external = _write(tmp_path, "outside.txt", b"DO NOT FOLLOW THIS POINTER")
    status = PREFIX + "model/" + STATUS
    _write(run, status, json.dumps({"stdout_path": str(external), "status": "failed"}).encode())
    before = _snapshot(run)
    manifest, members = _pack(tmp_path, run)
    assert not set(excluded) & members.keys()
    assert not any(b"DO NOT FOLLOW THIS POINTER" in payload for payload in members.values())
    assert status in members
    assert manifest["management_cpu_reference_diagnostics"]["expected_members"] == [status]
    assert _snapshot(run) == before


def test_t0406_existing_native_energy_and_runtime_diagnostics_remain_exact(tmp_path):
    run = _run(tmp_path)
    exact = [
        PREFIX + "model/" + STATUS,
        "reports/native_producer_stage.json",
        "reports/native_energy_measurements/native_producer_energy_results.json",
        "models/model/benchmark_results/remote_diagnostics/logs/runner.log",
        "models/model/benchmark_results/normalized_results.json",
    ]
    for relative in exact:
        _write(run, relative, b"{}")
    _index(run, [exact[0]])
    manifest, members = _pack(tmp_path, run)
    assert all(members[path] == b"{}" for path in exact)
    assert manifest["runtime_execution_diagnostics"]["complete"]
    assert manifest["management_cpu_reference_diagnostics"]["complete"]
