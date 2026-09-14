"""Real archive roundtrips for the existing regular-energy raw opt-in."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.workflow import debug_pack as packs


ROOT = "reports/native_energy_measurements/measurements/case/plan_id/attempt_id"
TRACE = ROOT + "/run_000/collector_storage/fast_firmware.parquet"
PROBE = "reports/window_method_validation_probe/run_000/collector_storage/fast_firmware.parquet"


def _write(run: Path, relative: str, payload: bytes) -> Path:
    path = run / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _run(tmp_path: Path, *, raw: bool | None = True, probe: bool = False) -> Path:
    run = tmp_path / "run"
    energy = {}
    if raw is not None:
        energy["include_raw_parquet_in_debug_pack"] = raw
    _write(run, "profile.yaml", yaml.safe_dump({
        "energy": energy,
        "native_producers": {"energy": {
            "window_method_validation_probe": {"include_raw_parquet": probe},
        }},
    }).encode())
    _write(run, "evaluation_workflow.log", b"complete workflow diagnostics\n")
    return run


def _reference(run: Path, trace: str = TRACE, *, raw_path: str | None = None) -> None:
    parent = Path(trace).parent.parent
    declared = raw_path or str(run / trace)
    _write(run, (parent / "command_window_request.json").as_posix(), json.dumps({
        "schema": "onnx-splitpoint/command-window-request", "trace_path": declared,
    }).encode())
    _write(run, (parent / "energy_summary.json").as_posix(), json.dumps({
        "parquet_files": [declared],
    }).encode())


def _pack(tmp_path: Path, run: Path) -> tuple[dict, dict[str, bytes]]:
    out = tmp_path / "debug.zip"
    result = packs.create_evaluation_debug_pack(run, out)
    assert result["archive_verification"] == "verified"
    with zipfile.ZipFile(out) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        return json.loads(archive.read("debug_pack_manifest.json")), {
            name: archive.read(name) for name in names
        }


def test_requested_regular_trace_is_archived_exactly_once_with_existing_hash(tmp_path):
    run = _run(tmp_path)
    payload = b"PAR1\x00test trace bytes\xffPAR1"
    _write(run, TRACE, payload)
    _reference(run)
    manifest, members = _pack(tmp_path, run)
    assert members[TRACE] == payload
    raw = manifest["native_energy_raw_traces"]
    assert [raw[key] for key in (
        "requested_count", "present_count", "archived_count", "omitted_count",
    )] == [1, 1, 1, 0]
    assert raw["status"] == "complete"
    assert raw["complete"] and manifest["complete"]
    assert raw["max_file_bytes"] == 64 * 1024**2
    assert raw["max_total_bytes"] == 512 * 1024**2
    record = next(row for row in manifest["files"] if row["path"] == TRACE)
    assert record["diagnostic_kind"] == "native_energy_raw_trace"
    assert record["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("raw", [False, None])
def test_off_or_absent_flag_keeps_probe_independent_and_default_compact(tmp_path, raw):
    run = _run(tmp_path, raw=raw, probe=True)
    _write(run, TRACE, b"regular")
    _write(run, PROBE, b"probe")
    manifest, members = _pack(tmp_path, run)
    assert TRACE not in members
    assert members[PROBE] == b"probe"
    regular = manifest["native_energy_raw_traces"]
    assert regular["requested_count"] == 0
    assert regular["present_count"] == 1
    assert regular["omitted_count"] == 1
    assert regular["status"] == "not_requested"
    assert manifest["complete"]


def test_regular_flag_does_not_enable_probe_or_other_parquet_locations(tmp_path):
    run = _run(tmp_path)
    others = [
        PROBE,
        "reports/arbitrary/collector_storage/leak.parquet",
        "reports/native_energy_measurements/unrelated/collector_storage/leak.parquet",
        ROOT + "/run_000/processed/copy.parquet",
        ROOT + "/run_000/collector_storage/nested/copy.parquet",
        "reports/native_energy_measurements/measurements_other/collector_storage/leak.parquet",
    ]
    _write(run, TRACE, b"regular")
    for path in others:
        _write(run, path, b"excluded")
    manifest, members = _pack(tmp_path, run)
    assert TRACE in members
    assert not set(others) & members.keys()
    assert manifest["window_method_validation_probe"]["raw_parquet_count"] == 0


@pytest.mark.parametrize("canonical", [False, None])
def test_nested_noncanonical_flag_cannot_override_effective_energy_setting(tmp_path, canonical):
    run = _run(tmp_path, raw=canonical)
    profile = yaml.safe_load((run / "profile.yaml").read_text())
    profile["native_producers"]["energy"]["include_raw_parquet_in_debug_pack"] = True
    (run / "profile.yaml").write_text(yaml.safe_dump(profile))
    _write(run, TRACE, b"regular")
    manifest, members = _pack(tmp_path, run)
    assert TRACE not in members
    assert manifest["native_energy_raw_traces"]["status"] == "not_requested"


def test_regular_file_limit_reports_partial_without_blocking_diagnostics(tmp_path, monkeypatch):
    run = _run(tmp_path)
    _write(run, TRACE, b"12345")
    monkeypatch.setattr(packs, "NATIVE_RAW_MAX_FILE_BYTES", 4)
    manifest, members = _pack(tmp_path, run)
    assert TRACE not in members
    assert "evaluation_workflow.log" in members
    raw = manifest["native_energy_raw_traces"]
    assert raw["requested_count"] == raw["present_count"] == raw["omitted_count"] == 1
    assert raw["omitted_members"][0]["reason"] == "native energy raw trace exceeds file limit"
    assert raw["status"] == "partial"
    assert not raw["complete"] and not manifest["complete"]


def test_regular_total_budget_is_deterministic_and_independent_of_probe(tmp_path, monkeypatch):
    run = _run(tmp_path, probe=True)
    second = TRACE.replace("run_000", "run_001")
    third = TRACE.replace("run_000", "run_002")
    # Deliberately create in reverse order. Exactly equal to the limit is allowed.
    for path in (third, second, TRACE):
        _write(run, path, b"1234")
    _write(run, PROBE, b"123456")
    monkeypatch.setattr(packs, "NATIVE_RAW_MAX_TOTAL_BYTES", 8)
    manifest, members = _pack(tmp_path, run)
    assert TRACE in members and second in members and third not in members
    assert PROBE in members
    raw = manifest["native_energy_raw_traces"]
    assert raw["archived_count"] == 2 and raw["omitted_count"] == 1
    assert raw["archived_total_bytes"] == 8
    assert raw["omitted_members"][0]["reason"] == "native energy raw total limit exceeded"
    assert not manifest["complete"]


@pytest.mark.parametrize("relocated", [False, True])
def test_declared_missing_trace_is_explicit_and_never_claimed_complete(tmp_path, relocated):
    run = _run(tmp_path)
    declared = "/former/evaluation/run/" + TRACE if relocated else None
    _reference(run, raw_path=declared)
    manifest, members = _pack(tmp_path, run)
    raw = manifest["native_energy_raw_traces"]
    assert TRACE not in members
    assert raw["requested_count"] == raw["omitted_count"] == 1
    assert raw["present_count"] == raw["archived_count"] == 0
    assert raw["missing_source_members"] == [TRACE]
    assert raw["omitted_members"] == [{"path": TRACE, "reason": "source_missing"}]
    assert raw["status"] == "partial" and not manifest["complete"]


def test_external_reference_is_not_followed_or_silently_complete(tmp_path):
    run = _run(tmp_path)
    external = tmp_path / "outside.parquet"
    external.write_bytes(b"private external contents")
    _reference(run, raw_path=str(external))
    manifest, members = _pack(tmp_path, run)
    assert not any(name.endswith(".parquet") for name in members)
    assert external.read_bytes() == b"private external contents"
    assert manifest["native_energy_raw_traces"]["reference_failures"]
    assert not manifest["complete"]


@pytest.mark.parametrize("directory_link", [False, True])
def test_existing_safe_enumerator_never_follows_raw_symlinks(tmp_path, directory_link):
    run = _run(tmp_path)
    target = tmp_path / "external" / "fast_firmware.parquet"
    target.parent.mkdir()
    target.write_bytes(b"private trace")
    source = run / TRACE
    source.parent.parent.mkdir(parents=True)
    if directory_link:
        source.parent.symlink_to(target.parent, target_is_directory=True)
    else:
        source.parent.mkdir()
        source.symlink_to(target)
    _reference(run)
    manifest, members = _pack(tmp_path, run)
    assert TRACE not in members
    assert manifest["native_energy_raw_traces"]["missing_source_members"] == [TRACE]
    assert not manifest["complete"]


def test_invalid_collector_reference_is_visible_but_debug_pack_survives(tmp_path):
    run = _run(tmp_path)
    _write(run, ROOT + "/run_000/command_window_request.json", b"{invalid")
    manifest, members = _pack(tmp_path, run)
    assert "evaluation_workflow.log" in members
    raw = manifest["native_energy_raw_traces"]
    assert raw["reference_failures"][0]["reason"] == "raw_reference_invalid:JSONDecodeError"
    assert raw["status"] == "partial" and not manifest["complete"]
