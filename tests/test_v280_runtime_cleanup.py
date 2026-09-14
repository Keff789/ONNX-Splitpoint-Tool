"""Real collector lifecycle with synthetic SSH/inference, never hardware acceptance."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load("runtime_cleanup_v280", ROOT / "scripts/hailo_model_runtime_probe_v27934.py")
fixtures = load("runtime_cleanup_baseline_v280", ROOT / "tests/test_v27934_model_diagnostic.py")
baseline = fixtures.baseline


@pytest.fixture
def case(baseline, tmp_path, monkeypatch):
    args, request = baseline
    build = tmp_path / "private_build"
    build.mkdir()
    hef = build / "compiled.hef"
    hef.write_bytes(Path(request["cpu_hef"]["path"]).read_bytes())
    receipt = copy.deepcopy(request["cpu_build_receipt"])
    receipt.update(diagnostic_only=True, publish_artifacts=False)
    fixtures.backend._atomic_write_json(fixtures.backend._hailo_receipt_path(hef), receipt)
    summary = {"model_build_status": "pass", "gpu_execution_status": "pass",
               "recipe_matches_cpu_baseline": True, "private_hef": probe.file_identity(hef)}
    probe.write_json(build / "request.json", request)
    probe.write_json(build / "summary.json", summary)
    registry = tmp_path / "hardware.json"
    registry.write_text(json.dumps({"hardware_setups": [{"id": "fixture_h10", "accelerator": "hailo10h",
        "host": {"address": "fixture.invalid", "user": "fixture", "port": 22}}]}))
    output = tmp_path / "runtime"
    state = {"mktemp": "/tmp/onnx-v27934-hailo-model-A1b2C3d4E5\n", "runtime_rc": 0,
             "supervision": {"cleanup_complete": True}, "calls": [], "cleanup_rc": 0,
             "output": output, "runtime_error": None}

    def transport(command, **kwargs):
        state["calls"].append((command, kwargs))
        if "mktemp -d" in command[-1]:
            if state.get("mktemp_error"):
                raise state["mktemp_error"]
            return subprocess.CompletedProcess(command, 0, state["mktemp"])
        if command[0] == "scp" and command[-1].endswith("/results"):
            results = output / "results"
            results.mkdir()
            if "supervision" in state:
                payload = state["supervision"]
                (results / "supervision.json").write_text(payload if isinstance(payload, str) else json.dumps(payload))
            if state.get("foreign_supervision"):
                (results / "supervision.json").unlink(missing_ok=True)
                (results / "supervision.json").symlink_to(state["foreign_supervision"])
            runtime_result = {"runtime_status": "failed" if state["runtime_error"] else "pass",
                "stages": {"cpu_hef": {"inference_count": 16}, "gpu_hef": {"inference_count": 16}}}
            if state["runtime_error"]:
                runtime_result["error"] = state["runtime_error"]
            probe.write_json(results / "runtime_result.json", runtime_result)
            (results / "runtime_arrays.npz").write_bytes(b"private arrays never exported")
            (results / "model.hef").write_bytes(b"private model never exported")
            if state.get("collect_error"):
                raise state["collect_error"]
        elif command[0] == "scp":
            if state.get("upload_error"):
                raise state["upload_error"]
        elif command[-1].startswith("rm -rf"):
            if state.get("noop_rm_postcondition"):
                # Real shell filesystem postcondition; only the external rm is
                # made a no-op to emulate an unsuccessful deletion with RC0.
                return subprocess.run(["bash", "-c", "rm() { return 0; }; " + command[-1]],
                                      timeout=kwargs["timeout"], capture_output=True)
            if state.get("cleanup_error"):
                raise state["cleanup_error"]
            value = subprocess.CompletedProcess(command, state["cleanup_rc"])
            if kwargs.get("check"):
                value.check_returncode()
            return value
        elif "hailo_model_runtime_worker_v27934.py" in command[-1]:
            if state.get("runtime_exception"):
                raise state["runtime_exception"]
            return subprocess.CompletedProcess(command, state["runtime_rc"])
        return subprocess.CompletedProcess(command, 0)

    def infer(*args):
        if state.get("comparison_error"):
            raise state["comparison_error"]
        return {"runtime_status": "pass", "stages": {name: {"finite": True, "count": 16}
            for name in ("source_float", "build_float", "cpu_hef", "gpu_hef")},
            "diagnostic_measurement": 0.25, "quality_status": "not_evaluated", "claim_eligible": False}

    monkeypatch.setattr(probe, "run_transport", transport)
    monkeypatch.setattr(probe, "float_comparison", infer)
    state["argv"] = ["--build-dir", str(build), "--hardware-registry", str(registry),
        "--setup-id", "fixture_h10", "--remote-python", sys.executable, "--output-dir", str(output)]
    return state


def run(case):
    rc = probe.main(case["argv"])
    output = case["output"]
    report = json.loads((output / "comparison.json").read_text())
    archive = output.parent / (output.name + "_evidence.zip")
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert json.loads(zipped.read(output.name + "/comparison.json")) == report
        assert not any(name.endswith((".npz", ".hef", ".onnx")) for name in zipped.namelist())
    return rc, report


def cleanup_calls(case):
    return [(cmd, kw) for cmd, kw in case["calls"] if cmd[-1].startswith("rm -rf")]


def test_rm_failure_preserves_process_proof_but_invalidates_full_cleanup(case):
    case["cleanup_rc"] = 1
    rc, report = run(case)
    assert report["remote_cleanup_complete"] is False
    assert report["remote_process_cleanup_complete"] is True
    assert report["remote_staging_cleanup"] == {"status": "failed", "attempted": True,
        "returncode": 1, "timed_out": False, "error": report["remote_staging_cleanup"]["error"]}
    assert report["remote_staging_cleanup"]["error"]
    assert report["runtime_status"] == "failed" and report["g3_status"] == "incomplete" and rc == 2
    assert report["diagnostic_measurement"] == 0.25


def test_success_requires_rm_and_absence_postcondition_in_same_bounded_call(case):
    rc, report = run(case)
    assert rc == 0 and report["runtime_status"] == report["g3_status"] == "pass"
    assert report["remote_process_cleanup_complete"] is True
    assert report["remote_cleanup_complete"] is True
    assert report["remote_staging_cleanup"] == {"status": "pass", "attempted": True,
        "returncode": 0, "timed_out": False, "error": None}
    [(command, options)] = cleanup_calls(case)
    target = case["mktemp"].strip()
    assert command[-1] == f"rm -rf -- {target} && test '!' -e {target} && test '!' -L {target}"
    assert options["timeout"] == 30
    assert report["claim_eligible"] is False


@pytest.mark.parametrize("error,code,timed_out", [
    (subprocess.TimeoutExpired("ssh cleanup", 30), None, True),
    (ConnectionError("SSH connection lost"), None, False),
    (subprocess.CalledProcessError(255, "ssh cleanup"), 255, False),
])
def test_cleanup_unknown_does_not_invent_success_or_exit_code(case, error, code, timed_out):
    case["cleanup_error"] = error
    rc, report = run(case)
    assert rc == 2 and report["g3_status"] == "incomplete"
    assert report["remote_process_cleanup_complete"] is True
    assert report["remote_cleanup_complete"] is False
    assert report["remote_staging_cleanup"]["status"] == "unknown"
    assert report["remote_staging_cleanup"]["returncode"] == code
    assert report["remote_staging_cleanup"]["timed_out"] is timed_out
    assert report["error_phase"] == "remote_staging_cleanup"


@pytest.mark.parametrize("cleanup_rc", [0, 1, 255])
def test_original_remote_inference_error_survives_cleanup(case, cleanup_rc):
    case.update(runtime_rc=2, runtime_error="RuntimeError: exact original Hailo inference error", cleanup_rc=cleanup_rc)
    rc, report = run(case)
    assert rc == 2 and report["runtime_status"] == "failed" and report["g3_status"] == "incomplete"
    assert report["error"] == report["remote_runtime_error"] == case["runtime_error"]
    assert report["error_phase"] == "runtime"
    assert report["remote_cleanup_complete"] is (cleanup_rc == 0)
    if cleanup_rc:
        assert report["remote_staging_cleanup"]["error"]
    stored = json.loads((case["output"] / "results/runtime_result.json").read_text())
    assert stored["stages"]["cpu_hef"]["inference_count"] == 16


@pytest.mark.parametrize("cleanup_rc", [0, 1])
def test_float_comparison_error_remains_primary(case, cleanup_rc):
    case.update(comparison_error=ValueError("bound float mismatch"), cleanup_rc=cleanup_rc)
    rc, report = run(case)
    assert rc == 2 and report["g3_status"] == "incomplete"
    assert report["error"] == "ValueError: bound float mismatch"
    assert report["error_phase"] == "float_comparison"
    assert report["remote_cleanup_complete"] is (cleanup_rc == 0)


@pytest.mark.parametrize("supervision", [None, [], {}, "{broken", {"cleanup_complete": "true"},
    {"cleanup_complete": 1}, {"cleanup_complete": False},
    {"cleanup_complete": True, "owned_survivors": [12345]}])
def test_unproven_or_inconsistent_process_evidence_never_deletes(case, supervision):
    case["supervision"] = supervision
    rc, report = run(case)
    assert rc == 2 and report["remote_cleanup_complete"] is False
    assert report["remote_staging_cleanup"]["status"] == "blocked_process_cleanup_unproven"
    assert report["remote_staging_path"] == case["mktemp"].strip()
    expected = supervision.get("cleanup_complete") if isinstance(supervision, dict) else None
    assert report["remote_process_cleanup_complete"] is (expected if type(expected) is bool else None)
    assert not cleanup_calls(case)


def test_missing_supervision_never_deletes(case):
    del case["supervision"]
    rc, report = run(case)
    assert rc == 2 and report["remote_process_cleanup_complete"] is None
    assert report["remote_staging_cleanup"]["status"] == "blocked_process_cleanup_unproven"
    assert not cleanup_calls(case)


@pytest.mark.parametrize("candidate", ["/tmp", "/tmp/onnx-v27934-hailo-model-short",
    "/tmp/onnx-v27934-hailo-model-A1b2C3d4E5/../foreign", "/other/owned-A1b2C3d4E5",
    "/tmp/onnx-v27934-hailo-model-A1b2C3d4E5\n/tmp/other", "$(touch /tmp/bad)"])
def test_invalid_mktemp_output_is_never_adopted_or_used(case, candidate):
    case["mktemp"] = candidate
    rc, report = run(case)
    assert rc == 2 and report["remote_staging_path"] is None
    assert report["remote_process_cleanup_complete"] is None
    assert report["remote_staging_cleanup"]["status"] == "not_created"
    assert len(case["calls"]) == 1 and not cleanup_calls(case)


def test_failed_mktemp_does_not_own_even_a_matching_path(case):
    case["mktemp_error"] = subprocess.CalledProcessError(1, "mktemp", output=case["mktemp"])
    rc, report = run(case)
    assert rc == 2 and report["remote_staging_path"] is None
    assert report["remote_staging_cleanup"]["status"] == "not_created"
    assert not cleanup_calls(case)


def test_upload_failure_retains_owned_path_and_primary_error(case):
    case["upload_error"] = RuntimeError("original SCP upload failed")
    rc, report = run(case)
    assert rc == 2 and report["error"] == "RuntimeError: original SCP upload failed"
    assert report["error_phase"] == "upload_stage"
    assert report["remote_staging_path"] == case["mktemp"].strip()
    assert report["remote_staging_cleanup"]["status"] == "blocked_process_cleanup_unproven"
    assert report["remote_process_cleanup_complete"] is None and not cleanup_calls(case)


def test_partial_collection_keeps_remote_detail_and_prior_runtime_error(case):
    case.update(runtime_rc=2, runtime_error="RuntimeError: original device error",
                collect_error=RuntimeError("SCP partial results failure"), cleanup_rc=1)
    rc, report = run(case)
    assert rc == 2 and report["error"] == case["runtime_error"]
    assert report["remote_process_cleanup_complete"] is True
    assert report["remote_staging_cleanup"]["status"] == "failed"


@pytest.mark.parametrize("phase", ["mktemp_error", "upload_error", "runtime_exception", "collect_error", "comparison_error", "cleanup_error"])
def test_keyboard_interrupt_preserves_compact_evidence_and_returns_130(case, phase):
    case[phase] = KeyboardInterrupt("user cancelled")
    rc, report = run(case)
    assert rc == 130 and report["g3_status"] == "incomplete" and report["runtime_status"] == "failed"
    if phase in ("mktemp_error", "upload_error", "runtime_exception"):
        assert report["remote_process_cleanup_complete"] is None and not cleanup_calls(case)
    if phase == "cleanup_error":
        assert report["remote_staging_cleanup"]["status"] == "unknown"
        assert report["remote_staging_cleanup"]["returncode"] is None


def test_interrupt_before_output_has_clear_message_and_no_invented_archive(case, monkeypatch, capsys):
    def interrupt(*args):
        raise KeyboardInterrupt("preflight cancelled")
    monkeypatch.setattr(probe, "hardware_setup", interrupt)
    assert probe.main(case["argv"]) == 130
    assert not case["output"].exists()
    captured = capsys.readouterr()
    assert "RUNTIME_DIAGNOSTIC_NOT_STARTED=KeyboardInterrupt" in captured.err
    assert "EVIDENCE_ZIP=" not in captured.out


def test_zip_write_failure_is_visible_without_replacing_runtime_error(case, monkeypatch, capsys):
    case.update(runtime_rc=2, runtime_error="RuntimeError: preserve device root cause")
    def fail_archive(*args, **kwargs):
        raise OSError("disk full during archive")
    monkeypatch.setattr(probe.zipfile, "ZipFile", fail_archive)
    assert probe.main(case["argv"]) == 2
    report = json.loads((case["output"] / "comparison.json").read_text())
    assert report["error"] == case["runtime_error"]
    assert report["evidence_errors"][0]["phase"] == "evidence_zip"
    captured = capsys.readouterr()
    assert "EVIDENCE_WRITE_FAILED=evidence_zip" in captured.err
    assert "EVIDENCE_ZIP=" not in captured.out


def test_transient_comparison_write_error_is_recorded(case, monkeypatch, capsys):
    case.update(runtime_rc=2, runtime_error="RuntimeError: primary inference failure")
    original = probe.write_json
    failed = []
    def write(path, value):
        if Path(path).name == "comparison.json" and not failed:
            failed.append(True)
            raise OSError("comparison write unavailable")
        return original(path, value)
    monkeypatch.setattr(probe, "write_json", write)
    assert probe.main(case["argv"]) == 2
    report = json.loads((case["output"] / "comparison.json").read_text())
    assert report["error"] == case["runtime_error"]
    assert report["evidence_errors"][0]["phase"] == "comparison_report"
    assert "EVIDENCE_WRITE_FAILED=comparison_report" in capsys.readouterr().err


@pytest.mark.parametrize("runtime_failure", [False, True])
def test_mid_archive_write_failure_never_publishes_partial_pass_evidence(case, monkeypatch, capsys, runtime_failure):
    if runtime_failure:
        case.update(runtime_rc=2, runtime_error="RuntimeError: original inference error")
    original = probe.zipfile.ZipFile
    observed = []
    class BrokenArchive(original):
        def write(self, filename, *args, **kwargs):
            result = super().write(filename, *args, **kwargs)
            if Path(filename).name == "comparison.json":
                observed.append(True)
                raise OSError("write failed after archived comparison")
            return result
    monkeypatch.setattr(probe.zipfile, "ZipFile", BrokenArchive)
    assert probe.main(case["argv"]) == 2
    assert observed == [True]
    report = json.loads((case["output"] / "comparison.json").read_text())
    assert report["runtime_status"] == "failed" and report["g3_status"] == "incomplete"
    assert report["evidence_errors"][0]["phase"] == "evidence_zip"
    if runtime_failure:
        assert report["error"] == case["runtime_error"] and report["error_phase"] == "runtime"
    archive = case["output"].parent / (case["output"].name + "_evidence.zip")
    assert not archive.exists()
    assert not list(archive.parent.glob('.' + archive.name + '.*.tmp'))
    captured = capsys.readouterr()
    assert "EVIDENCE_WRITE_FAILED=evidence_zip" in captured.err
    assert "EVIDENCE_ZIP=" not in captured.out


def test_existing_evidence_archive_is_never_overwritten(case, capsys):
    archive = case["output"].parent / (case["output"].name + "_evidence.zip")
    before = b"previous invocation evidence must remain unchanged"
    archive.write_bytes(before)
    assert probe.main(case["argv"]) == 2
    assert archive.read_bytes() == before
    report = json.loads((case["output"] / "comparison.json").read_text())
    assert report["g3_status"] == "incomplete"
    assert report["evidence_errors"][0]["phase"] == "evidence_zip"
    assert "EVIDENCE_ZIP=" not in capsys.readouterr().out
    assert not list(archive.parent.glob('.' + archive.name + '.*.tmp'))


@pytest.mark.parametrize("entry", ["hailo_model_runtime_probe_v27934.py", "hailo_model_runtime_probe_v280.py"])
def test_old_and_current_starters_share_implementation(entry):
    result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts" / entry), "--help"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "--build-dir" in result.stdout


def test_current_starter_returns_130_when_main_is_interrupted_before_output(case):
    code = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
import hailo_model_runtime_probe_v27934 as collector
def interrupt(*args):
    raise KeyboardInterrupt('starter cancellation fixture')
collector.hardware_setup = interrupt
entry = sys.argv[2]
sys.argv = [entry] + sys.argv[3:]
runpy.run_path(entry, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "scripts"),
        str(ROOT / "scripts/hailo_model_runtime_probe_v280.py"), *case["argv"]],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 130
    assert "RUNTIME_DIAGNOSTIC_NOT_STARTED=KeyboardInterrupt" in result.stderr
    assert "EVIDENCE_ZIP=" not in result.stdout and not case["output"].exists()


def test_existing_transport_finishes_own_process_cleanup_before_reraising_interrupt(tmp_path):
    # Real local transport process, real cancellation and existing group cleanup.
    # Network denial (when used by release validation) is inherited by this child.
    code = """
import os, signal, subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from deepx_full_workflow_smoke_v27930 import run_transport
marker = Path(sys.argv[2])
child = 'import os,time; from pathlib import Path; Path(' + repr(str(marker)) + ').write_text(str(os.getpid())); time.sleep(60)'
def interrupt(*args):
    raise KeyboardInterrupt('transport cancelled')
signal.signal(signal.SIGALRM, interrupt)
signal.setitimer(signal.ITIMER_REAL, 0.5)
try:
    run_transport([sys.executable, '-I', '-B', '-c', child], timeout=30)
except KeyboardInterrupt:
    signal.setitimer(signal.ITIMER_REAL, 0)
    pid = int(marker.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print('OWNED_TRANSPORT_PROCESS_GONE')
        raise SystemExit(130)
    raise SystemExit('transport child survived cancellation')
raise SystemExit('interrupt not observed')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "scripts"),
        str(tmp_path / "owned_pid")], capture_output=True, text=True, timeout=20)
    assert result.returncode == 130 and "OWNED_TRANSPORT_PROCESS_GONE" in result.stdout


@pytest.mark.parametrize("remaining_type", ["directory", "dangling_symlink"])
def test_real_shell_postcondition_rejects_remaining_directory_and_dangling_symlink(case, tmp_path, remaining_type):
    target = Path("/tmp/onnx-v27934-hailo-model-" + uuid.uuid4().hex[:10])
    if remaining_type == "directory":
        target.mkdir(mode=0o700)
    else:
        target.symlink_to(tmp_path / "intentionally_absent_target")
    case.update(mktemp=str(target) + "\n", noop_rm_postcondition=True)
    try:
        rc, report = run(case)
        assert rc == 2 and report["remote_process_cleanup_complete"] is True
        assert report["remote_staging_cleanup"]["status"] == "failed"
        assert report["remote_staging_cleanup"]["returncode"] == 1
        assert report["remote_cleanup_complete"] is False and report["g3_status"] == "incomplete"
        assert target.exists() or target.is_symlink()
    finally:
        if target.is_symlink():
            target.unlink()
        else:
            target.rmdir()


def test_foreign_symlinked_supervision_cannot_supply_process_proof(case, tmp_path):
    foreign = tmp_path / "other_run_supervision.json"
    foreign.write_text('{"cleanup_complete": true}')
    case["foreign_supervision"] = foreign
    rc, report = run(case)
    assert rc == 2 and report["remote_process_cleanup_complete"] is None
    assert report["remote_staging_cleanup"]["status"] == "blocked_process_cleanup_unproven"
    assert not cleanup_calls(case)
    with zipfile.ZipFile(case["output"].parent / (case["output"].name + "_evidence.zip")) as archive:
        assert not any(n.endswith("supervision.json") for n in archive.namelist())
