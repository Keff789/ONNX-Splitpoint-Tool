"""R9E: real builder/parent transitions with an explicitly substituted SDK.

No proprietary SDK, accelerator, GUI workflow or compilation is used here.
"""
from __future__ import annotations

import builtins
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool.build_evidence import (
    ABORTED_UNKNOWN, COMPILE_INFEASIBLE, TRANSIENT_INFRASTRUCTURE,
    classify_build_outcome,
)
from onnx_splitpoint_tool.gui.hailo_diagnostics import (
    collect_hailo_diagnostics, format_hailo_diagnostics_short_lines,
    format_hailo_diagnostics_text, load_hailo_result_json,
)
from test_v27934_hailo_backend import managed  # shared synthetic SDK boundary
from test_v27922_build_evidence_store import _key
from test_v283_r9a_backfill import controller


def _build(managed, out, **kwargs):
    return backend.hailo_build_hef_auto(
        **managed, outdir=out, backend="venv", force=False,
        publish_artifacts=False, compute_device="cpu", **kwargs,
    )


@pytest.mark.parametrize("net_name,boundary", [("renamed_network", "cut_left"), ("other_graph", "split_right")])
def test_timeout_retains_current_hars_and_real_compile_phase(managed, tmp_path, monkeypatch, net_name, boundary):
    monkeypatch.setenv("TEST_SDK_FAIL", "timeout")
    managed.update(net_name=net_name, end_node_names=[boundary])
    out = tmp_path / boundary
    started = time.monotonic()
    result = _build(managed, out, wsl_timeout_s=4)
    assert 4 <= time.monotonic() - started < 15
    assert not result.ok and result.timed_out and result.failure_kind == "timeout"
    assert result.details["hard_timeout_s"] == 4
    assert result.last_stage == "compile", result.error
    assert "Last active stage: compile." in result.error
    assert result.details["process_summary"]["last_stage"] == "compile_prep"
    assert result.details["process_cleanup"]["remaining_process_count"] == 0
    assert result.details["phase_events"][-1]["state"] == "started"
    assert any(e["phase"] == "optimize" and e["state"] == "completed" for e in result.details["phase_events"])
    assert Path(result.parsed_har_path).is_file() and Path(result.quant_har_path).is_file()
    assert result.hef_path is None and not (out / "compiled.hef").exists()
    assert result.details["cache_hit"] is False
    assert result.details["har_artifacts"]["quantized"]["sdk_loadability"] == "not_checked"
    assert classify_build_outcome(asdict(result)) == TRANSIENT_INFRASTRUCTURE

    # The ordinary persisted result and GUI formatter keep both axes visible.
    payload = json.loads((out / "hailo_hef_build_result.json").read_text())
    assert payload["quant_har_path"] == result.quant_har_path
    entry = load_hailo_result_json(out / "hailo_hef_build_result.json")
    rendered = format_hailo_diagnostics_text(entry)
    short = "\n".join(format_hailo_diagnostics_short_lines(entry))
    assert entry["status"] == "TIMEOUT" and "hef" not in entry["paths"]
    assert "optimize: completed" in rendered and "compile: started" in rendered
    assert "Quant HAR:" in rendered and "resume contract: not validated" in rendered
    assert "status=TIMEOUT" in short and "saved intermediate; not a HEF" in short
    terminal = json.loads((out / "hailo_attempt_receipts" / "terminal_attempt.json").read_text())
    assert terminal["semantic_status"] == "timeout"
    assert terminal["result_summary"]["quant_har_path"] == result.quant_har_path
    ctl, _, _ = controller()
    ctl.observe(2, ctl.begin(2), {"hailo8": {"part1_build": asdict(result)}})
    assert ctl.active() == [] and ctl.state["contracts"][0]["status"] == "infrastructure_blocked"
    assert ctl.state["contracts"][0]["selected_case_ids"] == ["b002"]


@pytest.mark.parametrize("failure,quantized", [("optimize", False), ("empty", True), ("compile", True), ("exit", True)])
def test_exception_preserves_only_completed_intermediates(managed, tmp_path, monkeypatch, failure, quantized):
    monkeypatch.setenv("TEST_SDK_FAIL", failure)
    result = _build(managed, tmp_path / "failed")
    assert not result.ok and not result.timed_out and result.hef_path is None
    assert Path(result.parsed_har_path).is_file()
    assert bool(result.quant_har_path) is quantized
    assert result.last_stage == ("compile" if quantized else "optimize")
    assert result.details["phase_events"][-1]["state"] == ("started" if failure == "exit" else "failed")
    if failure == "exit":
        assert result.failure_kind == "missing_structured_result"
    assert collect_hailo_diagnostics(result)["status"] == "FAILED"
    if failure == "compile":
        assert classify_build_outcome(asdict(result)) == COMPILE_INFEASIBLE


@pytest.mark.parametrize("fault,status", [("missing", "save_failed"), ("empty", "save_failed"),
                                          ("corrupt", "invalid"), ("foreign", "foreign"),
                                          ("parsed", "wrong_state")])
def test_invalid_or_absent_quantized_har_never_becomes_checkpoint(managed, tmp_path, monkeypatch, fault, status):
    monkeypatch.setenv("TEST_SDK_FAIL", "compile")
    monkeypatch.setenv("TEST_HAR_FAULT", fault)
    result = _build(managed, tmp_path / "failed")
    assert not result.ok and result.quant_har_path is None and result.hef_path is None
    assert result.parsed_har_path
    assert result.details["har_artifacts"]["quantized"]["status"] == status
    if fault in {"missing", "empty"}:
        assert result.details["har_artifacts"]["quantized"]["reason"] == fault + "_saved_har"


def test_old_quantized_har_is_not_adopted_after_new_optimization_failure(managed, tmp_path, monkeypatch):
    out = tmp_path / "same_directory"
    monkeypatch.setenv("TEST_SDK_FAIL", "compile")
    old = _build(managed, out)
    old_bytes = Path(old.quant_har_path).read_bytes()
    monkeypatch.setenv("TEST_SDK_FAIL", "optimize")
    new = _build(managed, out)
    assert new.parsed_har_path and new.quant_har_path is None
    assert new.details["har_artifacts"]["quantized"]["status"] == "not_saved"
    assert (out / "quantized.har").read_bytes() == old_bytes


@pytest.fixture
def saved(managed, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_SDK_FAIL", "compile")
    out = tmp_path / "saved"
    result = _build(managed, out)
    assert result.quant_har_path
    progress = json.loads((out / "hailo_build_phases.json").read_text())
    return managed, out, result, progress


@pytest.mark.parametrize("mutation", ["old_phase", "foreign_source", "foreign_pid", "missing_identity",
                                      "missing", "replaced", "symlink", "empty", "truncated", "unproven_optimize"])
def test_saved_evidence_is_bound_to_generation_and_file(saved, tmp_path, mutation):
    managed, out, _, progress = saved
    har = out / "quantized.har"
    started = progress["events"][0]["timestamp"] - 1
    pid = progress["events"][0]["pid"]
    if mutation == "old_phase":
        started = time.time() + 1
    elif mutation == "foreign_source":
        progress["identity"]["source_onnx_sha256"] = "f" * 64
    elif mutation == "foreign_pid":
        pid += 1
    elif mutation == "missing_identity":
        progress.pop("identity")
    elif mutation == "missing":
        har.unlink()
    elif mutation == "replaced":
        other = out / "replacement.har"
        other.write_bytes(har.read_bytes())
        os.replace(other, har)
    elif mutation == "symlink":
        foreign = tmp_path / "foreign.har"
        har.rename(foreign)
        har.symlink_to(foreign)
    elif mutation in {"empty", "truncated"}:
        data = har.read_bytes()
        har.write_bytes(b"" if mutation == "empty" else data[:700])
        # Also reject a damaged container even if a save claimed these bytes.
        progress["har_artifacts"]["quantized"]["file_identity"] = backend._hailo_har_file_identity(har)
    elif mutation == "unproven_optimize":
        progress["events"] = [e for e in progress["events"] if not (e["phase"] == "optimize" and e["state"] == "completed")]
    (out / "hailo_build_phases.json").write_text(json.dumps(progress))
    result = backend.HailoHefBuildResult(ok=False, elapsed_s=1, net_name=managed["net_name"], hw_arch=managed["hw_arch"], timed_out=True)
    backend._retain_hailo_build_progress(result, {**managed, "outdir": out}, started_at=started, expected_pid=pid)
    assert result.quant_har_path is None and result.hef_path is None and not result.ok


def test_recovery_never_loads_sdk_or_claims_resume_contract(saved, monkeypatch):
    managed, out, result, progress = saved
    original_import = builtins.__import__
    def no_sdk(name, *args, **kwargs):
        if name.startswith(("hailo_sdk", "hailo_platform")):
            pytest.fail("HAR evidence recovery must not load a proprietary SDK")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_sdk)
    backend._retain_hailo_build_progress(result, {**managed, "outdir": out},
        started_at=progress["events"][0]["timestamp"] - 1, expected_pid=progress["events"][0]["pid"])
    assert result.quant_har_path
    assert result.details["har_artifacts"]["quantized"]["resume_contract"] == "not_validated"


def test_missing_parsed_har_does_not_discard_saved_quantized_har(saved):
    managed, out, result, progress = saved
    (out / "parsed.har").unlink()
    backend._retain_hailo_build_progress(result, {**managed, "outdir": out},
        started_at=progress["events"][0]["timestamp"] - 1, expected_pid=progress["events"][0]["pid"])
    assert result.parsed_har_path is None and result.quant_har_path
    assert result.details["har_artifacts"]["parsed"]["status"] == "missing_or_foreign"


def test_normal_gui_diagnostic_handlers_replay_saved_failure_without_window_or_workflow(saved, monkeypatch):
    from types import MethodType, SimpleNamespace
    from onnx_splitpoint_tool.gui import app
    from onnx_splitpoint_tool.gui_app import SplitPointAnalyserGUI as LegacyGUI
    _, out, result, _ = saved
    windows, log_lines = [], []
    gui = SimpleNamespace(model_path="", _hailo_gui_diag_history=[],
                          _popup_text=lambda title, body, **kwargs: windows.append((title, body)))
    gui._hailo_gui_record_diagnostics = MethodType(app.SplitPointAnalyserGUI._hailo_gui_record_diagnostics, gui)
    monkeypatch.setattr(app.filedialog, "askopenfilename", lambda **kwargs: str(out / "hailo_hef_build_result.json"))
    monkeypatch.setattr(app.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    LegacyGUI._hailo_publish_gui_diagnostics(gui, "Saved failed build", result, log_cb=log_lines.append)
    app.SplitPointAnalyserGUI._hailo_gui_open_result_json(gui)
    app.SplitPointAnalyserGUI._hailo_gui_show_last_diagnostics(gui)
    assert len(windows) == 2 and len(gui._hailo_gui_diag_history) == 2
    assert any("status=FAILED" in line for line in log_lines)
    for _, body in windows:
        assert "Status: FAILED" in body and "Status: OK" not in body
        assert "Quant HAR:" in body and "compile: failed" in body
    (out / "gui_diagnostic_replay.txt").write_text("\n\n".join(body for _, body in windows))


@pytest.mark.parametrize("rc", [-15, -9, 124, 143])
def test_supervisor_timeout_stays_transient_despite_cleanup_signal_or_mapper_prose(tmp_path, rc):
    from onnx_splitpoint_tool.build_evidence_store import BuildEvidenceStore
    row = {"ok": False, "timed_out": True, "returncode": rc,
           "last_stage": "compile", "error": "Agent infeasible; SIGTERM after deadline"}
    state = classify_build_outcome(row)
    assert state == TRANSIENT_INFRASTRUCTURE
    store = BuildEvidenceStore(tmp_path / "evidence")
    store.record(_key(), state, evidence_origin={"source": "controlled_timeout"})
    decision = store.lookup(_key())
    assert decision.status == "MISS" and not decision.reusable


def test_real_infeasibility_remains_reusable_and_cancellation_remains_separate(tmp_path):
    from onnx_splitpoint_tool.build_evidence_store import BuildEvidenceStore
    row = {"ok": False, "returncode": 3, "last_stage": "compile", "error": "No successful assignments: Agent infeasible"}
    state = classify_build_outcome(row)
    assert state == COMPILE_INFEASIBLE
    store = BuildEvidenceStore(tmp_path / "evidence")
    store.record(_key(), state, evidence_origin={"source": "controlled_compile_exception"})
    assert store.lookup(_key()).reusable
    assert classify_build_outcome({**row, "returncode": -15}) == ABORTED_UNKNOWN


def test_explicit_small_timeout_policy_is_not_enlarged(monkeypatch):
    for name in ("ONNX_SPLITPOINT_HAILO_HEF_TIMEOUT_S", "OSP_HAILO_HARD_TIMEOUT_S",
                 "ONNX_SPLITPOINT_HAILO_HEF_IDLE_TIMEOUT_S", "OSP_HAILO_IDLE_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    assert backend._resolve_hef_timeout_policy(2) == (2, None)
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_HEF_TIMEOUT_S", "1")
    assert backend._resolve_hef_timeout_policy(1750) == (1, None)
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_HEF_TIMEOUT_S", "1750")
    assert backend._resolve_hef_timeout_policy(2) == (1750, None)
