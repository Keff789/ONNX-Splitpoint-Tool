"""R9G: Full quality containers and completed local transport captures."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.process_control import ProcessTreeRegistry, bind_process_registry
from onnx_splitpoint_tool.remote import ssh_transport


@pytest.fixture
def suite(tmp_path):
    script = Path(write_benchmark_suite_script(tmp_path / "suite"))
    spec = importlib.util.spec_from_file_location("r9g_dispatch_suite", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = [{"case_dir": "b002", "boundary": 2}, {"case_dir": "b007", "boundary": 7}]
    run = {
        "id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt",
        "backend": "tensorrt", "variants": ["full", "composed"],
        "setup_id": "producer_c", "case_ids": ["b007"],
        "backend_selection_contracts": [{
            "run_id": "ort_tensorrt", "setup_id": "producer_c",
            "selected_case_ids": ["b007"],
        }],
    }
    (script.parent / "benchmark_set.json").write_text(json.dumps({"model_id": "model_a", "cases": cases}))
    (script.parent / "benchmark_plan.json").write_text(json.dumps({"runs": [run]}))
    return module, script, cases, run


@pytest.mark.parametrize("setup", ["producer_a", "producer_b", "producer_c"])
@pytest.mark.parametrize("case_filter", [None, "b007"])
def test_full_quality_uses_model_container_without_borrowing_split_scope(suite, monkeypatch, setup, case_filter):
    module, script, cases, run = suite
    argv = [str(script), "--run-ids", "ort_tensorrt", "--no-plot",
            "--quality-evidence-eval-id", "eval_a", "--quality-evidence-model-id", "model_a",
            "--quality-evidence-setup-id", setup, "--quality-evidence-endpoint-id", "trt_at_" + setup]
    if setup != "producer_c":
        argv += ["--quality-only-run-ids", "ort_tensorrt"]
    if case_filter:
        argv += ["--case", case_filter]
    monkeypatch.setattr(sys, "argv", argv)
    observed = []

    class ReachedRuntimeBoundary(Exception):
        pass

    def inspect_companion(**kwargs):
        observed.append(kwargs)
        # Stop before any engine/runtime operation; do not fabricate evidence.
        raise ReachedRuntimeBoundary

    monkeypatch.setattr(module, "_run_native_full_trt_quality_companion", inspect_companion)
    with pytest.raises(ReachedRuntimeBoundary):
        module.main()
    assert len(observed) == 1
    call = observed[0]
    assert call["run_cases"] == ([cases[1]] if case_filter else [cases[0]])
    assert call["args"].quality_evidence_setup_id == setup
    assert call["run"]["_native_full_trt_quality_companion_id"] == "trt_at_" + setup
    assert call["run"]["backend_selection_contracts"] == run["backend_selection_contracts"]
    # Split/performance ownership remains exactly setup-local and fail-closed.
    assert module._cases_for_run(cases, call["run"], setup_id=setup) == (
        [cases[1]] if setup == "producer_c" else []
    )


@pytest.mark.parametrize("invalid", ["missing_case", "endpoint_conflict"])
def test_full_quality_still_rejects_invalid_explicit_scope(suite, monkeypatch, invalid):
    module, script, _cases, run = suite
    argv = [str(script), "--run-ids", "ort_tensorrt", "--no-plot",
            "--quality-evidence-eval-id", "eval_a", "--quality-evidence-model-id", "model_a",
            "--quality-evidence-setup-id", "producer_a", "--quality-evidence-endpoint-id", "trt_at_a",
            "--quality-only-run-ids", "ort_tensorrt"]
    if invalid == "missing_case":
        argv += ["--case", "b999"]
    else:
        run.update(quality_canary_endpoint_ids=["different_endpoint"], quality_canary_setup_ids=["producer_a"])
        (script.parent / "benchmark_plan.json").write_text(json.dumps({"runs": [run]}))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(module, "_run_native_full_trt_quality_companion", lambda **kw: pytest.fail("Runtime must not start"))
    assert module.main() == 2


@pytest.mark.parametrize("exit_code", [0, 7])
def test_capture_waits_for_bounded_reader_completion_before_classifying(exit_code, monkeypatch):
    original_thread = threading.Thread

    class DelayedReader(original_thread):
        def run(self):
            if self.name.startswith("ssh-capture-"):
                # A completed child can precede its reader under host load.
                time.sleep(0.6)
            super().run()

    registry = ProcessTreeRegistry()
    monkeypatch.setattr(ssh_transport.threading, "Thread", DelayedReader)
    transport = ssh_transport.SSHTransport(ssh_transport.HostConfig(id="local", label="local", host="unused.invalid"))
    diagnostic = {}
    with bind_process_registry(registry):
        rc, output = transport._run_capture(
            [sys.executable, "-c", f"print('/controlled/path'); raise SystemExit({exit_code})"],
            timeout=5, diagnostics=diagnostic,
        )
    assert output == "/controlled/path\n"
    assert diagnostic["local_exit_code"] == exit_code
    assert diagnostic["reader_finished"] and diagnostic["local_completion_proven"]
    assert diagnostic["local_tree_survivors"] == []
    assert rc == exit_code
    assert diagnostic["phase"] == "completed"
    registry.assert_quiescent()


def test_capture_keeps_inherited_pipe_failure_and_cleans_owned_child(tmp_path):
    registry = ProcessTreeRegistry()
    transport = ssh_transport.SSHTransport(ssh_transport.HostConfig(id="local", label="local", host="unused.invalid"))
    marker = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); "
        f"Path({str(marker)!r}).write_text(str(p.pid)); "
        "print('root exiting',flush=True); time.sleep(.3)"
    )
    diagnostic = {}
    try:
        with bind_process_registry(registry):
            rc, output = transport._run_capture([sys.executable, "-c", code], timeout=5, diagnostics=diagnostic)
        assert rc == 70
        assert diagnostic["phase"] == "pipe_drain_incomplete"
        assert output == "root exiting\n"
        assert diagnostic["local_completion_proven"]
        registry.assert_quiescent()
        child = Path(f"/proc/{int(marker.read_text())}/stat")
        assert not child.exists() or child.read_text().split(") ", 1)[1].startswith("Z ")
    finally:
        registry.terminate_all(grace_s=.2)
