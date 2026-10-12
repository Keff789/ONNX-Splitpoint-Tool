"""Saved GUI configuration -> real backend requests, with no hardware dispatch."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import shlex
import sys
from types import MethodType, SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool.workflow import run_evaluation
from onnx_splitpoint_tool.workflow.execution_binding import _remote_args_from_options
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.resources.remote_scripts import run_evalrun_native_producer_variants as coordinator
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
from onnx_splitpoint_tool.run_modes import default_run_modes_config


class Value:
    def __init__(self, value=""):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class Text:
    def configure(self, **kwargs): pass
    def delete(self, *args): pass
    def insert(self, *args): pass


class BeforeDispatch(BaseException):
    pass


def profile_file(tmp_path, benchmark_warmup):
    repo = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load((repo / "onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml").read_text())
    payload["name"] = "tool_parity_fixture"
    payload["model_suite"]["primary"] = payload["model_suite"]["primary"][:1]
    payload["model_suite"]["reserve"] = []
    registry = yaml.safe_load((repo / "onnx_splitpoint_tool/resources/hardware_setups.example.yaml").read_text())
    for index, setup in enumerate(registry["hardware_setups"], 1):
        setup["host"]["address"] = f"192.0.2.{index}"
        setup["energy"]["urecs_address"] = f"192.0.2.{index + 10}"
    registry_path = tmp_path / "user hardware.yaml"
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False))
    payload["hardware"] = {"setups_file": str(registry_path), "selected_setups": ["orin_nx_hailo8_01"]}
    mode = default_run_modes_config()["modes"]["standard"]
    mode["runtime"]["benchmark"].update(warmup=benchmark_warmup, runs=13)
    mode["runtime"]["parallel"].update(remote_setups=False, max_setups=1)
    payload["execution_preset"] = {"id": "standard", "follow_tool_config": False, "snapshot": mode}
    payload["benchmark_execution"] = {"warmup": benchmark_warmup, "runs": 13}
    payload["remote"] = {"enabled": True, "warmup": 0, "iters": 19,
                         "parallel_remote_setups": False, "max_parallel_setups": 1,
                         "throughput_warmup_frames": 0, "throughput_queue_depth": 4}
    payload["native_producers"] = {
        "enabled": True, "frames": 217, "warmup": 0, "repetitions": 2,
        "queue_depth": 5, "inflight": 4,
        "variants": [{"id": "fixture", "case_map": {"resnet50": ["b001"]}}],
        "energy": {"enabled": True, "mode": "measure", "duration_s": 17,
                   "repeat_override": 2, "physical_scope": "FS", "window_label": "command",
                   "task_budget": {"enabled": True, "max_retries": 0, "max_transport_failures": 2}},
    }
    path = tmp_path / "user profile.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def backend_request(options, output):
    runner = EvaluationWorkflowRunner(options)
    runner._load_profile()
    runner.run_dir, runner.run_id = output, output.name
    output.mkdir(parents=True, exist_ok=True)
    runner.session_id = "tool-parity-controller"
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(run_id=output.name, session_id=runner.session_id),
        journal_dir=output / "jobs/remote_process_leases" / runner.session_id)
    (output / "run_manifest.json").write_text(json.dumps({"run_id": output.name, "resume_contract": {}}))
    remote = asdict(_remote_args_from_options(options, runner.profile_payload,
                                              model_task="classification", model_id="resnet50"))
    steps = runner._stage_run_native_producers_steps()
    try:
        next(steps)  # The normal stage writes its request before yielding any physical work.
        request = json.loads((output / "reports/native_producer_stage_config.json").read_text())
    finally:
        steps.close()
    energy = request["energy"]
    argv = coordinator._common_energy_args(request, run_dir=output,
        duration_s=energy["duration_s"], timeout_s=energy.get("timeout", 900))
    argv += ["--runs", str(coordinator._energy_repeat_count(request, energy))]
    argv += coordinator._energy_window_ab_cli_args(request, energy)
    contract = coordinator._energy_contract_context(request, energy, run_dir=output, reports=output / "reports")
    return {"remote": remote, "native": request, "energy_argv": argv, "energy_contract": contract,
            "hardware": runner.profile_payload["hardware"], "models": runner.profile_payload["model_suite"]}


@pytest.mark.parametrize("benchmark_warmup", [0, 7])
def test_actual_gui_queue_and_profile_cli_generate_identical_requests(tmp_path, monkeypatch, benchmark_warmup):
    import matplotlib
    monkeypatch.setattr(matplotlib, "use", lambda *args, **kwargs: None)
    gui = importlib.import_module("onnx_splitpoint_tool.gui.app")
    panel = gui.panel_evaluation_workflow
    profile = profile_file(tmp_path, benchmark_warmup)
    out = tmp_path / "user output"
    app = SimpleNamespace(var_eval_workflow_profile=Value(str(profile)),
        var_eval_workflow_out_root=Value(str(out)), var_eval_workflow_status=Value(),
        _eval_workflow_default_out_root=lambda: out)
    for name in ("_eval_workflow_load_profile_payload", "_eval_workflow_snapshot_options",
                 "_eval_workflow_bool", "_eval_workflow_remote_host_payload_from_profile"):
        setattr(app, name, MethodType(getattr(gui.SplitPointAnalyserGUI, name), app))
    lines, snapshot = panel._profile_summary_payload(str(profile))
    assert snapshot, lines
    assert panel._commit_profile_summary(app, Text(), "\n".join(lines), snapshot)
    captured = {}
    resolve = app._eval_workflow_snapshot_options
    def gui_options(**kwargs):
        captured["gui"] = resolve(**kwargs)
        return captured["gui"]
    app._eval_workflow_snapshot_options = gui_options
    real_mkdir = Path.mkdir
    def stop_before_dispatch(path, *args, **kwargs):
        if path == out:
            raise BeforeDispatch()
        return real_mkdir(path, *args, **kwargs)
    with monkeypatch.context() as boundary:
        boundary.setattr(Path, "mkdir", stop_before_dispatch)
        boundary.setattr(gui.messagebox, "askyesno", lambda *args, **kwargs: True)
        boundary.setattr(gui.messagebox, "showwarning", lambda *args, **kwargs: pytest.fail(str(args)))
        with pytest.raises(BeforeDispatch):
            gui.SplitPointAnalyserGUI._queue_evaluation_workflow(app, resume=False)
    class RunnerBoundary:
        def __init__(self, options, **kwargs):
            captured["cli"] = options
        def run(self):
            raise BeforeDispatch()
    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", RunnerBoundary)
    with pytest.raises(BeforeDispatch):
        run_evaluation.main(["--profile-driven", "--profile", str(profile), "--out", str(out), "--json"])
    request_root = tmp_path / "backend request"
    gui_request = backend_request(captured["gui"], request_root)
    cli_request = backend_request(captured["cli"], request_root)
    (tmp_path / "gui_request.json").write_text(json.dumps(gui_request, indent=2, default=str))
    (tmp_path / "cli_request.json").write_text(json.dumps(cli_request, indent=2, default=str))
    # Resolution wall-clock metadata is not a measurement or executor setting.
    for request in (gui_request, cli_request):
        request["native"]["_workflow_context"]["execution_preset"].pop("resolved_at", None)
    assert gui_request == cli_request
    assert captured["gui"].benchmark_warmup == benchmark_warmup
    # Run-mode budgets are authoritative over legacy remote profile fields.
    assert gui_request["remote"]["warmup"] == benchmark_warmup
    assert gui_request["remote"]["throughput_warmup_frames"] == 6
    assert gui_request["remote"]["throughput_queue_depth"] == 2
    assert gui_request["remote"]["iters"] == 13
    native = gui_request["native"]
    assert tuple(native[k] for k in ("frames", "warmup", "repetitions", "queue_depth", "inflight")) == (217, 0, 2, 5, 4)
    assert native["energy"]["task_budget"]["max_retries"] == 0
    argv = gui_request["energy_argv"]
    assert float(argv[argv.index("--duration-s") + 1]) == 17.0
    assert argv[argv.index("--runs") + 1] == "2"
    assert argv[argv.index("--campaign-max-retries") + 1] == "0"
    assert gui_request["energy_contract"]["physical_scope"] == "FS"
    assert gui_request["energy_contract"]["window_label"] == "command"
    preview = gui.SplitPointAnalyserGUI._eval_workflow_command_preview(app, captured["gui"])
    preview_argv = shlex.split(preview)
    assert preview_argv[:3] == [sys.executable, "-m", "onnx_splitpoint_tool.workflow.run_evaluation"]
    with pytest.raises(BeforeDispatch):
        run_evaluation.main(preview_argv[3:])
    (tmp_path / "GUI_CLI_REQUEST_PARITY.json").write_text(json.dumps(gui_request, indent=2, default=str))


def test_legacy_cli_preserves_explicit_zero_benchmark_warmup(tmp_path, monkeypatch):
    captured = {}
    class Boundary:
        def __init__(self, options, **kwargs): captured["options"] = options
        def run(self): raise BeforeDispatch()
    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", Boundary)
    with pytest.raises(BeforeDispatch):
        run_evaluation.main(["--profile", "unused", "--out", str(tmp_path), "--benchmark-warmup", "0"])
    assert captured["options"].benchmark_warmup == 0


def test_explicit_remote_zero_survives_actual_request(tmp_path, monkeypatch):
    captured = {}
    class Boundary:
        def __init__(self, options, **kwargs): captured["options"] = options
        def run(self): raise BeforeDispatch()
    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", Boundary)
    with pytest.raises(BeforeDispatch):
        run_evaluation.main(["--profile", "unused", "--out", str(tmp_path),
                             "--benchmark-warmup", "7", "--remote-warmup", "0",
                             "--remote-throughput-warmup-frames", "0"])
    request = _remote_args_from_options(captured["options"], {})
    assert request.warmup == 0
    assert request.throughput_warmup_frames == 0
