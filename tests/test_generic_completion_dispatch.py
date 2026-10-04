"""Normal selective CLI with controlled SSH execution leaves and real parsing."""
import copy
import csv
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

import pytest

from onnx_splitpoint_tool.generic_completion_reporting import load_valid_repetitions
from onnx_splitpoint_tool.workflow import generic_completed_task as entry
from onnx_splitpoint_tool.workflow import run_control
from test_generic_completion_reporting import case, repetition, result, write_result


def test_original_environment_uses_bound_successful_attempt(tmp_path):
    base = tmp_path / "primary/models/yolo26m/benchmark_results"
    attempt = base / "attempts/successful_supplement"
    attempt.mkdir(parents=True)
    name = "remote_benchmark_stdout_setup.txt"
    (base / name).write_text("old failed attempt without runtime entry\n")
    (attempt / name).write_text("[remote] python: RUN_PY=/venv/bin/python3 | EXTRA_SITES=/sdk/site-packages\n")
    row = dict(original_run=str(tmp_path / "primary"), model_id="yolo26m", setup_id="setup",
               generic_raw_source=str(attempt / "benchmark_results.json"))
    environment = entry.original_generic_environment(row)
    assert environment["python"] == "/venv/bin/python3"
    assert environment["source"] == str(attempt / name)
    assert environment["environment"] == {"SPLITPOINT_EXTRA_SITES": "/sdk/site-packages"}
    row["generic_raw_source"] = str(tmp_path / "another_model/benchmark_results.json")
    with pytest.raises(ValueError, match="outside its original model"):
        entry.original_generic_environment(row)


class LocalTransport:
    """SSH leaves execute scratch-file operations locally; no SDK or model runs."""
    def __init__(self, *, stop_at=None):
        self.calls = []
        self.stop_at = stop_at

    def run_read_only(self, command, **kwargs):
        if "GENERIC_CONFIGURATION" in command:
            return 0, "CONFIGURATION={}"
        if "GENERIC_RUNTIME_ENVIRONMENT" in command:
            return 0, "RUNTIME_ENVIRONMENT=" + json.dumps({"executable_sha256": "fixture-python-sha",
                "modules": {name: {"available": True} for name in (
                    "numpy", "onnx", "onnxruntime", "tensorrt", "dx_engine", "hailo_platform")}})
        return self.run(command, **kwargs)

    def run(self, command, **kwargs):
        child = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return child.returncode, child.stdout + child.stderr

    def scp_upload(self, source, destination, **kwargs):
        shutil.copyfile(source, destination)
        return 0, ""

    def scp_download(self, source, destination, **kwargs):
        if not Path(source).is_dir():
            return 1, "controlled output directory absent"
        shutil.copytree(source, Path(destination) / Path(source).name)
        return 0, ""

    def run_streaming(self, command, *, on_line, **kwargs):
        argv = shlex.split(command)
        config = json.loads(Path(argv[argv.index("--generic-completion-config") + 1]).read_text())
        index = int(argv[argv.index("--generic-completion-repetition") + 1])
        output = Path(argv[argv.index("--out-dir") + 1]) / "generic_completion_results.json"
        if output.exists():
            # Same parser admission as the real runtime before any next task.
            load_valid_repetitions(output, config)
        self.calls.append((config["identity"], index))
        payload = json.loads(output.read_text()) if output.exists() else result(config, 0)
        if self.stop_at == len(self.calls):
            payload["repetitions"].append({"repetition_index": index, "execution_id": "new-session", "status": "failed", "error": "controlled technical stop"})
            payload["status"] = "failed"
            write_result(output, payload)
            output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            on_line("controlled technical failure")
            return 1
        payload["repetitions"].append(repetition(config, index))
        payload["status"] = "ok" if index == 3 else "partial"
        write_result(output, payload)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        on_line(f"completed={1000} repetition={index}")
        return 0


def prepared_case(tmp_path, *, index=1, setup="test_setup", task="classification"):
    row = case(f"b{index:03d}", setup)
    row.update(row["identity"])
    row["task"] = task
    row["original_run"] = str(tmp_path / "primary")
    row["case_output_relative"] = f"{setup}/{row['model_id']}/{row['case_id']}"
    row["remote_root"] = str(tmp_path / "remote" / setup)
    row["result_path"] = str(tmp_path / "output/results" / row["case_output_relative"] / "generic_completion_results.json")
    row["remote_python_executable"] = "controlled-no-runtime"
    row["staging_files"] = []
    config = {k: copy.deepcopy(row[k]) for k in (
        "identity", "input_binding", "completion_contract", "measurement_boundary", "expected_runtime_artifacts",
        "functional_reference", "original_run", "task")}
    config.update(execution_id="new-session", repetitions=3, warmup_count=100, task_count=1000,
                  original_run_cfg={"native_split_quality_binding": row["remote_root"] + f"/configs/{index}/binding.json",
                                    "streaming_impl": "generic", "throughput_queue_depth": 2})
    row["runtime_config"] = config
    row["native_command_contract"] = {"backend": "hailo8_to_trt",
                                       "artifacts": {"python_executable": {"sha256": "fixture-python-sha"}}}
    return row


def test_run_case_three_processes_download_parser_and_complete_reentry(tmp_path):
    row = prepared_case(tmp_path)
    transport = LocalTransport()
    entry.run_case(row, transport, tmp_path / "output", 1)
    assert [i for _, i in transport.calls] == [1, 2, 3]
    assert len(load_valid_repetitions(row["result_path"], row)) == 3
    before = Path(row["result_path"]).read_bytes()
    entry.run_case(row, transport, tmp_path / "output", 1)
    assert len(transport.calls) == 3
    assert Path(row["result_path"]).read_bytes() == before


def test_failure_downloads_valid_partial_repetitions_and_does_not_retry(tmp_path):
    row = prepared_case(tmp_path)
    transport = LocalTransport(stop_at=2)
    with pytest.raises(RuntimeError, match="stopped"):
        entry.run_case(row, transport, tmp_path / "output", 1)
    assert [i for _, i in transport.calls] == [1, 2]
    assert [r["repetition_index"] for r in load_valid_repetitions(row["result_path"], row)] == [1]
    attempts = json.loads(Path(row["result_path"]).with_name("dispatch_attempts.json").read_text())["attempts"]
    assert attempts[-1]["returncode"] == 1 and attempts[-1]["download"]["result_collected"]
    # Deliberate local replay of the real preserved state, not a second
    # production entry or an automatic retry after an observed failure.
    transport.stop_at = None
    entry.run_case(row, transport, tmp_path / "output", 1)
    assert [i for _, i in transport.calls] == [1, 2, 2, 3]
    assert len(load_valid_repetitions(row["result_path"], row)) == 3


def test_normal_cli_accepts_six_pairs_before_remainder_and_finally_reports_stop(tmp_path, monkeypatch):
    primary = tmp_path / "primary"; primary.mkdir()
    output = tmp_path / "output"; output.mkdir()
    setups = ["setup_a", "setup_b", "setup_c"]
    rows = [prepared_case(tmp_path, index=index + 1, setup=setups[index % 3],
             task="classification" if index < 3 else "detection") for index in range(192)]
    starts = [row["case_key"] for row in rows[:6]]
    targets = [{"id": setup, "remote": {"host": f"192.0.2.{index + 1}", "user": "fixture"}} for index, setup in enumerate(setups)]
    entry.write_json(primary / "run_manifest.json", {"hardware_targets": targets})
    entry.write_json(primary / "reports/native_host_telemetry_summary.json", {"captures": []})
    scope = tmp_path / "scope.csv"
    with scope.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]["identity"]))
        writer.writeheader(); writer.writerows(r["identity"] for r in rows)
    with tarfile.open(output / "runner-bundle.tar.gz", "w:gz") as archive:
        data = b"controlled runtime leaf\n"
        info = tarfile.TarInfo("README.txt"); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    plan = {"rows": rows, "start_case_keys": starts, "scope_source": str(scope),
            "original_run": str(primary), "output_root": str(output), "execution_id": "new-session"}
    plan_path = output / "plan/execution-plan.json"; entry.write_json(plan_path, plan)
    transport = LocalTransport(stop_at=19)
    monkeypatch.setattr(entry, "transport_for", lambda *args, **kwargs: transport)
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: tmp_path / "test-locks/platform.lock")
    assert entry.main(["--execute", str(plan_path)]) == 1
    accepted = json.loads((output / "start-cases-accepted.json").read_text())
    assert accepted["valid_repetitions"] == 18 and accepted["case_keys"] == starts
    assert len(transport.calls) == 19
    assert len({tuple(identity.values()) for identity, _ in transport.calls[:18]}) == 6
    terminal = json.loads((output / "terminal-status.json").read_text())
    assert terminal["status"] == "stopped" and not terminal["automatic_restart"]
    assert terminal["summary"]["status_counts"]["completed"] == 6
    assert terminal["summary"]["valid_repetitions"] == 18
    assert (output / "completion-diagnostics.tar.gz").is_file()
    # The durable once-only entry remains a hard stop, even with a valid plan.
    protected = {path: path.read_bytes() for path in output.rglob("*")
                 if path.is_file() and path.suffix in {".json", ".csv", ".txt", ".gz"}}
    assert entry.main(["--execute", str(plan_path)]) == 1
    assert len(transport.calls) == 19
    assert all(path.read_bytes() == content for path, content in protected.items())
    with run_control.EvaluationRunLock(out_root=output.parent, run_dir=output,
            owner={"run_id": primary.name, "session_id": "existing-owned-campaign"}):
        assert entry.main(["--execute", str(plan_path)]) == 1
    assert len(transport.calls) == 19
    assert all(path.read_bytes() == content for path, content in protected.items())


def test_scope_drift_and_primary_output_are_rejected_before_transport(tmp_path):
    row = prepared_case(tmp_path)
    scope = tmp_path / "scope.csv"; scope.write_text("model_id\nresnet50\n")
    with pytest.raises(ValueError, match="exactly 192"):
        entry.validate_plan({"rows": [row]}, scope, tmp_path / "primary")
    with pytest.raises(ValueError, match="outside"):
        entry.outside_primary(tmp_path / "primary/reports", tmp_path / "primary")


def test_partial_reentry_restages_exact_samples_after_remote_workspace_loss(tmp_path):
    row = prepared_case(tmp_path)
    transport = LocalTransport(stop_at=2)
    with pytest.raises(RuntimeError, match="stopped"):
        entry.run_case(row, transport, tmp_path / "output", 1)
    preserved = copy.deepcopy(load_valid_repetitions(row["result_path"], row)[0])
    # Only the test's controlled remote-output fixture is removed.
    shutil.rmtree(Path(row["remote_root"]) / "results")
    transport.stop_at = None
    entry.run_case(row, transport, tmp_path / "output", 1)
    valid = load_valid_repetitions(row["result_path"], row)
    assert len(valid) == 3 and valid[0] == preserved
    assert [index for _, index in transport.calls] == [1, 2, 2, 3]


@pytest.mark.parametrize("changed", ["runtime_identity", "result_path"])
def test_execution_plan_rejects_inner_identity_drift_and_result_path_escape(tmp_path, changed):
    rows = [prepared_case(tmp_path, index=index + 1) for index in range(192)]
    scope = tmp_path / "scope.csv"
    with scope.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]["identity"]))
        writer.writeheader(); writer.writerows(row["identity"] for row in rows)
    primary = tmp_path / "primary"
    plan = {"rows": rows, "original_run": str(primary), "output_root": str(tmp_path / "output"),
            "execution_id": "new-session"}
    assert entry.validate_plan(plan, scope, primary) == rows
    if changed == "runtime_identity":
        rows[0]["runtime_config"]["identity"]["case_id"] = "b999"
        reason = "runtime identity"
    else:
        rows[0]["result_path"] = str(primary / "reports/generic_completion_results.json")
        reason = "result path"
    with pytest.raises(ValueError, match=reason):
        entry.validate_plan(plan, scope, primary)
