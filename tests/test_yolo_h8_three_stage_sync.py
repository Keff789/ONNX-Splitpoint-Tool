"""Execute the normal H8 sync closure with a local filesystem transport only."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

import pytest

from tests.test_v2791_concurrent_three_stage_normal_runner import _args

ROOT = Path(__file__).resolve().parents[1]
RESOURCE = Path("onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7")
HELPER = "native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"


def load_updater(packaged):
    path = ROOT / ("onnx_splitpoint_tool/resources/remote_scripts" if packaged else "scripts") / "update_evalset_native_producers.py"
    spec = importlib.util.spec_from_file_location("h8_sync_" + str(packaged), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def h8_sync_block(source):
    candidates = [node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If) and ast.unparse(node.test) == "b == 'hailo8'"
        and any("native_fifo_smoke_matrix.py" in ast.unparse(child) for child in node.body)]
    assert len(candidates) == 1
    return ast.Module(body=[candidates[0]], type_ignores=[])


def local_transfer(module, monkeypatch, remote, *, damage=""):
    copied = []

    def transport(command, **kwargs):
        if command[0] == "rsync":
            assert command[1:3] == ["-a", "--checksum"]
            source = Path(command[3]); host, destination = command[4].split(":", 1)
            assert host == "local-fixture" and source.is_file()
            target = Path(destination)
            assert target.is_relative_to(remote)
            shutil.copy2(source, target); copied.append((source, target))
            if damage == "helper" and target.name == HELPER:
                target.write_bytes(target.read_bytes() + b"changed")
            if damage == "resource" and target.name == "fast_decode_runtime.py":
                target.unlink()
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        assert command[0] == "ssh" and command[-2] == "local-fixture"
        operation = shlex.split(command[-1])
        if operation[:2] == ["mkdir", "-p"]:
            target = Path(operation[2]); assert target.is_relative_to(remote)
            target.mkdir(parents=True, exist_ok=True)
            return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}
        assert operation[:2] == ["python", "-c"]
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output): exec(operation[2], {})
        except SystemExit as error:
            return {"rc": int(error.code or 0), "stdout_tail": output.getvalue(), "stderr_tail": ""}
        except OSError as error:
            return {"rc": 1, "stdout_tail": output.getvalue(), "stderr_tail": str(error)}
        raise AssertionError("normal transfer verifier must return a status")

    monkeypatch.setattr(module, "_run", transport)
    return copied


def execute_block(module, source, remote):
    namespace = {**vars(module), "b": "hailo8", "ssh": "local-fixture",
        "rtool": str(remote), "timeout": 10, "result": {"steps": []}}
    exec(compile(h8_sync_block(source), "normal-updater-h8-sync", "exec"), namespace)
    return namespace["result"]["steps"]


def parent_command(remote):
    parent = remote / "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
    tree = ast.parse(parent.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name == "_v2791_concurrent_three_stage_command")
    namespace = {"__file__": str(parent), "Path": Path, "sys": sys}
    node.returns = None
    for arg in [*node.args.args, *node.args.kwonlyargs]: arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(parent), "exec"), namespace)
    return namespace[node.name](args=_args(), work=remote / "not-executed", result_json=remote / "not-produced.json")


@pytest.mark.parametrize("packaged", [False, True])
def test_normal_sync_materializes_actual_parent_helper_and_required_resources(tmp_path, monkeypatch, packaged):
    module, path = load_updater(packaged)
    remote = tmp_path / "tool"
    copied = local_transfer(module, monkeypatch, remote)
    steps = execute_block(module, path.read_text(), remote)
    command = parent_command(remote)
    assert command[1] == str(remote / "scripts" / HELPER)
    assert command[command.index("--frames") + 1] == "1000"
    assert command[command.index("--warmup") + 1] == "100"
    assert command[command.index("--repetitions") + 1] == "3"
    expected = json.loads((remote / RESOURCE / "expected_artifacts.json").read_text())
    required = {"three_stage_canary.py", "fast_decode_runtime.py", "stage_corpus.py",
        "cpp/CMakeLists.txt", "cpp/three_stage_canary.cpp", "expected_artifacts.json",
        "reference/dataset_detection_validation.json", "reference/multi_image_fast_decode_canary_report.json"}
    required.update(item["file"] for item in expected["reference_metadata"].values())
    assert len(required) == 12
    for relative in required:
        assert (remote / RESOURCE / relative).read_bytes() == (ROOT / RESOURCE / relative).read_bytes()
    assert all(source.read_bytes() == target.read_bytes() for source, target in copied)
    assert all(step["rc"] == 0 for step in steps)
    assert not (remote / "not-executed").exists()
    assert not any(source.suffix in {".hef", ".engine", ".dxnn", ".onnx"} for source, _ in copied)


def test_exact_previous_sync_reproduces_missing_helper(tmp_path, monkeypatch):
    before = os.environ.get("YOLO_H8_SYNC_SOURCE_BEFORE")
    if not before: pytest.skip("requires preserved original updater source")
    module, _ = load_updater(False)
    remote = tmp_path / "tool"
    local_transfer(module, monkeypatch, remote)
    execute_block(module, Path(before).read_text(), remote)
    with pytest.raises(FileNotFoundError, match="concurrent_three_stage_helper_missing"):
        parent_command(remote)


@pytest.mark.parametrize("damage", ["helper", "resource"])
def test_existing_byte_verifier_blocks_missing_or_changed_closure(tmp_path, monkeypatch, damage):
    module, path = load_updater(False)
    remote = tmp_path / "tool"
    local_transfer(module, monkeypatch, remote, damage=damage)
    with pytest.raises(RuntimeError, match="verification failed"):
        execute_block(module, path.read_text(), remote)


def test_updater_source_and_packaged_mirror_are_identical():
    assert (ROOT / "scripts/update_evalset_native_producers.py").read_bytes() == (
        ROOT / "onnx_splitpoint_tool/resources/remote_scripts/update_evalset_native_producers.py").read_bytes()
