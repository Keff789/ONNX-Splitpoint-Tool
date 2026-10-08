"""The normal sync must make a cold concurrent-helper import executable."""
import json
from pathlib import Path
import subprocess
import sys

from onnx_splitpoint_tool.remote_runtime_closure import (
    native_remote_package_closure,
    native_remote_resource_closure,
)
from onnx_splitpoint_tool.workflow import runner
from test_v2803_fix5_native_asset_sync import _LocalTransport, _sync


def test_cold_target_imports_actual_helper_and_canary_after_verified_sync(tmp_path):
    target = tmp_path / "cold remote"
    transport = _LocalTransport(target)
    for relative in ("onnx_splitpoint_tool/__init__.py", "onnx_splitpoint_tool/release_identity.py"):
        _sync(transport, relative, ())
    for relative, tokens in native_remote_resource_closure():
        _sync(transport, relative, tokens)
    helper = "native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
    runner._sync_remote_script_v60i(
        ssh="nx@isolated-native-test", remote_tool_dir=str(target),
        script_name=helper, required_tokens=("--campaign-config",),
        timeout=20, process_runner=transport,
    )
    resource = target / "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7"
    program = f"""
import sys, importlib.util, json
sys.path.insert(0, {str(target)!r})
sys.path.insert(0, {str(resource)!r})
for name, path in [('cold_helper', {str(target / 'scripts' / helper)!r}),
                   ('cold_canary', {str(resource / 'three_stage_canary.py')!r})]:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
assert callable(sys.modules['cold_canary'].run_campaign)
print(json.dumps({{'imported': True, 'inference_started': False}}))
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", program],
        cwd=target, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"imported": True, "inference_started": False}


def test_non_python_resources_do_not_enter_module_import_closure():
    packages = {relative for relative, _, _ in native_remote_package_closure()}
    resources = {relative for relative, _ in native_remote_resource_closure()}
    assert not packages.intersection(resources)
    assert all(relative.endswith(".py") for relative in packages)
    assert any(relative.endswith("cpp/CMakeLists.txt") for relative in resources)
    assert any(relative.endswith("cpp/three_stage_canary.cpp") for relative in resources)
