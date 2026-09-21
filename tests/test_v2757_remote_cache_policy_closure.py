from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from onnx_splitpoint_tool.remote_runtime_closure import (
    native_remote_package_closure,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cache_policy_is_staged_before_remote_hailo_backend_import(
    tmp_path: Path,
) -> None:
    """Reproduce the clean-Orin import preflight used by the Native runner."""

    closure = native_remote_package_closure()
    modules = [module for _path, module, _tokens in closure]
    config_module = "onnx_splitpoint_tool.config_values"
    cache_module = "onnx_splitpoint_tool.cache_verify_policy"
    hailo_module = "onnx_splitpoint_tool.runners.backends.hailo_backend"

    assert cache_module in modules
    assert modules.index(config_module) < modules.index(cache_module)
    assert modules.index(cache_module) < modules.index(hailo_module)

    remote = tmp_path / "remote"
    required = {
        "onnx_splitpoint_tool/config_values.py",
        "onnx_splitpoint_tool/cache_verify_policy.py",
        "onnx_splitpoint_tool/hailo_attempt_receipts.py",
        "onnx_splitpoint_tool/hailo_timeout_policy.py",
        "onnx_splitpoint_tool/native_command_contract.py",
        "onnx_splitpoint_tool/native_split_quality.py",
        "onnx_splitpoint_tool/resources_utils.py",
        "onnx_splitpoint_tool/runners/_types.py",
        "onnx_splitpoint_tool/runners/request_latency.py",
        "onnx_splitpoint_tool/runners/backends/base.py",
        "onnx_splitpoint_tool/runners/backends/hailo_utils.py",
        "onnx_splitpoint_tool/runners/backends/hailo_backend.py",
        "onnx_splitpoint_tool/runners/native_split_quality_runtime.py",
    }
    closure_paths = {path for path, _module, _tokens in closure}
    assert required <= closure_paths

    for relative in sorted(required):
        source = ROOT / relative
        target = remote / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for relative in (
        "onnx_splitpoint_tool/__init__.py",
        "onnx_splitpoint_tool/runners/__init__.py",
        "onnx_splitpoint_tool/runners/backends/__init__.py",
    ):
        target = remote / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# isolated remote package marker\n", encoding="utf-8")

    materializer = "scripts/materialize_cache_verify_native_split_binding.py"
    materializer_target = remote / materializer
    materializer_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / materializer, materializer_target)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(remote)!r}); "
                f"import {hailo_module} as backend; "
                "assert backend.compiler_dispatch_forbidden is not None"
            ),
        ],
        cwd=remote,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert probe.returncode == 0, probe.stderr

    materializer_probe = subprocess.run(
        [sys.executable, "-I", "-B", str(materializer_target), "--help"],
        cwd=remote,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert materializer_probe.returncode == 0, materializer_probe.stderr
    assert "--engine-cache-root" in materializer_probe.stdout


def test_clean_imports_during_actual_remote_closure_staging_order(tmp_path: Path) -> None:
    """A clean remote must import each new dependency before its consumer.

    Stage exact source bytes in the production inventory order, including all
    preceding assets. Fresh isolated subprocesses cannot inherit management
    imports or find missing modules in the source checkout.
    """
    remote = tmp_path / "clean-default-order"
    consumers = {
        "onnx_splitpoint_tool.config_values",
        "onnx_splitpoint_tool.cache_verify_policy",
        "onnx_splitpoint_tool.runners.backends.hailo_backend",
    }
    checked = []
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for relative, module, _tokens in native_remote_package_closure():
        target = remote / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        parent = target.parent
        while parent != remote:
            marker = parent / "__init__.py"
            if not marker.exists():
                marker.write_text("# isolated remote package marker\n", encoding="utf-8")
            parent = parent.parent
        if module not in consumers:
            continue
        probe = subprocess.run(
            [sys.executable, "-I", "-B", "-c", (
                "import importlib,pathlib,sys; "
                f"root=pathlib.Path({str(remote)!r}); sys.path.insert(0,str(root)); "
                f"loaded=importlib.import_module({module!r}); "
                "from onnx_splitpoint_tool.config_values import parse_config_bool; "
                "assert parse_config_bool(False,field='remote.force_build') is False; "
                "assert parse_config_bool(True,field='remote.force_build') is True; "
                "assert pathlib.Path(loaded.__file__).is_relative_to(root); "
                "assert pathlib.Path(sys.modules['onnx_splitpoint_tool.config_values'].__file__).is_relative_to(root)"
            )],
            cwd=remote, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=30,
        )
        assert probe.returncode == 0, f"{module}: {probe.stderr}"
        checked.append(module)
    assert checked == [
        "onnx_splitpoint_tool.config_values",
        "onnx_splitpoint_tool.cache_verify_policy",
        "onnx_splitpoint_tool.runners.backends.hailo_backend",
    ]
