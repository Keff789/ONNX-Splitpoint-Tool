"""Global safety boundary for the Tool test suite.

The application intentionally materializes user defaults below ``$HOME``.
Tests that exercise those defaults must never read or modify a developer's
real ``~/.onnx_splitpoint_tool`` tree.  Install the isolated home before test
modules are imported so plain ``pytest`` is safe without caller-managed
environment variables.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


_TEST_HOME: Path | None = None
_ORIGINAL_ENV: dict[str, str | None] = {}
_ISOLATED_ENV = (
    "HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "ONNX_SPLITPOINT_RUN_MODES_FILE",
)


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    del config
    global _TEST_HOME
    if _TEST_HOME is not None:
        return

    _TEST_HOME = Path(tempfile.mkdtemp(prefix="onnx-splitpoint-pytest-"))
    values = {
        "HOME": _TEST_HOME,
        "XDG_CACHE_HOME": _TEST_HOME / ".cache",
        "XDG_CONFIG_HOME": _TEST_HOME / ".config",
        "XDG_DATA_HOME": _TEST_HOME / ".local" / "share",
        "ONNX_SPLITPOINT_RUN_MODES_FILE": _TEST_HOME / ".onnx_splitpoint_tool" / "run_modes.yaml",
    }
    for name in _ISOLATED_ENV:
        _ORIGINAL_ENV[name] = os.environ.get(name)
        value = values[name]
        if name == "ONNX_SPLITPOINT_RUN_MODES_FILE":
            value.parent.mkdir(parents=True, exist_ok=True)
        else:
            value.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(value)

    # Use the runtime's supported API before tests construct sessions. The
    # environment variable alone is not a substitute for this control.
    # Missing dependencies are reported by the release environment gate.
    import importlib.util
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime
        onnxruntime.disable_telemetry_events()


def pytest_unconfigure(config) -> None:  # type: ignore[no-untyped-def]
    del config
    global _TEST_HOME
    isolated = _TEST_HOME
    _TEST_HOME = None
    for name in _ISOLATED_ENV:
        previous = _ORIGINAL_ENV.pop(name, None)
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    if isolated is not None:
        shutil.rmtree(isolated, ignore_errors=True)
