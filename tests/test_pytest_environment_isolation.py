from __future__ import annotations

import os
from pathlib import Path
import tempfile


def test_default_tool_home_is_isolated_from_the_real_user_home() -> None:
    home = Path(os.environ["HOME"]).resolve()

    assert home.name.startswith("onnx-splitpoint-pytest-")
    assert home.parent == Path(tempfile.gettempdir()).resolve()
    assert (home / ".onnx_splitpoint_tool") != (
        Path("/root") / ".onnx_splitpoint_tool"
    )


def test_xdg_roots_are_inside_the_isolated_home() -> None:
    home = Path(os.environ["HOME"]).resolve()

    for name in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        Path(os.environ[name]).resolve().relative_to(home)
