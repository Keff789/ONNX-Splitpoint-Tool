from pathlib import Path

from onnx_splitpoint_tool.gui.panels.panel_benchmark_analysis import _analysis_cache_base


class _DummyApp:
    def __init__(self, root: str | None):
        self.default_output_dir = root


def test_analysis_cache_base_uses_configured_workdir(tmp_path: Path) -> None:
    app = _DummyApp(str(tmp_path))
    cache = _analysis_cache_base(app)
    assert cache == tmp_path.resolve() / "Results" / "benchmark_analysis_cache"


def test_analysis_cache_base_falls_back_to_cwd_when_root_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cache = _analysis_cache_base(None)
    assert cache == tmp_path.resolve() / "benchmark_analysis_cache"
