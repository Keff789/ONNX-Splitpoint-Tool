from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_hailo_build_hef_auto_accepts_outdir_alias(tmp_path, monkeypatch):
    """Regression test: older GUI code used 'outdir' instead of 'out_dir'.

    The GUI helper must accept both, otherwise HEF generation fails early with
    "out_dir is required".
    """

    # Local import so this test stays lightweight.
    from onnx_splitpoint_tool.hailo_backend import hailo_build_hef_auto
    from onnx_splitpoint_tool.runners.backends.base import PreparedHandle
    from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend

    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"dummy")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def _fake_prepare(self, run_cfg, artifacts_dir):
        hef = Path(artifacts_dir) / "m.hef"
        hef.write_bytes(b"hef")
        return PreparedHandle(
            input_names=["x"],
            output_names=["y"],
            handle=SimpleNamespace(hef_path=str(hef)),
        )

    monkeypatch.setattr(HailoBackend, "prepare", _fake_prepare)

    # NOTE: use the legacy kwarg "outdir" on purpose.
    res = hailo_build_hef_auto(
        onnx_path=str(onnx_path),
        outdir=str(out_dir),
        hw_arch="hailo8",
    )

    assert res.ok is True
    assert res.hef_path is not None
    assert Path(res.hef_path).exists()
