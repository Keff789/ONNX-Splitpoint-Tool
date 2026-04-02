from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import log_runtime



def test_publish_active_log_metadata_writes_pointer_files(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    log_path = tmp_path / "logs" / "gui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("hello", encoding="utf-8")

    log_runtime.publish_active_log_metadata(log_path, publish_cwd_alias=False)

    assert (home / ".onnx_splitpoint_tool" / "logs" / "active_log_path.txt").read_text(encoding="utf-8").strip() == str(log_path)
    assert (home / ".onnx_splitpoint_tool" / "active_log_path.txt").read_text(encoding="utf-8").strip() == str(log_path)
    assert log_runtime.resolve_active_log_path() == log_path
