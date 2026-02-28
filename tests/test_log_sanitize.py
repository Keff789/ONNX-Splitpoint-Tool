from __future__ import annotations

from onnx_splitpoint_tool.log_utils import sanitize_log


def test_sanitize_log_removes_ansi_and_cr() -> None:
    raw = (
        "hello\rworld\n"
        "\x1b[0;93m2026-02-28 12:00:00 [W:onnxruntime:, transformer_memcpy.cc:85] Memcpy nodes\x1b[m\r\n"
        "done\x1b[0m"
    )

    cleaned = sanitize_log(raw)

    assert "\r" not in cleaned
    assert "\x1b" not in cleaned

    # CR becomes NL, and warnings remain readable.
    lines = cleaned.splitlines()
    assert "hello" in lines[0]
    assert "world" in lines[1]
    assert any("Memcpy nodes" in ln for ln in lines)
    assert any(ln.startswith("[warn]") for ln in lines if "onnxruntime" in ln)
