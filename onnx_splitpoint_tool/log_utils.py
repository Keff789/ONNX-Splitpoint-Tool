"""Logging-related utilities.

This module intentionally has *no* heavy dependencies so it can be reused by
both the GUI and benchmark/remote code paths.
"""

from __future__ import annotations

import re


# A reasonably complete ANSI escape sequence matcher (CSI + single-character).
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def sanitize_log(text: str) -> str:
    """Sanitize captured logs for better UX.

    - Normalize carriage returns (``\r``) into newlines (``\n``), so progress
      outputs become readable in text widgets and exported logs.
    - Strip ANSI escape sequences (colors, cursor movement, etc.).
    - Keep warnings/errors, but prefix ORT warnings/errors with ``[warn]`` / ``[error]``
      for consistency with our own log style.

    The function is conservative: it avoids filtering content; it only
    normalizes formatting artifacts.
    """

    if not text:
        return ""

    # Normalize CR to NL (including CRLF).
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip ANSI control sequences.
    text = _ANSI_ESCAPE_RE.sub("", text)

    out_lines: list[str] = []
    for line in text.split("\n"):
        # Preserve empty lines.
        if line == "":
            out_lines.append("")
            continue

        s = line.rstrip("\n")

        # ORT warnings are often emitted with color; after stripping ANSI they
        # still start with a timestamp and [W:onnxruntime,...]. Prefix them with
        # [warn] so they visually align with our own logs.
        if "[W:onnxruntime" in s and not s.lstrip().startswith("[warn]"):
            s = "[warn] " + s
        elif "[E:onnxruntime" in s and not s.lstrip().startswith("[error]"):
            s = "[error] " + s

        out_lines.append(s)

    return "\n".join(out_lines)
