"""Logging-related utilities.

This module intentionally has *no* heavy dependencies so it can be reused by
both the GUI and benchmark/remote code paths.
"""

from __future__ import annotations

import re
import json
from pathlib import Path


# A reasonably complete ANSI escape sequence matcher (CSI + single-character).
_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def redact_diagnostic(value):
    """Redact credentials before diagnostic persistence or display."""
    if isinstance(value, dict):
        return {k: "[REDACTED]" if re.search(r"(?i)password|secret|token|credential|private_key|authorization|api_key|access_key", str(k)) else redact_diagnostic(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_diagnostic(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?is)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[REDACTED PRIVATE KEY]", value)
        value = re.sub(r"(?i)(authorization:\s*bearer\s+)\S+", r"\1[REDACTED]", value)
        value = re.sub(r'''(?i)((?:password|secret|token|credential|authorization|api_key|access_key)["']?\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;]+)''', r"\1[REDACTED]", value)
        value = re.sub(r"(://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
    return value


def compact_diagnostic_cause(text: str, limit: int = 240) -> str:
    lines = str(redact_diagnostic(text)).splitlines()
    candidates = [s.strip() for s in lines if re.search(r"(?i)error|failed|timeout|denied|no space|cleanup|cancel", s)]
    line = candidates[-1] if candidates else (lines[-1].strip() if lines else "no additional output")
    return line[:limit] + ("…" if len(line) > limit else "")


class NativeDisplayLog:
    """Display-only stream projection; the process/parser retains original bytes."""

    def __init__(self, path, emit, *, label, command):
        self.path = Path(path)
        self.emit = emit
        self.label = label
        self.failed = False
        self.hidden = 0
        self.json_depth = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"command_argv": redact_diagnostic(command)}, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._failure(exc)
        emit(f"[native:{label}] gestartet · Diagnose: {self.path}")

    def _failure(self, exc):
        if not self.failed:
            self.failed = True
            self.emit(f"[warn] Native-Diagnose nicht schreibbar: {self.path} ({type(exc).__name__})")

    def __call__(self, line):
        safe = str(redact_diagnostic(line))
        if not self.failed:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(safe.rstrip("\n") + "\n")
            except OSError as exc:
                self._failure(exc)
        stripped = safe.strip()
        # JSON bodies and long payloads stay in the diagnostic. Scalar failure
        # fields and explicit warnings remain visible even inside pretty JSON.
        event = stripped.startswith("[") and not stripped.startswith(('["', '[{'))
        in_json = self.json_depth > 0 or stripped.startswith(('{', '[{', '["')) or stripped == '['
        if in_json and not (event and stripped != '['):
            quoted = escaped = False
            for char in stripped:
                if escaped:
                    escaped = False
                elif char == '\\' and quoted:
                    escaped = True
                elif char == '"':
                    quoted = not quoted
                elif not quoted and char in '{[':
                    self.json_depth += 1
                elif not quoted and char in '}]':
                    self.json_depth = max(0, self.json_depth - 1)
        important = bool(re.search(r"(?i)\b(warn(?:ing)?|error|failed|timeout|cleanup|cancelled)\b", stripped))
        if event or important:
            self.emit(f"[native:{self.label}] {stripped[:320]}" + ("…" if len(stripped) > 320 else ""))
        elif not in_json and stripped and not stripped.startswith(('"', '{', '}', '[', ']', ',')) and len(stripped) <= 240:
            self.emit(f"[native:{self.label}] {stripped}")
        else:
            self.hidden += 1

    def finish(self, *, returncode, elapsed_s):
        self.emit(f"[native:{self.label}] beendet · rc={returncode} · {elapsed_s:.1f}s · Diagnose: {self.path}")


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
