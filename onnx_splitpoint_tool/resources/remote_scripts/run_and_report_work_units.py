#!/usr/bin/env python3
"""Run a workload command and emit its actual completed work-unit count.

The helper is designed for u.RECS command-window measurements.  It streams the
child output unchanged, inspects JSON payloads and referenced result JSON files,
and finally emits one machine-readable marker::

    __SPLITPOINT_WORK_UNITS__=<integer>

The marker lets the energy collector prefer the work completed in the exact
measurement command over a precomputed ``FPS * duration`` fallback.  No marker
is emitted when the child fails or when no defensible completed-count field is
available.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

MARKER = "__SPLITPOINT_WORK_UNITS__="
EXACT_COUNT_KEYS = (
    "completed_work_units",
    "completed_frames",
    "frames_completed",
    "measured_frames",
    "consumed_frames",
    "inferences_completed",
)
# Ambiguous legacy fields are diagnostic only.  They may be requested counts,
# so even the retained compatibility flag must never promote them to exact.
LEGACY_COUNT_KEYS = (
    "frames",
    "num_frames",
)
REPORT_KEYS = (
    "report",
    "result_json",
    "results_json",
    "native_fifo_result",
    "native_fifo_results_json",
    "output_json",
    "json",
)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not (number > 0 and number.is_integer()):
        return None
    return int(number)


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _count_from_payload(payload: Any, *, trust_legacy_frames: bool = False) -> tuple[int | None, str, bool]:
    if not isinstance(payload, (dict, list)):
        return None, "", False
    flattened = list(_walk(payload))
    for wanted in EXACT_COUNT_KEYS + LEGACY_COUNT_KEYS:
        candidates: list[int] = []
        for key, value in flattened:
            if key == wanted:
                parsed = _positive_int(value)
                if parsed is not None:
                    candidates.append(parsed)
        if candidates:
            # The maximum is conservative for nested summaries that contain both
            # per-stage and total counts, while the key priority avoids requested
            # frame counts entirely.
            exact = wanted in EXACT_COUNT_KEYS
            return max(candidates), f"json_field:{wanted}", exact
    return None, "", False


def _json_values(text: str) -> list[Any]:
    """Best-effort extraction of JSON objects embedded in mixed stdout."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(text):
        brace_positions = [pos for pos in (text.find("{", index), text.find("[", index)) if pos >= 0]
        if not brace_positions:
            break
        start = min(brace_positions)
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(value)
        index = start + max(1, end)
    return values


def _report_paths(payloads: Iterable[Any], cwd: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for payload in payloads:
        for key, value in _walk(payload):
            if key not in REPORT_KEYS or not isinstance(value, str) or not value.strip():
                continue
            candidate = Path(os.path.expanduser(os.path.expandvars(value.strip())))
            if not candidate.is_absolute():
                candidate = cwd / candidate
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            marker = str(resolved)
            if marker not in seen:
                seen.add(marker)
                paths.append(resolved)
    return paths


def _load_json(path: Path) -> Any:
    try:
        if path.is_file() and path.stat().st_size <= 64 * 1024 * 1024:
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _run_child(command: list[str]) -> tuple[int, str, str]:
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def relay(stream: Any, target: Any, chunks: list[str]) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            chunks.append(line)
            target.write(line)
            target.flush()
        stream.close()

    threads = [
        threading.Thread(target=relay, args=(process.stdout, sys.stdout, stdout_chunks), daemon=True),
        threading.Thread(target=relay, args=(process.stderr, sys.stderr, stderr_chunks), daemon=True),
    ]
    for thread in threads:
        thread.start()
    return_code = process.wait()
    for thread in threads:
        thread.join(timeout=5)
    return return_code, "".join(stdout_chunks), "".join(stderr_chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trust-legacy-frames-as-completed",
        action="store_true",
        help="Deprecated compatibility flag. Ambiguous 'frames' fields remain non-exact; emit completed_frames instead.",
    )
    parser.add_argument(
        "--sidecar",
        default="",
        help="Optional JSON sidecar for completed-work evidence (also ONNX_SPLITPOINT_WORK_UNITS_SIDECAR).",
    )
    # BooleanOptionalAction exists only from Python 3.9.  Accelerator images
    # still using Python 3.8 must parse the wrapper before any workload starts.
    follow_group = parser.add_mutually_exclusive_group()
    follow_group.add_argument(
        "--follow-reports", dest="follow_reports", action="store_true", default=True,
        help="Inspect result files referenced by child JSON.",
    )
    follow_group.add_argument(
        "--no-follow-reports", dest="follow_reports", action="store_false",
        help="Do not inspect result files referenced by child JSON.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Child command, normally introduced by --")
    ns = parser.parse_args()
    command = list(ns.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a child command is required after --")

    return_code, stdout_text, stderr_text = _run_child(command)
    if return_code != 0:
        return int(return_code)

    payloads = _json_values(stdout_text + "\n" + stderr_text)
    best_count: int | None = None
    best_source = ""
    best_exact = False
    # Referenced files are generally richer than the final compact stdout JSON.
    if ns.follow_reports:
        for report_path in _report_paths(payloads, Path.cwd()):
            payload = _load_json(report_path)
            count, source, exact = _count_from_payload(payload, trust_legacy_frames=ns.trust_legacy_frames_as_completed)
            if count is not None and (best_count is None or (exact and not best_exact) or (exact == best_exact and count > best_count)):
                best_count = count
                best_source = f"result_file:{report_path}:{source}"
                best_exact = exact
    for payload in payloads:
        count, source, exact = _count_from_payload(payload, trust_legacy_frames=ns.trust_legacy_frames_as_completed)
        if count is not None and (best_count is None or (exact and not best_exact) or (exact == best_exact and count > best_count)):
            best_count = count
            best_source = f"stdout:{source}"
            best_exact = exact

    sidecar_value = str(ns.sidecar or os.environ.get("ONNX_SPLITPOINT_WORK_UNITS_SIDECAR", "") or "").strip()
    sidecar_payload = {
        "schema": "onnx-splitpoint/runtime-completed-work-units",
        "schema_version": 1,
        "count": best_count,
        "completed_work_units": best_count if best_exact else None,
        "source": best_source,
        "exact": bool(best_count is not None and best_exact),
        "child_returncode": return_code,
    }
    if sidecar_value:
        sidecar = Path(os.path.expanduser(os.path.expandvars(sidecar_value)))
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"__SPLITPOINT_WORK_UNITS_SIDECAR_ERROR__={type(exc).__name__}:{exc}", file=sys.stderr, flush=True)

    if best_count is not None:
        print(f"{MARKER}{best_count}", flush=True)
        print(f"__SPLITPOINT_WORK_UNITS_SOURCE__={best_source}", flush=True)
        print(f"__SPLITPOINT_WORK_UNITS_EXACT__={1 if best_exact else 0}", flush=True)
    else:
        print("__SPLITPOINT_WORK_UNITS_UNAVAILABLE__=no_completed_count_in_runtime_output", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
