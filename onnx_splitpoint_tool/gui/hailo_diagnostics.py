"""GUI-friendly formatting helpers for Hailo HEF diagnostics.

These helpers keep Hailo build postmortems readable in the GUI without pulling
Tkinter into unit-testable code.  They operate on either:

- :class:`onnx_splitpoint_tool.hailo_backend.HailoHefBuildResult` objects, or
- the JSON payload written to ``hailo_hef_build_result.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_HEF_RESULT_KEYS = (
    "ok",
    "elapsed_s",
    "hw_arch",
    "net_name",
    "backend",
    "error",
    "hef_path",
    "parsed_har_path",
    "quant_har_path",
    "fixed_onnx_path",
    "fixup_report",
    "skipped",
    "calib_info",
    "returncode",
    "debug_log",
    "timed_out",
    "timeout_kind",
    "last_stage",
    "failure_kind",
    "unsupported_reason",
    "details",
    "result_json_path",
    "cuda_probe",
)

_STAGE_ORDER = (
    "translation",
    "mixed_precision",
    "statistics_collector",
    "bias_correction",
    "layer_noise_analysis",
    "partition_search",
    "allocation",
    "compile",
)

_ALGO_ORDER = (
    "Mixed Precision",
    "Statistics Collector",
    "Bias Correction",
    "Layer Noise Analysis",
)


def _dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _status_text(payload: Dict[str, Any]) -> str:
    if bool(payload.get("ok")):
        return "OK"
    if bool(payload.get("skipped")):
        return "SKIPPED"
    if bool(payload.get("timed_out")):
        return "TIMEOUT"
    return "FAILED"


def _fmt_seconds(value: Any) -> str:
    try:
        s = float(value)
    except Exception:
        return str(value)
    if s < 60.0:
        return f"{s:.1f}s"
    minutes, secs = divmod(s, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {secs:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes):02d}m {secs:04.1f}s"


def _short_error(text: Any, *, max_len: int = 220) -> str:
    s = str(text or "").strip().replace("\r", "")
    if not s:
        return ""
    first = s.splitlines()[0].strip()
    if len(first) > max_len:
        return first[: max_len - 3].rstrip() + "..."
    return first


def normalize_hailo_result_payload(result: Any) -> Dict[str, Any]:
    """Convert a Hailo result object / JSON payload into a plain dict."""

    if isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {}
        for key in _HEF_RESULT_KEYS:
            try:
                value = getattr(result, key)
            except Exception:
                continue
            payload[key] = value

    details = _dict(payload.get("details"))
    extra = _dict(details.get("extra_fields"))
    if payload.get("result_json_path") is None and extra.get("result_json_path"):
        payload["result_json_path"] = extra.get("result_json_path")
    if not isinstance(payload.get("cuda_probe"), dict) and isinstance(extra.get("cuda_probe"), dict):
        payload["cuda_probe"] = extra.get("cuda_probe")
    payload["details"] = details or None
    return payload


def collect_hailo_diagnostics(result: Any, *, label: str = "") -> Dict[str, Any]:
    """Create a compact GUI entry from a Hailo HEF result / JSON payload."""

    payload = normalize_hailo_result_payload(result)
    details = _dict(payload.get("details"))
    proc = _dict(details.get("process_summary"))
    snap = _dict(details.get("system_snapshot"))
    cuda = _dict(payload.get("cuda_probe"))
    status = _status_text(payload)

    entry: Dict[str, Any] = {
        "label": str(label or payload.get("net_name") or "Hailo build").strip(),
        "status": status,
        "ok": bool(payload.get("ok")),
        "skipped": bool(payload.get("skipped")),
        "timed_out": bool(payload.get("timed_out")),
        "backend": str(payload.get("backend") or "").strip(),
        "hw_arch": str(payload.get("hw_arch") or "").strip(),
        "net_name": str(payload.get("net_name") or "").strip(),
        "elapsed_s": payload.get("elapsed_s"),
        "error": str(payload.get("error") or "").strip(),
        "failure_kind": str(payload.get("failure_kind") or "").strip(),
        "timeout_kind": str(payload.get("timeout_kind") or "").strip(),
        "last_stage": str(payload.get("last_stage") or proc.get("last_stage") or "").strip(),
        "returncode": payload.get("returncode"),
        "unsupported_reason": str(payload.get("unsupported_reason") or "").strip(),
        "paths": {
            "result_json": str(payload.get("result_json_path") or "").strip(),
            "hef": str(payload.get("hef_path") or "").strip(),
            "fixed_onnx": str(payload.get("fixed_onnx_path") or "").strip(),
            "debug_log": str(payload.get("debug_log") or "").strip(),
            "parsed_har": str(payload.get("parsed_har_path") or "").strip(),
            "quant_har": str(payload.get("quant_har_path") or "").strip(),
        },
        "process_summary": proc,
        "cuda_probe": cuda,
        "system_snapshot": snap,
        "calib_info": _dict(payload.get("calib_info")),
        "fixup_report": _dict(payload.get("fixup_report")),
    }
    entry["paths"] = {k: v for k, v in entry["paths"].items() if v}
    return entry


def load_hailo_result_json(path: Path | str, *, label: str = "") -> Dict[str, Any]:
    """Load and normalize a ``hailo_hef_build_result.json`` file."""

    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    payload.setdefault("result_json_path", str(p))
    return collect_hailo_diagnostics(payload, label=(label or p.parent.name or p.name))


def _ordered_pairs(data: Dict[str, Any], preferred: Iterable[str]) -> List[tuple[str, Any]]:
    seen = set()
    out: List[tuple[str, Any]] = []
    for key in preferred:
        if key in data:
            out.append((str(key), data[key]))
            seen.add(str(key))
    for key, value in data.items():
        k = str(key)
        if k not in seen:
            out.append((k, value))
    return out


def format_hailo_diagnostics_short_lines(entry: Dict[str, Any]) -> List[str]:
    """Return a compact multi-line summary suitable for live GUI logs."""

    label = str(entry.get("label") or "Hailo build")
    status = str(entry.get("status") or "")
    head_bits = [f"status={status}"]
    backend = str(entry.get("backend") or "")
    hw_arch = str(entry.get("hw_arch") or "")
    if backend:
        head_bits.append(f"backend={backend}")
    if hw_arch:
        head_bits.append(f"hw={hw_arch}")
    if entry.get("elapsed_s") not in (None, ""):
        head_bits.append(f"elapsed={_fmt_seconds(entry.get('elapsed_s'))}")
    if entry.get("last_stage"):
        head_bits.append(f"last_stage={entry.get('last_stage')}")
    lines = [f"[hailo diag] {label}: " + " | ".join(head_bits)]

    proc = _dict(entry.get("process_summary"))
    core_bits: List[str] = []
    if proc.get("context_count") is not None:
        core_bits.append(f"contexts={proc.get('context_count')}")
    if proc.get("partition_time_s") is not None:
        part = f"partition={_fmt_seconds(proc.get('partition_time_s'))}"
        if proc.get("partition_iterations") is not None:
            part += f"/{int(proc.get('partition_iterations'))} iters"
        core_bits.append(part)
    if proc.get("allocation_time_s") is not None:
        core_bits.append(f"alloc={_fmt_seconds(proc.get('allocation_time_s'))}")
    if proc.get("compilation_time_s") is not None:
        core_bits.append(f"compile={_fmt_seconds(proc.get('compilation_time_s'))}")
    if core_bits:
        lines.append("[hailo diag] " + " | ".join(core_bits))

    detected = [k for k, v in _dict(proc.get("detected")).items() if bool(v)]
    if detected:
        lines.append("[hailo diag] detected=" + ", ".join(detected[:6]))

    snr_pairs = _ordered_pairs(_dict(proc.get("snr_db")), ())[:3]
    if snr_pairs:
        snr_txt = ", ".join(f"{name.split('/')[-1]}={float(val):.2f}dB" for name, val in snr_pairs)
        lines.append(f"[hailo diag] SNR: {snr_txt}")

    err = _short_error(entry.get("error"))
    if err and status != "OK":
        lines.append(f"[hailo diag] error: {err}")

    result_json = _dict(entry.get("paths")).get("result_json")
    if result_json:
        lines.append(f"[hailo diag] result_json: {result_json}")
    return lines


def format_hailo_diagnostics_text(entry: Dict[str, Any]) -> str:
    """Render a human-readable multi-section diagnostic text block."""

    label = str(entry.get("label") or "Hailo build")
    lines: List[str] = [f"Label: {label}"]
    lines.append(f"Status: {entry.get('status') or 'unknown'}")
    if entry.get("backend"):
        lines.append(f"Backend: {entry.get('backend')}")
    if entry.get("hw_arch"):
        lines.append(f"HW arch: {entry.get('hw_arch')}")
    if entry.get("net_name"):
        lines.append(f"Net: {entry.get('net_name')}")
    if entry.get("elapsed_s") not in (None, ""):
        lines.append(f"Elapsed: {_fmt_seconds(entry.get('elapsed_s'))}")
    if entry.get("failure_kind"):
        lines.append(f"Failure kind: {entry.get('failure_kind')}")
    if entry.get("timeout_kind"):
        lines.append(f"Timeout kind: {entry.get('timeout_kind')}")
    if entry.get("last_stage"):
        lines.append(f"Last stage: {entry.get('last_stage')}")
    if entry.get("returncode") not in (None, ""):
        lines.append(f"Return code: {entry.get('returncode')}")
    if entry.get("unsupported_reason"):
        lines.append(f"Unsupported reason: {entry.get('unsupported_reason')}")

    paths = _dict(entry.get("paths"))
    if paths:
        lines.append("")
        lines.append("Artifacts:")
        name_map = {
            "result_json": "Result JSON",
            "hef": "HEF",
            "fixed_onnx": "Fixed ONNX",
            "debug_log": "Debug log",
            "parsed_har": "Parsed HAR",
            "quant_har": "Quant HAR",
        }
        for key in ("result_json", "hef", "fixed_onnx", "parsed_har", "quant_har", "debug_log"):
            if paths.get(key):
                lines.append(f"  - {name_map.get(key, key)}: {paths[key]}")

    calib = _dict(entry.get("calib_info"))
    if calib:
        lines.append("")
        lines.append("Calibration:")
        if calib.get("source"):
            lines.append(f"  - source: {calib.get('source')}")
        if calib.get("used_count") is not None or calib.get("requested_count") is not None:
            req = calib.get("requested_count")
            used = calib.get("used_count")
            lines.append(f"  - used/requested: {used}/{req}")
        if calib.get("batch_size") is not None:
            lines.append(f"  - batch size: {calib.get('batch_size')}")

    fixup = _dict(entry.get("fixup_report"))
    if fixup:
        lines.append("")
        lines.append("Fixup report:")
        for key in ("kernel_shape_patched", "conv_defaults_added"):
            if fixup.get(key) is not None:
                lines.append(f"  - {key}: {fixup.get(key)}")
        notes = fixup.get("notes")
        if isinstance(notes, list) and notes:
            for note in notes[:8]:
                lines.append(f"  - note: {note}")

    proc = _dict(entry.get("process_summary"))
    if proc:
        lines.append("")
        lines.append("Process summary:")
        if proc.get("context_count") is not None:
            lines.append(f"  - contexts: {proc.get('context_count')}")
        if proc.get("partition_time_s") is not None or proc.get("partition_iterations") is not None:
            part = "  - partition search: "
            if proc.get("partition_time_s") is not None:
                part += _fmt_seconds(proc.get("partition_time_s"))
            else:
                part += "n/a"
            if proc.get("partition_iterations") is not None:
                part += f" over {int(proc.get('partition_iterations'))} iterations"
            lines.append(part)
        if proc.get("allocation_time_s") is not None:
            lines.append(f"  - allocation: {_fmt_seconds(proc.get('allocation_time_s'))}")
        if proc.get("compilation_time_s") is not None:
            lines.append(f"  - compilation: {_fmt_seconds(proc.get('compilation_time_s'))}")
        if proc.get("elapsed_s_observed") is not None:
            lines.append(f"  - observed elapsed: {_fmt_seconds(proc.get('elapsed_s_observed'))}")

        stage_durations = _dict(proc.get("stage_durations_s"))
        if stage_durations:
            lines.append("  - observed stages:")
            for name, val in _ordered_pairs(stage_durations, _STAGE_ORDER)[:10]:
                lines.append(f"      {name}: {_fmt_seconds(val)}")

        algo_times = _dict(proc.get("algo_times_s"))
        if algo_times:
            lines.append("  - optimization algorithms:")
            for name, val in _ordered_pairs(algo_times, _ALGO_ORDER)[:10]:
                lines.append(f"      {name}: {_fmt_seconds(val)}")

        snr = _dict(proc.get("snr_db"))
        if snr:
            lines.append("  - SNR:")
            for name, val in _ordered_pairs(snr, ())[:8]:
                try:
                    val_txt = f"{float(val):.2f} dB"
                except Exception:
                    val_txt = str(val)
                lines.append(f"      {name}: {val_txt}")

        detected = [k for k, v in _dict(proc.get("detected")).items() if bool(v)]
        if detected:
            lines.append("  - detected: " + ", ".join(detected))
        if proc.get("single_context_failure"):
            lines.append("  - single-context failure: " + _short_error(proc.get("single_context_failure"), max_len=320))

    cuda = _dict(entry.get("cuda_probe"))
    if cuda:
        lines.append("")
        lines.append("Compute selection:")
        if cuda.get("summary"):
            lines.append(f"  - summary: {cuda.get('summary')}")
        if cuda.get("reason"):
            lines.append(f"  - reason: {cuda.get('reason')}")
        for key, label_name in (("cuda_root", "CUDA root"), ("libdevice_path", "libdevice"), ("ptxas_path", "ptxas")):
            if cuda.get(key):
                lines.append(f"  - {label_name}: {cuda.get(key)}")

    snap = _dict(entry.get("system_snapshot"))
    cmds = _dict(snap.get("commands"))
    if snap or cmds:
        lines.append("")
        lines.append("System snapshot:")
        if snap.get("platform"):
            lines.append(f"  - platform: {snap.get('platform')} {snap.get('platform_release') or ''}".rstrip())
        if snap.get("python"):
            lines.append(f"  - python: {snap.get('python')}")
        for name in ("nvidia_smi", "free_m", "df_h", "os_info"):
            cmd = _dict(cmds.get(name))
            if not cmd:
                continue
            rc = cmd.get("returncode")
            lines.append(f"  - {name} (rc={rc})")
            out = str(cmd.get("stdout") or cmd.get("stderr") or cmd.get("error") or "").strip()
            if out:
                for ln in out.splitlines()[:8]:
                    lines.append(f"      {ln}")

    err = str(entry.get("error") or "").strip()
    if err:
        lines.append("")
        lines.append("Error:")
        lines.extend(err.splitlines())

    return "\n".join(lines).strip() + "\n"
