"""Hailo backend helpers (optional).

This module is intentionally *optional*: the main tool should run without any
Hailo SDK installed. All imports of `hailo_sdk_client` happen at call time.

Current use cases
-----------------
- "Parse-only" feasibility check: can the given ONNX be translated by the Hailo
  DFC/SDK (i.e., `translate_onnx_model`)?

The goal is not to fully compile to HEF in the split-ranking stage, but to
quickly reject candidates that cannot be parsed/translated at all.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import onnx
from onnx import AttributeProto, helper


def hailo_sdk_available() -> bool:
    """Return True if the Hailo SDK python module can be imported."""
    try:
        import hailo_sdk_client  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class HailoProbeResult:
    """Result of a quick Hailo backend availability check."""

    ok: bool
    backend: str  # "local" or "wsl"
    reason: str = ""
    details: Optional[Dict[str, Any]] = None


# ------------------------------- WSL bridge -------------------------------

_WSL_RESULT_MARKER = "__SPLITPOINT_HAILO_RESULT__"


def hailo_wsl_available() -> bool:
    """Return True if we appear to be on Windows and `wsl` is callable."""
    if sys.platform != "win32":
        return False
    return (shutil.which("wsl.exe") is not None) or (shutil.which("wsl") is not None)


def _wsl_exe() -> str:
    """Pick the best WSL executable name for subprocess calls."""
    return "wsl.exe" if shutil.which("wsl.exe") is not None else "wsl"


def windows_path_to_wsl(path: Union[str, Path]) -> str:
    """Best-effort conversion of a Windows path to a WSL path.

    Examples
    --------
    "C:\\temp\\a b\\x.onnx" -> "/mnt/c/temp/a b/x.onnx"
    "D:/data/model.onnx"     -> "/mnt/d/data/model.onnx"

    If the input already looks like a Linux path ("/mnt/..." or "/home/..."),
    it is returned unchanged.
    """

    s = str(path)
    if s.startswith("/mnt/") or s.startswith("/home/") or s.startswith("/"):
        return s

    # Normalize backslashes
    s = s.replace("\\", "/")

    # Drive letter path
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2)
        return f"/mnt/{drive}/{rest}"

    return s


def _bash_quote(s: str) -> str:
    """Quote a string for `bash -lc` commands."""
    return shlex.quote(str(s))


def hailo_probe_local() -> HailoProbeResult:
    """Check whether the local Python environment can import the Hailo DFC SDK."""
    try:
        import hailo_sdk_client  # type: ignore

        details: Dict[str, Any] = {}
        details["hailo_sdk_client"] = getattr(hailo_sdk_client, "__file__", None)
        details["hailo_sdk_client_version"] = getattr(hailo_sdk_client, "__version__", None)
        return HailoProbeResult(ok=True, backend="local", details=details)
    except Exception as e:
        return HailoProbeResult(ok=False, backend="local", reason=str(e), details=None)


def hailo_probe_via_wsl(
    *,
    wsl_distro: str = "",
    wsl_venv_activate: str = "~/hailo_dfc_venv/bin/activate",
    timeout_s: int = 30,
) -> HailoProbeResult:
    """Check whether Hailo DFC is reachable inside WSL.

    This is meant for the Windows GUI, where the DFC lives in WSL.
    """
    if not hailo_wsl_available():
        return HailoProbeResult(ok=False, backend="wsl", reason="wsl.exe not found (WSL not available)")

    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl") or "wsl.exe"

    cmd = [wsl_exe]
    if wsl_distro:
        cmd += ["-d", wsl_distro]

    act = _bash_quote(wsl_venv_activate)
    # Print a unique marker so we can reliably detect success.
    bash = (
        "set -e; "
        f"source {act}; "
        "python3 -c \"import hailo_sdk_client; print('__HAILO_PROBE_OK__')\"; "
        "(hailo --version 2>/dev/null || true)"
    )
    cmd += ["--", "bash", "-lc", bash]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = "__HAILO_PROBE_OK__" in out
        details = {
            "returncode": proc.returncode,
            "output_tail": "\n".join(out.strip().splitlines()[-30:]),
        }
        return HailoProbeResult(ok=ok, backend="wsl", reason=("" if ok else "Probe failed"), details=details)
    except subprocess.TimeoutExpired:
        return HailoProbeResult(ok=False, backend="wsl", reason=f"WSL probe timed out after {timeout_s}s")
    except Exception as e:
        return HailoProbeResult(ok=False, backend="wsl", reason=str(e))


def hailo_probe_auto(
    *,
    backend: str = "auto",
    wsl_distro: str = "",
    wsl_venv_activate: str = "~/hailo_dfc_venv/bin/activate",
    timeout_s: int = 30,
) -> HailoProbeResult:
    backend = (backend or "auto").strip().lower()
    if backend not in ("auto", "local", "wsl"):
        return HailoProbeResult(ok=False, backend=backend, reason=f"Unknown backend: {backend!r}")

    if backend == "local":
        return hailo_probe_local()
    if backend == "wsl":
        return hailo_probe_via_wsl(wsl_distro=wsl_distro, wsl_venv_activate=wsl_venv_activate, timeout_s=timeout_s)

    # auto
    if hailo_sdk_available():
        return hailo_probe_local()
    return hailo_probe_via_wsl(wsl_distro=wsl_distro, wsl_venv_activate=wsl_venv_activate, timeout_s=timeout_s)


def _find_result_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON payload following our marker from mixed output."""
    if not text:
        return None
    idx = text.rfind(_WSL_RESULT_MARKER)
    if idx < 0:
        return None
    payload = text[idx + len(_WSL_RESULT_MARKER):].strip()
    try:
        return json.loads(payload)
    except Exception:
        return None


def hailo_parse_check_via_wsl(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    save_har: bool = True,
    disable_rt_metadata_extraction: bool = True,
    # WSL bridge settings
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "~/hailo_dfc_venv/bin/activate",
    wsl_timeout_s: int = 180,
) -> "HailoParseResult":
    """Run the parse-check inside WSL (Windows host -> WSL2 Linux backend).

    This is intended for the common situation where the Hailo DFC is only
    available as a Linux wheel, but the GUI is running on Windows.

    The function calls a tiny helper script *inside WSL* and parses a structured
    JSON result from mixed stdout/stderr using a marker token.
    """

    t0 = time.time()
    onnx_path = Path(onnx_path)
    if net_name is None:
        net_name = onnx_path.stem

    if sys.platform != "win32":
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error="WSL backend is only available when running on Windows.",
        )

    if not hailo_wsl_available():
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error="WSL backend not available (wsl.exe not found).",
        )

    # Resolve helper script path on Windows and convert to WSL path.
    helper_win = (Path(__file__).resolve().parent / "wsl_hailo_check.py")
    helper_wsl = windows_path_to_wsl(str(helper_win))

    # Convert ONNX and outdir paths.
    onnx_wsl = windows_path_to_wsl(str(onnx_path.resolve()))
    outdir_wsl = None
    if outdir is not None:
        outdir_wsl = windows_path_to_wsl(str(Path(outdir).resolve()))

    # Build a bash command that activates the venv and runs the helper.
    # Note: do not quote '~/...' to keep tilde expansion working.
    venv_activate = str(wsl_venv_activate).strip()
    if not venv_activate:
        venv_activate = "~/hailo_dfc_venv/bin/activate"

    cmd_parts = [
        "set -e",  # fail fast
        f"source {venv_activate}",
        # Make sure we always flush output.
        "export PYTHONUNBUFFERED=1",
        f"python3 {_bash_quote(helper_wsl)}"
        f" --onnx {_bash_quote(onnx_wsl)}"
        f" --hw-arch {_bash_quote(str(hw_arch))}"
        f" --net-name {_bash_quote(str(net_name))}"
        f" --fixup {'1' if fixup else '0'}"
        f" --add-conv-defaults {'1' if add_conv_defaults else '0'}"
        f" --save-har {'1' if save_har else '0'}"
        f" --disable-rt-metadata-extraction {'1' if disable_rt_metadata_extraction else '0'}",
    ]

    if outdir_wsl is not None:
        cmd_parts[-1] += f" --outdir {_bash_quote(outdir_wsl)}"

    # net_input_shapes is optional; for now we only support the default inference
    # on the WSL side. (Passing large dicts through CLI quoting is possible but
    # not necessary for the current use case.)

    bash_cmd = " && ".join(cmd_parts)

    wsl_cmd: List[str] = [_wsl_exe()]
    if wsl_distro:
        wsl_cmd += ["-d", str(wsl_distro)]
    wsl_cmd += ["--", "bash", "-lc", bash_cmd]

    try:
        proc = subprocess.run(
            wsl_cmd,
            capture_output=True,
            text=True,
            timeout=int(wsl_timeout_s),
            env=dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=f"WSL hailo check timed out after {wsl_timeout_s}s.",
        )
    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=f"WSL hailo check failed to launch: {type(e).__name__}: {e}",
        )

    mixed = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
    payload = _find_result_json(mixed)

    if payload is None:
        # Provide a short tail for debugging.
        tail = mixed[-2000:] if mixed else ""
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=(
                "WSL hailo check did not return a structured result. "
                f"exit_code={proc.returncode}. tail=\n{tail}"
            ),
        )

    # Map JSON payload back to our result object.
    return HailoParseResult(
        ok=bool(payload.get("ok")),
        elapsed_s=float(payload.get("elapsed_s", time.time() - t0)),
        hw_arch=str(payload.get("hw_arch", hw_arch)),
        net_name=str(payload.get("net_name", net_name)),
        backend=str(payload.get("backend") or "wsl"),
        error=payload.get("error"),
        har_path=payload.get("har_path"),
        fixed_onnx_path=payload.get("fixed_onnx_path"),
        fixup_report=payload.get("fixup_report"),
    )


def hailo_parse_check_auto(
    onnx_path: Union[str, Path],
    *,
    backend: str = "auto",
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    save_har: bool = True,
    disable_rt_metadata_extraction: bool = True,
    # WSL bridge settings
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "~/hailo_dfc_venv/bin/activate",
    wsl_timeout_s: int = 180,
) -> "HailoParseResult":
    """Convenience wrapper: pick the best available backend.

    backend:
      - "auto": local SDK if importable, else WSL (Windows only)
      - "local": require local SDK
      - "wsl": require WSL bridge
    """

    mode = str(backend or "auto").strip().lower()
    if mode not in {"auto", "local", "wsl"}:
        mode = "auto"

    if mode in {"auto", "local"} and hailo_sdk_available():
        return hailo_parse_check(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            save_har=save_har,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
        )

    if mode in {"auto", "wsl"}:
        return hailo_parse_check_via_wsl(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            save_har=save_har,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            wsl_distro=wsl_distro,
            wsl_venv_activate=wsl_venv_activate,
            wsl_timeout_s=wsl_timeout_s,
        )

    # Nothing available.
    return HailoParseResult(
        ok=False,
        elapsed_s=0.0,
        hw_arch=str(hw_arch),
        net_name=str(net_name or Path(str(onnx_path)).stem),
        error=(
            "No usable Hailo backend available. "
            "Install hailo_sdk_client in this Python env, or configure the WSL backend on Windows."
        ),
    )


# ------------------------------- ONNX fixups -------------------------------

def _get_attr(node: onnx.NodeProto, name: str) -> Optional[onnx.AttributeProto]:
    for a in node.attribute:
        if a.name == name:
            return a
    return None


def _set_or_patch_ints_attr(node: onnx.NodeProto, name: str, values: List[int]) -> None:
    """Ensure an INTS attribute exists and has the desired values."""
    a = _get_attr(node, name)
    if a is None:
        node.attribute.append(helper.make_attribute(name, list(values)))
        return
    if a.type == AttributeProto.INTS:
        # Patch in-place
        del a.ints[:]
        a.ints.extend([int(v) for v in values])
        return

    # Wrong type: replace
    node.attribute.remove(a)
    node.attribute.append(helper.make_attribute(name, list(values)))


def fix_onnx_for_hailo(
    model: onnx.ModelProto,
    *,
    add_conv_defaults: bool = True,
) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    """Apply a small set of pragmatic ONNX fixups that help the Hailo parser.

    This is intentionally conservative: it only fills in some missing Conv/
    ConvTranspose attributes (most commonly `kernel_shape`) and optionally adds
    a few default attributes when they are absent.

    Returns (patched_model, report).
    """

    patched = onnx.ModelProto()
    patched.CopyFrom(model)

    report: Dict[str, Any] = {
        "kernel_shape_patched": 0,
        "conv_defaults_added": 0,
        "notes": [],
    }

    g = patched.graph
    # Initializers are needed to infer some attributes (kernel from weight shape)
    init_by_name = {i.name: i for i in g.initializer}

    for n in g.node:
        if n.op_type not in {"Conv", "ConvTranspose"}:
            continue

        # ---- kernel_shape ----
        if _get_attr(n, "kernel_shape") is None:
            # Try to infer from weight tensor W (second input)
            if len(n.input) >= 2 and n.input[1] in init_by_name:
                W = init_by_name[n.input[1]]
                # Conv: [M, C/group, kH, kW]
                # ConvTranspose: [C, M/group, kH, kW]
                if len(W.dims) >= 4:
                    kh = int(W.dims[-2])
                    kw = int(W.dims[-1])
                    _set_or_patch_ints_attr(n, "kernel_shape", [kh, kw])
                    report["kernel_shape_patched"] += 1
                else:
                    report["notes"].append(
                        f"Could not infer kernel_shape for {n.op_type} '{n.name or '(unnamed)'}' (W dims={list(W.dims)})"
                    )

        if add_conv_defaults:
            # Add defaults if missing (ONNX spec defaults, but some parsers want explicit)
            # strides default: [1, 1]
            if _get_attr(n, "strides") is None:
                _set_or_patch_ints_attr(n, "strides", [1, 1])
                report["conv_defaults_added"] += 1

            # dilations default: [1, 1]
            if _get_attr(n, "dilations") is None:
                _set_or_patch_ints_attr(n, "dilations", [1, 1])
                report["conv_defaults_added"] += 1

            # pads default: [0, 0, 0, 0]
            if _get_attr(n, "pads") is None:
                _set_or_patch_ints_attr(n, "pads", [0, 0, 0, 0])
                report["conv_defaults_added"] += 1

    return patched, report


def infer_net_input_shapes_from_model(model: onnx.ModelProto) -> Optional[Union[List[int], Dict[str, List[int]]]]:
    """Infer a net_input_shapes structure from ONNX graph inputs.

    Hailo's `translate_onnx_model` accepts either:
      - a single shape list (for single-input networks)
      - a dict input_name -> shape list (for multi-input networks)

    We return None if all shapes are already fully static (no unknown dims),
    because in that case the translator usually does not need an override.
    """

    g = model.graph
    init_names = {i.name for i in g.initializer}
    inputs = [vi for vi in g.input if vi.name not in init_names]
    if not inputs:
        return None

    def _shape_of_vi(vi: onnx.ValueInfoProto) -> List[int]:
        tt = vi.type.tensor_type
        dims: List[int] = []
        if not tt.HasField("shape"):
            return []
        for d in tt.shape.dim:
            if d.HasField("dim_value") and int(d.dim_value) > 0:
                dims.append(int(d.dim_value))
            else:
                # Unknown/param dimension -> replace with 1 (safe default for feasibility checks)
                dims.append(1)
        return dims

    shapes: Dict[str, List[int]] = {vi.name: _shape_of_vi(vi) for vi in inputs}

    # If all dims are static already, return None.
    # (We cannot reliably detect if original was "unknown"; but if we inserted 1s,
    #  the shape will still look static. That's fine: it's a feasibility check.)
    if len(shapes) == 1:
        return list(next(iter(shapes.values())))
    return shapes


# ------------------------------- Parse check -------------------------------


@dataclass
class HailoParseResult:
    ok: bool
    elapsed_s: float
    hw_arch: str
    net_name: str
    backend: Optional[str] = None
    error: Optional[str] = None
    har_path: Optional[str] = None
    fixed_onnx_path: Optional[str] = None
    fixup_report: Optional[Dict[str, Any]] = None


def hailo_parse_check(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    save_har: bool = True,
    disable_rt_metadata_extraction: bool = True,
) -> HailoParseResult:
    """Run a *parse/translate-only* feasibility check via the Hailo SDK.

    This calls `ClientRunner.translate_onnx_model(...)` and treats success as
    "Hailo can translate this ONNX graph".
    """

    t0 = time.time()
    onnx_path = Path(onnx_path)
    if net_name is None:
        net_name = onnx_path.stem

    out_dir = Path(outdir) if outdir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from hailo_sdk_client import ClientRunner
    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            error=f"Hailo SDK not available: {e}",
        )

    fixed_path: Optional[Path] = None
    fixup_report: Optional[Dict[str, Any]] = None
    model_for_parse = onnx_path
    if fixup:
        try:
            m = onnx.load(str(onnx_path))
            m2, rep = fix_onnx_for_hailo(m, add_conv_defaults=add_conv_defaults)
            fixup_report = rep
            if out_dir is not None:
                fixed_path = out_dir / (onnx_path.stem + "_hailo_fixed.onnx")
            else:
                fixed_path = onnx_path.parent / (onnx_path.stem + "_hailo_fixed.onnx")
            onnx.save(m2, str(fixed_path))
            model_for_parse = fixed_path
        except Exception as e:
            # Fixup failed; fall back to original
            fixup_report = {"error": str(e)}
            model_for_parse = onnx_path

    # If user didn't provide shapes, try to infer.
    if net_input_shapes is None:
        try:
            m_tmp = onnx.load(str(model_for_parse))
            net_input_shapes = infer_net_input_shapes_from_model(m_tmp)
        except Exception:
            net_input_shapes = None

    try:
        runner = ClientRunner(hw_arch=str(hw_arch))
        runner.translate_onnx_model(
            model=str(model_for_parse),
            net_name=str(net_name),
            net_input_shapes=net_input_shapes,
            # Keep this on by default: avoids parsing issues with missing RT metadata
            disable_rt_metadata_extraction=bool(disable_rt_metadata_extraction),
        )

        har_path = None
        if save_har and out_dir is not None:
            har_path = str(out_dir / "parsed.har")
            runner.save_har(har_path)

        return HailoParseResult(
            ok=True,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            har_path=har_path,
            fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
            fixup_report=fixup_report,
        )

    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            error=str(e),
            har_path=None,
            fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
            fixup_report=fixup_report,
        )
