"""ONNX split/export: generated runner scripts and benchmark helpers.

This module holds code that writes helper scripts (runner skeletons, netron launcher, etc.)
used by exported split artifacts.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------- Runner resources ----------------------------

_TEST_IMAGE_PNG_B64 = """\
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
AAAASUVORK5CYII=
"""

def write_runner_skeleton_onnxruntime(out_dir: str, *, manifest_filename: str = "split_manifest.json", target: str = "auto") -> str:
    """Generate a Python runner skeleton (onnxruntime) next to the exported split models.

    The runner is meant as a starting point for benchmarking / integration and performs:
      - full model run: full(x)
      - split run: part2(part1(x))
      - timing with warmup + measured runs (defaults: warmup=5, runs=10)
      - output comparison + optional report plots

    The script is self-contained and can optionally use an image as input to "see something"
    (useful for CV models like ResNet/MobileNet/YOLO). If no image is provided, random inputs are used.

    It also drops a small default test image into the export folder: test_image.png
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "run_split_onnxruntime.py")

    # Runner script (template + placeholder replacement)
    script_template_path = Path(__file__).resolve().parent / "resources" / "templates" / "run_split_onnxruntime.py.txt"
    script = script_template_path.read_text(encoding="utf-8")
    script = script.replace("__MANIFEST_FILENAME__", str(manifest_filename))
    target = (target or "auto").lower()
    if target not in {"auto","cpu","cuda","tensorrt"}:
        target = "auto"
    script = script.replace("__DEFAULT_PROVIDER__", str(target))

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
    try:
        os.chmod(out_path, 0o755)
    except Exception:
        pass

    # Also drop a default test image for convenience (so you can run the runner immediately).
    # If the tool package provides a `test_image.png` next to this file, prefer that.
    # Otherwise fall back to the embedded tiny placeholder.
    try:
        img_path = os.path.join(out_dir, "test_image.png")
        if not os.path.exists(img_path):
            import shutil
            pkg_dir = os.path.dirname(__file__)
            src = os.path.join(pkg_dir, "test_image.png")
            if os.path.exists(src):
                shutil.copyfile(src, img_path)
            else:
                import base64 as _b64
                with open(img_path, "wb") as f:
                    f.write(_b64.b64decode(_TEST_IMAGE_PNG_B64))
    except Exception:
        pass

    # Convenience wrappers (double-click friendly on Windows)
    try:
        script_name = os.path.basename(out_path)
        bat_path = os.path.join(out_dir, "run_split_onnxruntime.bat")
        bat = (
            "@echo off\n"
            "setlocal\n"
            "cd /d %~dp0\n"
            f"python \"{script_name}\" --manifest \"{manifest_filename}\" %*\n"
            "pause\n"
        )
        with open(bat_path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(bat)
    except Exception:
        pass

    try:
        script_name = os.path.basename(out_path)
        sh_path = os.path.join(out_dir, "run_split_onnxruntime.sh")
        sh = (
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "cd \"$(dirname \"$0\")\"\n"
            f"python3 \"{script_name}\" --manifest \"{manifest_filename}\" \"$@\"\n"
        )
        with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(sh)
        try:
            os.chmod(sh_path, 0o755)
        except Exception:
            pass
    except Exception:
        pass

    return out_path


# Backwards compatible alias (older code imports this symbol).
def write_runner_onnxruntime(out_dir: str, *, manifest_filename: str = "split_manifest.json", target: str = "auto") -> str:
    return write_runner_skeleton_onnxruntime(out_dir, manifest_filename=manifest_filename, target=target)


# Backwards-compatible alias (older GUI versions may call write_runner_skeleton)

def write_runner_skeleton(out_dir: str, *, manifest_filename: str = "split_manifest.json", target: str = "auto") -> str:
    return write_runner_skeleton_onnxruntime(out_dir, manifest_filename=manifest_filename, target=target)


def write_netron_launcher(out_dir: str, *, manifest_filename: str = "split_manifest.json") -> dict:
    """Create helper scripts to open the (split) models in Netron.

    This writes three files into out_dir:
      - open_netron_split.py
      - open_netron_split.bat
      - open_netron_split.sh

    The scripts read the split manifest to locate full/part1/part2 ONNX models and start
    Netron for each present file.

    Notes:
      - Netron is optional and must be installed separately: `pip install netron`.
      - Netron does not provide a stable programmatic PDF export API. Use your browser's
        print-to-PDF instead.
    """

    os.makedirs(out_dir, exist_ok=True)

    py_name = "open_netron_split.py"
    bat_name = "open_netron_split.bat"
    sh_name = "open_netron_split.sh"

    py_path = os.path.join(out_dir, py_name)
    bat_path = os.path.join(out_dir, bat_name)
    sh_path = os.path.join(out_dir, sh_name)

    py = r"""#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path


def _load_manifest(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _start_netron(netron, model_path: Path, *, browse: bool, port: int):
    # Netron's Python API differs slightly across versions. Try common call patterns.
    try:
        return netron.start(str(model_path), browse=browse, port=port)
    except TypeError:
        try:
            return netron.start(str(model_path), browse=browse)
        except TypeError:
            return netron.start(str(model_path))


def main() -> int:
    ap = argparse.ArgumentParser(description='Open split models in Netron')
    ap.add_argument('--manifest', type=str, default='split_manifest.json')
    ap.add_argument('--what', type=str, default='all', choices=['all', 'full', 'part1', 'part2'])
    ap.add_argument('--no-browse', action='store_true', help='Do not auto-open the browser')
    ap.add_argument('--port', type=int, default=0, help='Port for Netron (0=auto). Use different ports if starting multiple servers.')
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 2

    m = _load_manifest(manifest_path)
    base = manifest_path.parent

    # Paths in the manifest are stored relative to the split folder.
    candidates = {
        'full': base / m.get('full_model', ''),
        'part1': base / m.get('part1', ''),
        'part2': base / m.get('part2', ''),
    }

    files = []
    if args.what in ('all', 'full') and candidates['full'].exists():
        files.append(('full', candidates['full']))
    if args.what in ('all', 'part1') and candidates['part1'].exists():
        files.append(('part1', candidates['part1']))
    if args.what in ('all', 'part2') and candidates['part2'].exists():
        files.append(('part2', candidates['part2']))

    if not files:
        print('No model files found to open (check manifest paths).')
        return 3

    try:
        import netron  # type: ignore
    except Exception as e:
        print('Netron is not installed or failed to import.')
        print('Install with:  pip install netron')
        print(f'Import error: {type(e).__name__}: {e}')
        return 4

    print('Starting Netron...')
    for tag, p in files:
        try:
            url = _start_netron(netron, p, browse=(not args.no_browse), port=args.port)
            print(f'  {tag}: {p} -> {url}')
        except Exception as e:
            print(f'  {tag}: failed to start Netron for {p} ({type(e).__name__}: {e})')

    print("\nIf the browser did not open automatically, copy one of the URLs above.")
    print('Press Ctrl+C to stop Netron.')
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('Stopping.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
"""

    bat = """@echo off
setlocal
cd /d %~dp0
python open_netron_split.py --manifest {manifest}
pause
""".format(manifest=manifest_filename)

    sh = """#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 open_netron_split.py --manifest {manifest}
""".format(manifest=manifest_filename)

    with open(py_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(py)
    try:
        os.chmod(py_path, 0o755)
    except Exception:
        pass

    with open(bat_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(bat)

    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    try:
        os.chmod(sh_path, 0o755)
    except Exception:
        pass

    return {
        'netron_py': py_name,
        'netron_bat': bat_name,
        'netron_sh': sh_name,
    }
