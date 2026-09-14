"""GUI controller utilities (no Tk widget code).

This module is intended to hold non-UI application logic that is invoked by the GUI.
Keep this free of tkinter/ttk imports so it can be unit-tested headlessly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ..resources_utils import copy_resource_file, copy_resource_tree, read_text

log = logging.getLogger(__name__)


def _templates_dir() -> Path:
    # Kept for backwards-compatible callers that still inspect the path.
    from ..resources_utils import persistent_resource_path

    return persistent_resource_path("resources", "templates")


def load_template_text(filename: str, *, encoding: str = "utf-8") -> str:
    """Load a text template from onnx_splitpoint_tool/resources/templates/.

    Raises FileNotFoundError if missing.
    """
    return read_text("resources", "templates", filename, encoding=encoding)


def write_benchmark_suite_script(dst_dir: str | Path, *, bench_json_name: str = "benchmark_set.json") -> str:
    """Create a benchmark suite runner script in *dst_dir*.

    Returns the written script path as string.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    script_path = dst_dir / "benchmark_suite.py"
    template = load_template_text("benchmark_suite.py.txt")
    script = template.replace("__BENCH_JSON__", bench_json_name)

    # Only rewrite the script if the content actually changed. This keeps
    # bundle caching effective (mtime stays stable on no-op updates).
    script_changed = True
    if script_path.exists():
        try:
            if script_path.read_text(encoding="utf-8") == script:
                script_changed = False
        except Exception:
            pass

    if script_changed:
        script_path.write_text(script, encoding="utf-8")
        try:
            # Make executable on POSIX; harmless on Windows.
            os.chmod(script_path, 0o755)
        except Exception:
            pass
        log.info("Wrote benchmark suite script: %s", script_path)

    # v60: vendor the canonical scientific reporter next to benchmark_suite.py.
    # This is refreshed independently from benchmark_suite.py so existing
    # generated suites can receive reporting fixes without changing the main
    # runner template.
    reporter_path = dst_dir / "scientific_reporter_v60.py"
    reporter = load_template_text("scientific_reporter_v60.py.txt")
    reporter_changed = True
    if reporter_path.exists():
        try:
            if reporter_path.read_text(encoding="utf-8") == reporter:
                reporter_changed = False
        except Exception:
            pass
    if reporter_changed:
        reporter_path.write_text(reporter, encoding="utf-8")
        try:
            os.chmod(reporter_path, 0o755)
        except Exception:
            pass
        log.info("Wrote scientific reporter companion: %s", reporter_path)

    # Phase-1: vendor the lightweight runner library into the suite root.
    # This must run even when benchmark_suite.py itself did not change,
    # because benchmark bundles for existing suites rely on the vendored
    # splitpoint_runners package being refreshed independently.
    try:
        _copy_runner_lib(dst_dir)
    except Exception as e:
        log.warning("Could not copy runner lib into suite: %s: %s", type(e).__name__, e)
    required_quality_first_runtime = (
        dst_dir / "splitpoint_runners" / "native_split_quality_runtime.py",
        dst_dir / "splitpoint_runners" / "native_split_quality.py",
        dst_dir / "splitpoint_runners" / "native_command_contract.py",
        dst_dir / "splitpoint_runners" / "native_trt_from_benchmarkset.py",
        dst_dir / "splitpoint_runners" / "native_detection_postprocess.py",
        dst_dir / "splitpoint_runners" / "native_full_input.py",
        dst_dir / "splitpoint_runners" / "preprocessing_contract.py",
    )
    missing_quality_first_runtime = [
        str(path) for path in required_quality_first_runtime if not path.is_file()
    ]
    if missing_quality_first_runtime:
        raise RuntimeError(
            "Generated BenchmarkSet lacks mandatory Quality-FIRST split runtime: "
            + ", ".join(missing_quality_first_runtime)
        )

    return str(script_path)


def _copy_runner_lib(suite_dir: Path) -> None:
    dst = suite_dir / "splitpoint_runners"
    copy_resource_tree("runners", dest=dst)
    # The generated case runner executes from a self-contained suite archive.
    # It cannot rely on the full ``onnx_splitpoint_tool`` package being installed
    # on an accelerator host, but central-quality endpoint attestation must use
    # exactly the same implementation as Native reporting.  Vendor the canonical
    # module byte-for-byte instead of maintaining a second implementation.
    copy_resource_file(
        "native_output_endpoint.py",
        dest=dst / "native_output_endpoint.py",
    )
    # Quality-FIRST split execution is performed inside the self-contained
    # remote suite.  Vendor the canonical contract modules and the exact TRT
    # builder so the remote producer cannot fall back to an older installed
    # tool checkout.
    copy_resource_file(
        "native_command_contract.py",
        dest=dst / "native_command_contract.py",
    )
    copy_resource_file(
        "native_split_quality.py",
        dest=dst / "native_split_quality.py",
    )
    copy_resource_file(
        "native_detection_postprocess.py",
        dest=dst / "native_detection_postprocess.py",
    )
    copy_resource_file(
        "preprocessing_contract.py",
        dest=dst / "preprocessing_contract.py",
    )
    copy_resource_file(
        "runners", "native_split_quality_runtime.py",
        dest=dst / "native_split_quality_runtime.py",
    )
    copy_resource_file(
        "resources", "remote_scripts", "native_trt_from_benchmarkset.py",
        dest=dst / "native_trt_from_benchmarkset.py",
    )


def _ignore_pycache(_dir: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__"}
    ignored.update({n for n in names if n.endswith(".pyc")})
    return ignored
