from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _log(log: Optional[Callable[[str], None]], line: str) -> None:
    if log is None:
        return
    try:
        log(str(line))
    except Exception:
        pass


def _copy_if_changed(src: Path, dst: Path) -> bool:
    """Copy ``src`` -> ``dst`` only when content actually differs."""
    try:
        if dst.exists() and src.read_bytes() == dst.read_bytes():
            return False
    except Exception:
        # If comparison fails, overwrite as the safe default.
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def assert_generated_runner_is_self_consistent(path: Path) -> None:
    """Reject obviously stale / broken generated runner scripts.

    The checks are intentionally static/lightweight so they do not import heavy
    runtime dependencies such as onnxruntime or Hailo Python packages.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Could not read generated runner for self-check: {path}: {e}") from e

    try:
        compile(text, str(path), "exec")
    except SyntaxError as e:
        raise RuntimeError(f"{path} failed syntax self-check: {e}") from e

    required_helpers = (
        "_maybe_cast_for_onnx_input",
        "_shape_from_ort_input",
    )
    missing: list[str] = []
    for helper_name in required_helpers:
        referenced = helper_name in text
        defined = f"def {helper_name}(" in text
        if referenced and not defined:
            missing.append(helper_name)

    if missing:
        raise RuntimeError(
            f"{path} references helper(s) {', '.join(missing)} but does not define them. "
            "Refusing to keep a stale or broken runner."
        )

    module_requirements = {
        "re": r"\bre\.(?:search|match|sub|compile|fullmatch|findall|finditer)\b",
    }
    for module_name, usage_pattern in module_requirements.items():
        uses_module = re.search(usage_pattern, text) is not None
        has_import = (
            re.search(
                rf"^\s*(?:import\s+{module_name}\b|from\s+{module_name}\s+import\b)",
                text,
                flags=re.M,
            )
            is not None
        )
        if uses_module and not has_import:
            raise RuntimeError(
                f"{path} references module '{module_name}' helpers but does not import '{module_name}'. "
                "Refusing to keep a stale or broken runner."
            )


def resolve_suite_bench_json_name(suite_dir: Path, *, benchmark_set_json: Optional[Path] = None) -> str:
    """Choose the benchmark json filename used by ``benchmark_suite.py``.

    Preference order:
    1. explicit ``benchmark_set_json`` filename when it points to a JSON file
    2. ``benchmark_set.json`` inside the suite
    3. first JSON file in the suite root
    4. fallback to ``benchmark_set.json``
    """
    suite_dir = Path(suite_dir)
    bench_json_name = "benchmark_set.json"

    try:
        if benchmark_set_json is not None:
            b = Path(benchmark_set_json)
            if b.is_file() and b.suffix.lower() == ".json":
                bench_json_name = b.name
    except Exception:
        pass

    if (suite_dir / bench_json_name).exists():
        return bench_json_name
    if (suite_dir / "benchmark_set.json").exists():
        return "benchmark_set.json"
    cand = sorted([p.name for p in suite_dir.glob("*.json") if p.is_file()])
    if cand:
        return cand[0]
    return bench_json_name


def refresh_suite_harness(
    suite_dir: Path | str,
    *,
    benchmark_set_json: Optional[Path | str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Refresh a benchmark suite's embedded harness files in-place.

    This updates, when necessary:
    - ``benchmark_suite.py``
    - vendored ``splitpoint_runners`` package
    - per-case ONNXRuntime runner wrappers under ``b*/``

    Files are only rewritten when bytes changed so bundle caching remains useful.
    """
    suite_dir = Path(suite_dir).expanduser().resolve()
    if not suite_dir.exists() or not suite_dir.is_dir():
        raise FileNotFoundError(f"Suite directory not found: {suite_dir}")

    try:
        bench_path = Path(benchmark_set_json).expanduser().resolve() if benchmark_set_json is not None else None
    except Exception:
        bench_path = None
    bench_json_name = resolve_suite_bench_json_name(suite_dir, benchmark_set_json=bench_path)

    stats: Dict[str, Any] = {
        "suite_dir": str(suite_dir),
        "bench_json_name": bench_json_name,
        "suite_script_updated": False,
        "runner_lib_files_updated": 0,
        "case_runner_cases_updated": 0,
        "case_runner_files_updated": 0,
        "case_count": 0,
        "changed": False,
    }

    try:
        from ..gui.controller import write_benchmark_suite_script
    except Exception as e:  # pragma: no cover - import failure is reported to caller
        raise RuntimeError(f"Could not import benchmark suite writer: {e}") from e

    try:
        from ..split_export_runners import write_runner_skeleton_onnxruntime as _write_runner_onnxruntime
    except Exception:  # pragma: no cover
        from ..split_export_runners import write_runner_onnxruntime as _write_runner_onnxruntime  # type: ignore

    with tempfile.TemporaryDirectory(prefix="osp_suite_refresh_") as _td:
        tmp_dir = Path(_td)
        tmp_script = Path(write_benchmark_suite_script(tmp_dir, bench_json_name=bench_json_name))
        src_runners = tmp_dir / "splitpoint_runners"

        if not src_runners.exists() or not src_runners.is_dir():
            try:
                from ..gui.controller import _copy_runner_lib as _vendor_runner_lib
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Could not import runner vendoring helper: {e}") from e
            _vendor_runner_lib(tmp_dir)
            src_runners = tmp_dir / "splitpoint_runners"

        dst_script = suite_dir / "benchmark_suite.py"
        if tmp_script.exists() and _copy_if_changed(tmp_script, dst_script):
            stats["suite_script_updated"] = True
            stats["changed"] = True
            _log(log, f"[info] Refreshed benchmark_suite.py: {dst_script}")

        if src_runners.exists() and src_runners.is_dir():
            dst_runners = suite_dir / "splitpoint_runners"
            n_updated = 0
            for src_file in src_runners.rglob("*"):
                if src_file.is_dir():
                    continue
                rel = src_file.relative_to(src_runners)
                dst_file = dst_runners / rel
                if _copy_if_changed(src_file, dst_file):
                    n_updated += 1
            if n_updated:
                stats["runner_lib_files_updated"] = int(n_updated)
                stats["changed"] = True
                _log(log, f"[info] Refreshed splitpoint_runners: {n_updated} file(s) updated")

    case_manifests = sorted(suite_dir.glob("b*/split_manifest.json"))
    stats["case_count"] = len(case_manifests)

    def _refresh_case_runner(case_dir: Path, manifest_filename: str) -> int:
        updated = 0
        with tempfile.TemporaryDirectory(prefix="osp_runner_refresh_") as _td:
            tmp_case = Path(_td)
            try:
                _write_runner_onnxruntime(str(tmp_case), manifest_filename=manifest_filename, target="auto")  # type: ignore[arg-type]
            except TypeError:
                _write_runner_onnxruntime(str(tmp_case), Path(manifest_filename), export_mode="folder")  # type: ignore[misc]

            src_runner = tmp_case / "run_split_onnxruntime.py"
            if src_runner.exists():
                assert_generated_runner_is_self_consistent(src_runner)

            for fname in (
                "run_split_onnxruntime.py",
                "run_split_onnxruntime.sh",
                "run_split_onnxruntime.bat",
            ):
                src = tmp_case / fname
                if not src.exists():
                    continue
                dst = case_dir / fname

                dst_needs_repair = False
                if fname == "run_split_onnxruntime.py" and dst.exists():
                    try:
                        assert_generated_runner_is_self_consistent(dst)
                    except Exception:
                        dst_needs_repair = True

                if dst_needs_repair:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    updated += 1
                elif _copy_if_changed(src, dst):
                    updated += 1

            final_runner = case_dir / "run_split_onnxruntime.py"
            if final_runner.exists():
                assert_generated_runner_is_self_consistent(final_runner)
        return updated

    total_case_files_updated = 0
    total_case_dirs_updated = 0
    for manifest in case_manifests:
        case_dir = manifest.parent
        if case_dir.parent != suite_dir:
            continue
        n = _refresh_case_runner(case_dir, manifest.name)
        if n:
            total_case_dirs_updated += 1
            total_case_files_updated += n

    if total_case_files_updated:
        stats["case_runner_cases_updated"] = int(total_case_dirs_updated)
        stats["case_runner_files_updated"] = int(total_case_files_updated)
        stats["changed"] = True
        _log(
            log,
            f"[info] Refreshed runner scripts in {total_case_dirs_updated}/{len(case_manifests)} cases "
            f"(files updated: {total_case_files_updated}).",
        )

    if not stats["changed"]:
        _log(log, "[info] Suite harness already up to date.")

    return stats
