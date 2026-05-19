"""Local helpers for bundling benchmark results.

The remote benchmark pipeline collects artifacts from a suite directory into a
dedicated results directory, and then tars it. On remote hosts this is done
with shell commands.

This module provides a *Python* implementation of the same logic to:
  - make the behavior testable (contract/guardrail tests)
  - keep artifact expectations documented in code
  - support both **full** and **lean** results bundles
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from .schema import RESULTS_BUNDLE_SCHEMA, RESULTS_BUNDLE_SCHEMA_VERSION, now_iso


def _read_suite_evaluation_profile(suite_dir: Path) -> dict[str, object]:
    try:
        bench_path = Path(suite_dir) / 'benchmark_set.json'
        if not bench_path.is_file():
            return {}
        import json
        data = json.loads(bench_path.read_text(encoding='utf-8'))
        prof = data.get('evaluation_profile') if isinstance(data, dict) else None
        return dict(prof or {}) if isinstance(prof, dict) else {}
    except Exception:
        return {}


def _read_results_evaluation_profile(results_dir: Path) -> dict[str, object]:
    try:
        bench_path = Path(results_dir) / 'benchmark_set.json'
        if not bench_path.is_file():
            return {}
        import json
        data = json.loads(bench_path.read_text(encoding='utf-8'))
        prof = data.get('evaluation_profile') if isinstance(data, dict) else None
        return dict(prof or {}) if isinstance(prof, dict) else {}
    except Exception:
        return {}


BundleMode = Literal["full", "lean"]


SUITE_LEVEL_PATTERNS: List[str] = [
    "benchmark_results_*.json",
    "benchmark_results_*.csv",
    "benchmark_summary_*.md",
    "benchmark_table_*.tex",
    "benchmark_suite_status*.json",
    "benchmark_suite_status*.csv",
    "benchmark_suite_status*.md",
    "v42_pipeline_summary.*",
    "v47_pipeline_summary.*",
    "analysis_*.pdf",
    "analysis_*.svg",
]


LEAN_SUITE_LEVEL_PATTERNS: List[str] = [
    "benchmark_results_*.json",
    "benchmark_results_*.csv",
    "benchmark_summary_*.md",
    "benchmark_table_*.tex",
    "benchmark_plan.json",
    "benchmark_set.json",
    "run_meta.json",
    # v42+ thesis/reporting summaries are intentionally small and are the most
    # useful files in lean bundles.  Keep them even when heavy figures/detections
    # are omitted.
    "v42_pipeline_summary.json",
    "v42_pipeline_summary.csv",
    "v42_pipeline_summary.md",
    "benchmark_suite_status_matrix.json",
    "benchmark_suite_status_matrix.csv",
    "benchmark_suite_status_matrix.md",
]


CASE_LEVEL_PATTERNS_FILES: List[str] = [
    "benchmark_*",
    "paper_figures_*",
]


LEAN_CASE_LEVEL_PATTERNS_FILES: List[str] = [
    "benchmark_*",
]


CASE_LEVEL_PATTERNS_DIRS: List[str] = [
    "results_*",
    "validation_*",
]


LEAN_CASE_LEVEL_PATTERNS_DIRS: List[str] = [
    "results_*",
]


LEAN_EXCLUDE_GLOBS: List[str] = [
    "**/*.pdf",
    "**/*.svg",
    "**/*.png",
    "**/paper_figures_*",
    "**/detections_*.json",
    "**/detections_*.png",
    "**/classification_*.json",
    "**/classification_*.png",
]


def _matches_any(path: Path, patterns: Iterable[str]) -> bool:
    return any(path.match(pat) for pat in patterns)


def _copy_if_file(src: Path, dst: Path) -> bool:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def collect_results_from_suite(suite_dir: Path, dst_results_dir: Path, *, mode: BundleMode = "full") -> Dict[str, int]:
    """Collect the relevant artifacts from a suite directory.

    The resulting directory structure is compatible with the remote
    ``results_bundle.tar.gz`` layout. ``mode='lean'`` keeps compact summary
    artefacts that are sufficient for analysis/review while dropping heavy
    visualization payloads.
    """

    suite_dir = Path(suite_dir)
    dst_results_dir = Path(dst_results_dir)
    dst_results_dir.mkdir(parents=True, exist_ok=True)

    suite_patterns = SUITE_LEVEL_PATTERNS if mode == "full" else LEAN_SUITE_LEVEL_PATTERNS
    case_file_patterns = CASE_LEVEL_PATTERNS_FILES if mode == "full" else LEAN_CASE_LEVEL_PATTERNS_FILES
    case_dir_patterns = CASE_LEVEL_PATTERNS_DIRS if mode == "full" else LEAN_CASE_LEVEL_PATTERNS_DIRS

    copied_files = 0
    copied_dirs = 0

    for pat in suite_patterns:
        for p in suite_dir.glob(pat):
            if _copy_if_file(p, dst_results_dir / p.name):
                copied_files += 1

    for case_dir in sorted([p for p in suite_dir.iterdir() if p.is_dir() and p.name.startswith('b')]):
        cid = case_dir.name
        out_case = dst_results_dir / cid
        out_case.mkdir(parents=True, exist_ok=True)

        for pat in case_file_patterns:
            for p in case_dir.glob(pat):
                if mode == "lean" and _matches_any(Path(cid) / p.name, LEAN_EXCLUDE_GLOBS):
                    continue
                if _copy_if_file(p, out_case / p.name):
                    copied_files += 1

        for pat in case_dir_patterns:
            for p in case_dir.glob(pat):
                if not p.is_dir():
                    continue
                rel = Path(cid) / p.name
                if mode == "lean" and _matches_any(rel, LEAN_EXCLUDE_GLOBS):
                    continue
                shutil.copytree(p, out_case / p.name, dirs_exist_ok=True)
                copied_dirs += 1

    eval_profile = _read_suite_evaluation_profile(suite_dir)
    manifest = {
        "schema": RESULTS_BUNDLE_SCHEMA,
        "schema_version": RESULTS_BUNDLE_SCHEMA_VERSION,
        "mode": mode,
        "created_at": now_iso(),
        "suite_dir": str(suite_dir),
        "stats": {"files": copied_files, "dirs": copied_dirs},
    }
    if eval_profile:
        manifest['evaluation_profile'] = eval_profile
    (dst_results_dir / "results_bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["stats"]


def _copy_results_tree_filtered(src_results_dir: Path, dst_results_dir: Path, *, mode: BundleMode = "full") -> Dict[str, int]:
    src_results_dir = Path(src_results_dir)
    dst_results_dir = Path(dst_results_dir)
    dst_results_dir.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    copied_dirs = 0
    for p in src_results_dir.rglob('*'):
        rel = p.relative_to(src_results_dir)
        if p.is_dir():
            continue
        if mode == 'lean' and _matches_any(rel, LEAN_EXCLUDE_GLOBS):
            continue
        dst = dst_results_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied_files += 1
    eval_profile = _read_results_evaluation_profile(src_results_dir)
    manifest = {
        "schema": RESULTS_BUNDLE_SCHEMA,
        "schema_version": RESULTS_BUNDLE_SCHEMA_VERSION,
        "mode": mode,
        "created_at": now_iso(),
        "results_dir": str(src_results_dir),
        "stats": {"files": copied_files, "dirs": copied_dirs},
    }
    if eval_profile:
        manifest['evaluation_profile'] = eval_profile
    (dst_results_dir / "results_bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest["stats"]


def _sf(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def _iter_result_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("benchmark_results_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    r = dict(row)
                    r.setdefault("source_file", path.name)
                    rows.append(r)
        elif isinstance(payload, dict):
            maybe_rows = payload.get("rows") or payload.get("results") or []
            if isinstance(maybe_rows, list):
                for row in maybe_rows:
                    if isinstance(row, dict):
                        r = dict(row)
                        r.setdefault("source_file", path.name)
                        rows.append(r)
    return rows


def _write_compact_pipeline_summary_if_missing(results_dir: Path) -> None:
    """Best-effort v47 summary synthesis for lean bundles.

    Some older runner templates wrote benchmark_results_*.json/csv but did not
    include the aggregate v42/status files in the local lean bundle.  Rebuild a
    compact thesis-oriented summary from the per-run result JSONs so the bundle
    remains self-contained.
    """
    results_dir = Path(results_dir)
    rows = _iter_result_rows(results_dir)
    if not rows:
        return

    def _full_ms(row: Dict[str, Any]) -> Optional[float]:
        return _sf(row.get("full_e2e_mean_ms")) or _sf(row.get("full_mean_ms"))

    full_candidates: List[Dict[str, Any]] = []
    hetero_candidates: List[Dict[str, Any]] = []
    flat: List[Dict[str, Any]] = []
    for row in rows:
        tag = str(row.get("run_id") or row.get("tag") or row.get("provider") or "").strip()
        source = str(row.get("source_file") or "")
        if not tag and source.startswith("benchmark_results_"):
            tag = source[len("benchmark_results_"):-len(".json")]
        st1 = str(row.get("stage1_provider") or "").lower()
        st2 = str(row.get("stage2_provider") or "").lower()
        provider = str(row.get("provider") or "").lower()
        item = {
            "tag": tag,
            "provider": row.get("provider"),
            "stage1_provider": row.get("stage1_provider"),
            "stage2_provider": row.get("stage2_provider"),
            "case_dir": row.get("case_dir"),
            "boundary": row.get("boundary"),
            "full_mean_ms": row.get("full_mean_ms"),
            "full_e2e_mean_ms": row.get("full_e2e_mean_ms"),
            "composed_mean_ms": row.get("composed_mean_ms"),
            "throughput_fps_makespan": row.get("throughput_fps_makespan"),
            "throughput_fps_cycle_est": row.get("throughput_fps_cycle_est"),
            "mini_coco_ap50_primary": row.get("mini_coco_ap50_primary"),
            "mini_coco_ap50_full": row.get("mini_coco_ap50_full"),
            "semantic_validation_passed": row.get("semantic_validation_passed"),
            "final_pass_all": row.get("final_pass_all"),
        }
        ms = _full_ms(row)
        tag_l = tag.lower()
        hetero = ("hailo" in st1 and ("trt" in st2 or "tensorrt" in st2)) or (("trt" in st1 or "tensorrt" in st1) and "hailo" in st2) or "to_hailo" in tag_l or "hailo8_to_trt" in tag_l
        single_system_full = bool(ms and ms > 0 and not hetero and "_to_" not in tag_l)
        if single_system_full:
            fc = dict(item)
            fc["baseline_ms_used"] = ms
            fc["baseline_fps"] = 1000.0 / ms
            full_candidates.append(fc)
        fps = _sf(row.get("throughput_fps_makespan")) or _sf(row.get("throughput_fps_cycle_est"))
        if hetero and fps and fps > 0:
            hc = dict(item)
            hc["streaming_fps"] = fps
            hc["streaming_metric_used"] = "makespan" if _sf(row.get("throughput_fps_makespan")) else "cycle_est"
            hetero_candidates.append(hc)
        flat.append(item)

    best_full = max(full_candidates, key=lambda x: float(x.get("baseline_fps") or 0.0), default=None)
    best_hetero = max(hetero_candidates, key=lambda x: float(x.get("streaming_fps") or 0.0), default=None)
    gain = None
    if best_full and best_hetero and _sf(best_full.get("baseline_fps")):
        gain = (float(best_hetero["streaming_fps"]) / float(best_full["baseline_fps"]) - 1.0) * 100.0
    payload = {
        "schema_version": 2,
        "description": "v47 compact pipeline summary synthesized from benchmark_results_*.json for self-contained result bundles.",
        "best_full_baseline": best_full,
        "best_heterogeneous_streaming": best_hetero,
        "gain_vs_best_full_percent": gain,
        "num_rows": len(flat),
        "num_full_candidates": len(full_candidates),
        "num_heterogeneous_streaming_candidates": len(hetero_candidates),
    }
    if not (results_dir / "v47_pipeline_summary.json").exists():
        (results_dir / "v47_pipeline_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not (results_dir / "v42_pipeline_summary.json").exists():
        (results_dir / "v42_pipeline_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    keys = ["tag", "provider", "stage1_provider", "stage2_provider", "case_dir", "boundary", "full_mean_ms", "full_e2e_mean_ms", "composed_mean_ms", "throughput_fps_makespan", "throughput_fps_cycle_est", "mini_coco_ap50_primary", "mini_coco_ap50_full", "semantic_validation_passed", "final_pass_all"]
    for name in ("v47_pipeline_summary.csv", "v42_pipeline_summary.csv"):
        path = results_dir / name
        if path.exists():
            continue
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in flat:
                w.writerow({k: row.get(k) for k in keys})

    def _fmt(v: Any) -> str:
        x = _sf(v)
        return "-" if x is None else f"{x:.3f}"
    md = ["# v47 Pipeline Summary", "", "Synthesized from bundled benchmark_results_*.json.", ""]
    if best_full:
        md.append(f"Best full baseline: `{best_full.get('tag')}` / `{best_full.get('case_dir')}` = {_fmt(best_full.get('baseline_fps'))} FPS.")
    if best_hetero:
        md.append(f"Best heterogeneous streaming: `{best_hetero.get('tag')}` / `{best_hetero.get('case_dir')}` = {_fmt(best_hetero.get('streaming_fps'))} FPS.")
    if gain is not None:
        md.append(f"Gain vs best full baseline: **{gain:.2f}%**.")
    md += ["", "| tag | case | full e2e ms | composed ms | streaming FPS | AP50 primary | semantic pass |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in sorted(flat, key=lambda x: _sf(x.get("throughput_fps_makespan")) or _sf(x.get("throughput_fps_cycle_est")) or -1.0, reverse=True)[:30]:
        fps = _sf(row.get("throughput_fps_makespan")) or _sf(row.get("throughput_fps_cycle_est"))
        md.append(f"| {row.get('tag')} | {row.get('case_dir')} | {_fmt(row.get('full_e2e_mean_ms'))} | {_fmt(row.get('composed_mean_ms'))} | {_fmt(fps)} | {_fmt(row.get('mini_coco_ap50_primary'))} | {row.get('semantic_validation_passed')} |")
    for name in ("v47_pipeline_summary.md", "v42_pipeline_summary.md"):
        path = results_dir / name
        if not path.exists():
            path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Minimal status matrix when the runner did not emit one.
    if not (results_dir / "benchmark_suite_status_matrix.json").exists():
        matrix = []
        for row in flat:
            matrix.append({
                "run": row.get("tag"),
                "case": row.get("case_dir"),
                "full": "ok" if _sf(row.get("full_mean_ms")) is not None else "missing",
                "composed": "ok" if _sf(row.get("composed_mean_ms")) is not None else "missing",
            })
        (results_dir / "benchmark_suite_status_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
        with (results_dir / "benchmark_suite_status_matrix.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["run", "case", "full", "composed"])
            w.writeheader(); w.writerows(matrix)
        lines = ["# Benchmark suite status matrix", "", "| run | case | full | composed |", "|---|---:|---|---|"]
        for m in matrix:
            lines.append(f"| {m.get('run')} | {m.get('case')} | {m.get('full')} | {m.get('composed')} |")
        (results_dir / "benchmark_suite_status_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_results_bundle(results_dir: Path, tar_gz_path: Path, *, mode: BundleMode = "full") -> None:
    """Create a .tar.gz bundle from a collected results directory."""

    results_dir = Path(results_dir)
    tar_gz_path = Path(tar_gz_path)
    tar_gz_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_gz_path, "w:gz") as tf:
        tf.add(results_dir, arcname=".")


def create_results_bundle_from_suite(suite_dir: Path, tar_gz_path: Path, *, mode: BundleMode = "full") -> None:
    """Collect suite results and create a tar.gz bundle in one step."""

    suite_dir = Path(suite_dir)
    tar_gz_path = Path(tar_gz_path)
    tmp_results = tar_gz_path.parent / (tar_gz_path.stem + f"_{mode}_tmp")
    if tmp_results.exists():
        shutil.rmtree(tmp_results, ignore_errors=True)
    tmp_results.mkdir(parents=True, exist_ok=True)
    try:
        collect_results_from_suite(suite_dir, tmp_results, mode=mode)
        _write_compact_pipeline_summary_if_missing(tmp_results)
        create_results_bundle(tmp_results, tar_gz_path, mode=mode)
    finally:
        shutil.rmtree(tmp_results, ignore_errors=True)


def create_results_bundle_from_results_dir(results_dir: Path, tar_gz_path: Path, *, mode: BundleMode = "full") -> None:
    """Create a bundle from an already-collected local results directory."""

    results_dir = Path(results_dir)
    tar_gz_path = Path(tar_gz_path)
    tmp_results = tar_gz_path.parent / (tar_gz_path.stem + f"_{mode}_tmp")
    if tmp_results.exists():
        shutil.rmtree(tmp_results, ignore_errors=True)
    tmp_results.mkdir(parents=True, exist_ok=True)
    try:
        _copy_results_tree_filtered(results_dir, tmp_results, mode=mode)
        _write_compact_pipeline_summary_if_missing(tmp_results)
        create_results_bundle(tmp_results, tar_gz_path, mode=mode)
    finally:
        shutil.rmtree(tmp_results, ignore_errors=True)
