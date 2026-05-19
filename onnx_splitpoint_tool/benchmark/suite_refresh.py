from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .classification_validation_presets import (
    default_available_classification_validation_preset,
    normalize_classification_validation_preset,
    provision_classification_validation_source_to_suite,
)
from .validation_assets import (
    default_detection_validation_source,
    provision_detection_validation_source_to_suite,
)

try:  # keep suite refresh usable even when imported from a vendored bundle
    from .. import __version__ as _TOOL_VERSION
except Exception:  # pragma: no cover
    _TOOL_VERSION = "unknown"


def _log(log: Optional[Callable[[str], None]], line: str) -> None:
    if log is None:
        return
    try:
        log(str(line))
    except Exception:
        pass


def _embedded_semantic_validation_dataset_source() -> Optional[Path]:
    """Return the prepared/default COCO-50 validation resource if present."""
    return default_detection_validation_source()


def _provision_embedded_semantic_validation_dataset(suite_dir: Path) -> tuple[Optional[str], bool]:
    """Provision the default detection validation set exactly once at suite level."""
    before = (Path(suite_dir) / 'resources' / 'validation' / 'coco_50_data').exists()
    rel = provision_detection_validation_source_to_suite(Path(suite_dir))
    after = bool(rel) and (Path(suite_dir) / str(rel)).exists()
    return rel, bool(after and not before)


def _read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _write_json_if_changed(path: Path, payload: Mapping[str, Any]) -> bool:
    new_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        old_text = path.read_text(encoding='utf-8')
        if old_text == new_text:
            return False
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding='utf-8')
    return True


def _json_text_safe(obj: Any, *, max_chars: int = 2_000_000) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        try:
            text = str(obj)
        except Exception:
            text = ""
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        try:
            if key in mapping and mapping.get(key) is not None:
                return mapping.get(key)
        except Exception:
            continue
    return None


def diagnose_suite_generation(
    suite_dir: Path | str,
    *,
    benchmark_set_json: Optional[Path | str] = None,
    bench_json_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect stale benchmark-suite generation that runner refresh cannot fix.

    ``refresh_suite_harness`` can replace Python runners in an old suite, but it
    cannot retroactively create graph artifacts that were missing at generation
    time.  For YOLO/Hailo this matters for the v41h+ host-tail path: old suites
    may have a Full raw-head HEF, but no ``part2_hailo_prefix`` and no
    ``part2_host_tail`` ONNX.  Running such a suite with a newer tool would still
    skip Hailo Part2 and look like the new fix did not work.  This diagnostic
    makes that state explicit before the remote bundle is uploaded.
    """

    suite_dir = Path(suite_dir).expanduser().resolve()
    bench_name = str(bench_json_name or "benchmark_set.json")
    try:
        if benchmark_set_json is not None:
            p = Path(benchmark_set_json).expanduser().resolve()
            bench_path = p if p.is_file() else suite_dir / bench_name
        else:
            bench_path = suite_dir / bench_name
    except Exception:
        bench_path = suite_dir / bench_name

    payload = _read_json_safe(bench_path) or {}
    text = _json_text_safe(payload)
    low = text.lower()

    generated_with = _first_present(
        payload,
        (
            "tool_version",
            "generator_version",
            "version",
            "tool",
            "created_by",
            "benchmark_set_version",
            "analysis_version",
        ),
    )
    if isinstance(generated_with, Mapping):
        generated_with = _first_present(generated_with, ("version", "name", "id")) or generated_with

    old_generation_markers = (
        "v41f" in low
        or "v41g" in low
        or "0.10.188" in low
        or "0.10.189" in low
        or "raw-head-baseline" in low and "part2_host_tail" not in low
    )
    has_yolo = "yolo" in low or "/model.23" in low or "/dfl/" in low
    has_hailo = "hailo" in low or "hef" in low
    has_raw_head = "raw_detection_head" in low or "raw-head" in low or "raw_head" in low
    has_host_tail_marker = any(
        marker in low
        for marker in (
            "part2_host_tail",
            "part2_host_tail_model",
            "part2_hailo_prefix",
            "part2_accel_model",
            "host_tail_model",
        )
    )

    case_count = 0
    cases_with_part2_host_tail = 0
    cases_with_raw_head_full = 0
    cases_with_yolo_hailo = 0
    sample_missing_host_tail: list[str] = []
    for manifest in sorted(suite_dir.glob("b*/split_manifest.json")):
        case_count += 1
        m = _read_json_safe(manifest) or {}
        mt = _json_text_safe(m, max_chars=250_000).lower()
        case_has_yolo = "yolo" in mt or "/model.23" in mt or "/dfl/" in mt or has_yolo
        case_has_hailo = "hailo" in mt or "hef" in mt
        case_has_raw = "raw_detection_head" in mt or "raw-head" in mt or "raw_head" in mt
        case_has_tail = any(
            marker in mt
            for marker in (
                "part2_host_tail",
                "part2_host_tail_model",
                "part2_hailo_prefix",
                "part2_accel_model",
                "host_tail_model",
            )
        )
        if case_has_yolo and case_has_hailo:
            cases_with_yolo_hailo += 1
        if case_has_raw and case_has_hailo:
            cases_with_raw_head_full += 1
        if case_has_tail:
            cases_with_part2_host_tail += 1
        if case_has_yolo and case_has_hailo and case_has_raw and not case_has_tail and len(sample_missing_host_tail) < 5:
            sample_missing_host_tail.append(manifest.parent.name)

    warnings: list[str] = []
    requires_regeneration = False
    if has_yolo and has_hailo and (old_generation_markers or (has_raw_head and not has_host_tail_marker)):
        requires_regeneration = True
        warnings.append(
            "This benchmark suite looks like an older YOLO/Hailo raw-head suite without "
            "v41h+ Part2 host-tail artifacts. Runner refresh can update Python code, but it cannot "
            "create missing part2_hailo_prefix/part2_host_tail ONNX or HEF files. Regenerate the "
            "benchmark set with this tool version before expecting Hailo Part2 host-tail runs."
        )
    if sample_missing_host_tail:
        warnings.append(
            "Cases with raw-head Hailo metadata but no Part2 host-tail manifest fields: "
            + ", ".join(sample_missing_host_tail)
        )

    return {
        "current_tool_version": str(_TOOL_VERSION),
        "benchmark_json": str(bench_path),
        "generated_with": str(generated_with or "unknown"),
        "has_yolo_markers": bool(has_yolo),
        "has_hailo_markers": bool(has_hailo),
        "has_raw_head_markers": bool(has_raw_head),
        "has_part2_host_tail_markers": bool(has_host_tail_marker or cases_with_part2_host_tail > 0),
        "old_generation_markers": bool(old_generation_markers),
        "case_count": int(case_count),
        "cases_with_yolo_hailo_markers": int(cases_with_yolo_hailo),
        "cases_with_raw_head_full_markers": int(cases_with_raw_head_full),
        "cases_with_part2_host_tail_markers": int(cases_with_part2_host_tail),
        "sample_cases_missing_part2_host_tail": sample_missing_host_tail,
        "requires_regeneration_for_part2_host_tail": bool(requires_regeneration),
        "warnings": warnings,
    }


def _path_looks_like_semantic_validation_source(path: Path) -> bool:
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    list_exts = {'.txt', '.json', '.jsonl'}
    try:
        if path.is_file():
            return path.suffix.lower() in image_exts | list_exts
        if path.is_dir():
            if (path / 'manifest.json').is_file() or (path / 'manifest.jsonl').is_file():
                return True
            for cand in path.rglob('*'):
                if cand.is_file() and cand.suffix.lower() in image_exts:
                    return True
            for cand in path.iterdir():
                if cand.is_file() and cand.suffix.lower() in list_exts:
                    return True
    except Exception:
        return False
    return False


def normalize_benchmark_task(value: Any, *, log: Optional[Callable[[str], None]] = None) -> str:
    raw = str(value or '').strip().lower() or 'auto'
    aliases = {
        'cls': 'classification',
        'classify': 'classification',
        'classification_logits': 'classification',
        'det': 'detection',
        'detect': 'detection',
        'object_detection': 'detection',
    }
    norm = aliases.get(raw, raw)
    if norm not in {'auto', 'detection', 'classification'}:
        _log(log, f"[info] Requested benchmark task {value!r} is not recognized; switching to 'auto'.")
        norm = 'auto'
    return norm


def normalize_mini_classification_eval(value: Any, *, log: Optional[Callable[[str], None]] = None) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(int(value))
    raw = str(value).strip().lower()
    if raw in {'1', 'true', 'yes', 'on', 'y'}:
        return True
    if raw in {'0', 'false', 'no', 'off', 'n', ''}:
        return False
    _log(log, f"[info] Requested Mini-Classification eval flag {value!r} is not recognized; switching to disabled.")
    return False


def normalize_semantic_validation_request(
    validation_images: Optional[str],
    validation_max_images: Optional[int],
    *,
    benchmark_task: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[Optional[str], int, bool]:
    """Normalize a requested semantic validation source.

    Detection suites default to the prepared COCO-50 bundle when no explicit
    validation source is configured. Classification suites prefer a locally
    imported preset such as ``imagenet_val_mini_200`` when available; otherwise
    an empty field still means "semantic dataset validation off" until the user
    provides a labeled classification set.
    """

    task = normalize_benchmark_task(benchmark_task, log=log)
    raw_images = str(validation_images or '').strip() or None
    try:
        max_images = int(validation_max_images) if validation_max_images is not None else 0
    except Exception:
        max_images = 0

    preset_name = normalize_classification_validation_preset(raw_images) if task == 'classification' else None
    if task == 'classification' and raw_images is None:
        default_preset = default_available_classification_validation_preset()
        if default_preset:
            raw_images = default_preset
            preset_name = default_preset
            _log(log, f"[info] Using local classification validation preset: {default_preset}")

    use_embedded = raw_images is None and task != 'classification'
    normalized_images = preset_name or raw_images

    if raw_images is not None and preset_name is None:
        try:
            cand = Path(os.path.expanduser(raw_images))
            looks_ok = _path_looks_like_semantic_validation_source(cand)
        except Exception:
            looks_ok = False
        if not looks_ok:
            if task == 'classification':
                _log(log, f"[info] Requested classification validation source {raw_images!r} is not usable; disabling dataset validation.")
                normalized_images = None
                use_embedded = False
            else:
                _log(log, f"[info] Requested semantic validation source {raw_images!r} is not usable; switching to prepared COCO-50.")
                normalized_images = None
                use_embedded = True

    if task == 'classification':
        max_images = max(0, int(max_images))
        if normalized_images and max_images <= 0:
            preset_default = 200 if str(normalized_images) == 'imagenet_val_mini_200' else (500 if str(normalized_images) == 'imagenet_val_mini_500' else 200)
            max_images = preset_default
    elif use_embedded:
        if max_images <= 0:
            max_images = 50
    else:
        max_images = max(0, int(max_images))

    return normalized_images, int(max_images), bool(use_embedded)


def normalize_validation_reference_mode(value: Optional[str], *, log: Optional[Callable[[str], None]] = None) -> str:
    raw = str(value or '').strip().lower() or 'auto'
    aliases = {
        'same-backend-full': 'same_backend_full',
        'same_backend': 'same_backend_full',
        'same-backend': 'same_backend_full',
        'gt': 'annotations',
        'ground_truth': 'annotations',
    }
    norm = aliases.get(raw, raw)
    if norm not in {'auto', 'same_backend_full', 'cpu_full', 'annotations'}:
        _log(log, f"[info] Requested validation reference mode {value!r} is not recognized; switching to 'auto'.")
        norm = 'auto'
    return norm


def normalize_mini_coco_ap50(value: Any, *, log: Optional[Callable[[str], None]] = None) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(int(value))
    raw = str(value).strip().lower()
    if raw in {'1', 'true', 'yes', 'on', 'y'}:
        return True
    if raw in {'0', 'false', 'no', 'off', 'n', ''}:
        return False
    _log(log, f"[info] Requested Mini-COCO AP50 flag {value!r} is not recognized; switching to disabled.")
    return False


def _normalize_suite_validation_payloads(
    suite_dir: Path,
    *,
    benchmark_set_json: Optional[Path],
    validation_images: Optional[str],
    validation_max_images: Optional[int],
    validation_reference_mode: Optional[str],
    mini_coco_ap50: Optional[bool],
    benchmark_task: Optional[str],
    mini_classification_eval: Optional[bool],
    log: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    """Patch stale or empty validation defaults in an existing suite.

    Refresh should be enough to switch old suites to the prepared COCO-50 dataset
    when no explicit semantic validation source is configured.
    """
    desired_task = normalize_benchmark_task(benchmark_task, log=log)
    desired_images, desired_max, use_embedded = normalize_semantic_validation_request(
        validation_images,
        validation_max_images,
        benchmark_task=desired_task,
        log=log,
    )
    desired_reference_mode = normalize_validation_reference_mode(validation_reference_mode, log=log)
    desired_mini_coco_ap50 = normalize_mini_coco_ap50(mini_coco_ap50, log=log)
    desired_mini_classification_eval = normalize_mini_classification_eval(mini_classification_eval, log=log)

    embedded_rel: Optional[str] = None
    embedded_created = False
    classification_rel_cache: Dict[str, Optional[str]] = {}
    changed = False
    patched_files: list[str] = []

    def _normalize_one_run(run: Dict[str, Any]) -> bool:
        nonlocal embedded_rel, embedded_created
        local_changed = False
        old_images = str(run.get('validation_images') or '').strip() or None
        try:
            old_max = int(run.get('validation_max_images') or 0)
        except Exception:
            old_max = 0

        effective_images = desired_images
        effective_max = desired_max

        if use_embedded:
            # Empty GUI field means: force the built-in suite-level dataset.
            # Do not preserve stale values from older suites such as /homes/kmika/Models.
            if embedded_rel is None:
                embedded_rel, embedded_created = _provision_embedded_semantic_validation_dataset(suite_dir)
                if embedded_rel:
                    _log(log, f"[info] Using prepared semantic validation dataset: {embedded_rel}")
            effective_images = embedded_rel
            effective_max = max(50, desired_max)
        else:
            effective_images = desired_images
            effective_max = desired_max
            if desired_task == 'classification' and str(desired_images or '').strip():
                cache_key = str(desired_images)
                if cache_key not in classification_rel_cache:
                    classification_rel_cache[cache_key] = provision_classification_validation_source_to_suite(
                        suite_dir,
                        desired_images,
                        base_dir=suite_dir,
                    )
                    if classification_rel_cache[cache_key]:
                        _log(log, f"[info] Using suite-local classification validation dataset: {classification_rel_cache[cache_key]}")
                if classification_rel_cache.get(cache_key):
                    effective_images = classification_rel_cache[cache_key]

        norm_images = str(effective_images or '')
        norm_max = int(max(0, int(effective_max or 0)))
        if str(run.get('validation_images') or '') != norm_images:
            run['validation_images'] = norm_images
            local_changed = True
        if int(run.get('validation_max_images') or 0) != norm_max:
            run['validation_max_images'] = norm_max
            local_changed = True
        if str(run.get('validation_reference_mode') or 'auto') != desired_reference_mode:
            run['validation_reference_mode'] = desired_reference_mode
            local_changed = True
        if bool(run.get('mini_coco_ap50')) != bool(desired_mini_coco_ap50):
            run['mini_coco_ap50'] = bool(desired_mini_coco_ap50)
            local_changed = True
        if str(run.get('benchmark_task') or 'auto') != desired_task:
            run['benchmark_task'] = desired_task
            local_changed = True
        if bool(run.get('mini_classification_eval')) != bool(desired_mini_classification_eval):
            run['mini_classification_eval'] = bool(desired_mini_classification_eval)
            local_changed = True
        return local_changed

    def _patch_runs(runs: Any) -> bool:
        local_changed = False
        if not isinstance(runs, list):
            return False
        for run in runs:
            if isinstance(run, dict) and _normalize_one_run(run):
                local_changed = True
        return local_changed

    plan_path = suite_dir / 'benchmark_plan.json'
    plan_payload = _read_json_safe(plan_path)
    if isinstance(plan_payload, dict) and _patch_runs(plan_payload.get('runs')):
        if _write_json_if_changed(plan_path, plan_payload):
            changed = True
            patched_files.append(plan_path.name)
            _log(log, f"[info] Normalized semantic validation defaults in {plan_path.name}")

    bench_path = benchmark_set_json if benchmark_set_json and benchmark_set_json.exists() else (suite_dir / 'benchmark_set.json')
    bench_payload = _read_json_safe(bench_path)
    if isinstance(bench_payload, dict):
        bench_changed = False
        if _patch_runs((bench_payload.get('plan') or {}).get('runs') if isinstance(bench_payload.get('plan'), dict) else None):
            bench_changed = True
        if _patch_runs(bench_payload.get('runs')):
            bench_changed = True
        if bench_changed and _write_json_if_changed(bench_path, bench_payload):
            changed = True
            patched_files.append(bench_path.name)
            _log(log, f"[info] Normalized semantic validation defaults in {bench_path.name}")

    if bool(embedded_created):
        changed = True
        _log(log, "[info] Provisioned suite-level prepared semantic validation dataset.")

    return {
        'validation_changed': bool(changed),
        'validation_images': embedded_rel if use_embedded else (classification_rel_cache.get(str(desired_images)) if (desired_task == 'classification' and str(desired_images or '').strip()) else desired_images),
        'validation_max_images': int(desired_max if desired_max is not None else 0),
        'validation_patched_files': patched_files,
        'validation_reference_mode': desired_reference_mode,
        'mini_coco_ap50': bool(desired_mini_coco_ap50),
        'benchmark_task': str(desired_task),
        'mini_classification_eval': bool(desired_mini_classification_eval),
        'validation_resource_provisioned': bool(embedded_created),
    }


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
    validation_images: Optional[str] = None,
    validation_max_images: Optional[int] = None,
    validation_reference_mode: Optional[str] = None,
    mini_coco_ap50: Optional[bool] = None,
    benchmark_task: Optional[str] = None,
    mini_classification_eval: Optional[bool] = None,
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
        "current_tool_version": str(_TOOL_VERSION),
        "suite_script_updated": False,
        "runner_lib_files_updated": 0,
        "case_runner_cases_updated": 0,
        "case_runner_files_updated": 0,
        "case_count": 0,
        "changed": False,
        "validation_changed": False,
        "validation_images": None,
        "validation_max_images": None,
        "validation_patched_files": [],
        "validation_reference_mode": "auto",
        "mini_coco_ap50": False,
        "benchmark_task": "auto",
        "mini_classification_eval": False,
        "suite_generation_warnings": [],
        "requires_regeneration_for_part2_host_tail": False,
        "suite_generation_status_changed": False,
    }

    generation_status = diagnose_suite_generation(
        suite_dir,
        benchmark_set_json=bench_path,
        bench_json_name=bench_json_name,
    )
    stats["suite_generation_status"] = generation_status
    stats["suite_generation_warnings"] = list(generation_status.get("warnings") or [])
    stats["requires_regeneration_for_part2_host_tail"] = bool(generation_status.get("requires_regeneration_for_part2_host_tail"))
    try:
        status_changed = _write_json_if_changed(suite_dir / "suite_generation_status.json", generation_status)
        if status_changed:
            stats["suite_generation_status_changed"] = True
            stats["changed"] = True
    except Exception:
        pass
    for warning in stats["suite_generation_warnings"]:
        _log(log, f"[suite-check][warn] {warning}")
    if bool(stats["requires_regeneration_for_part2_host_tail"]):
        _log(
            log,
            "[suite-check][warn] Regenerate this benchmark set; refreshing an old suite can update runners "
            "but cannot add missing part2_hailo_prefix/part2_host_tail graph artifacts.",
        )

    norm = _normalize_suite_validation_payloads(
        suite_dir,
        benchmark_set_json=bench_path,
        validation_images=validation_images,
        validation_max_images=validation_max_images,
        validation_reference_mode=validation_reference_mode,
        mini_coco_ap50=mini_coco_ap50,
        benchmark_task=benchmark_task,
        mini_classification_eval=mini_classification_eval,
        log=log,
    )
    stats.update(norm)
    if bool(norm.get('validation_changed')):
        stats['changed'] = True

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
