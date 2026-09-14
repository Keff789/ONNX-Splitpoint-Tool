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


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _embedded_semantic_validation_dataset_source() -> Optional[Path]:
    """Return the prepared/default COCO-50 validation resource if present."""
    return default_detection_validation_source()


def _provision_embedded_semantic_validation_dataset(
    suite_dir: Path,
    preset: str = "coco_50",
    *,
    max_images: int = 0,
    manifest_path: str = "",
) -> tuple[Optional[str], bool]:
    """Provision the bounded detection subset exactly once at suite level."""
    root = Path(suite_dir) / 'resources' / 'validation' / 'detection'
    before = {p.name for p in root.iterdir()} if root.is_dir() else set()
    rel = provision_detection_validation_source_to_suite(
        Path(suite_dir),
        preset=preset,
        base_dir=Path(suite_dir),
        max_images=max(0, int(max_images or 0)),
        manifest_path=manifest_path,
    )
    after_path = (Path(suite_dir) / str(rel)).resolve() if rel else None
    return rel, bool(after_path and after_path.exists() and after_path.name not in before)


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
    cases_requiring_part2_host_tail = 0
    cases_with_explicit_hailo_availability = 0
    sample_missing_host_tail: list[str] = []

    def _case_names(case: Mapping[str, Any]) -> set[str]:
        names: set[str] = set()
        for key in ("case_id", "case_dir", "folder"):
            v = str(case.get(key) or "").strip()
            if v:
                names.add(v)
        b = case.get("boundary") if case.get("boundary") is not None else case.get("split_index")
        try:
            names.add(f"b{int(b):03d}")
            names.add(f"b{int(b)}")
        except Exception:
            pass
        return names

    # New benchmarksets can intentionally have YOLO/Hailo raw-head Full/Part1
    # artifacts without a Hailo Part2 host-tail artifact.  That is valid for
    # Hailo->TensorRT/host-tail cases and must not be reported as an old-suite
    # warning.  Only cases whose availability says Hailo Part2/composed exists
    # actually require the v41h+ Part2 host-tail manifest fields.
    case_requires_tail: dict[str, Optional[bool]] = {}
    for raw_case in _as_list(payload.get("cases")):
        if not isinstance(raw_case, Mapping):
            continue
        hav_all = raw_case.get("hailo_case_variant_availability")
        if not isinstance(hav_all, Mapping):
            continue
        requires: Optional[bool] = None
        for backend, availability in hav_all.items():
            if "hailo" not in str(backend).lower() or not isinstance(availability, Mapping):
                continue
            cases_with_explicit_hailo_availability += 1
            requires = bool(availability.get("part2") or availability.get("composed"))
            break
        if requires is not None:
            for name in _case_names(raw_case):
                case_requires_tail[name] = requires
            if requires:
                cases_requiring_part2_host_tail += 1

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
        requires_tail = case_requires_tail.get(manifest.parent.name)
        if case_has_yolo and case_has_hailo and case_has_raw and not case_has_tail:
            # With explicit availability data, missing host-tail markers are only
            # suspicious for cases that actually claim Hailo Part2/composed
            # support.  Without availability data, keep the old conservative check.
            if (requires_tail is True) or (requires_tail is None and not case_requires_tail):
                if len(sample_missing_host_tail) < 5:
                    sample_missing_host_tail.append(manifest.parent.name)

    effective_has_host_tail_marker = bool(has_host_tail_marker or cases_with_part2_host_tail > 0)
    has_cases_that_require_tail = bool(cases_requiring_part2_host_tail > 0 or not case_requires_tail)

    warnings: list[str] = []
    requires_regeneration = False
    if has_yolo and has_hailo and has_cases_that_require_tail and (old_generation_markers or (has_raw_head and not effective_has_host_tail_marker)):
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
        "cases_requiring_part2_host_tail": int(cases_requiring_part2_host_tail),
        "cases_with_explicit_hailo_availability": int(cases_with_explicit_hailo_availability),
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


def _effective_override(value: Any) -> str:
    s = str(value or '').strip()
    return '' if s.lower() in {'', 'auto', 'none', 'null', 'default'} else s


def _task_gated_flags(task: str, mini_coco: bool, mini_cls: bool) -> tuple[bool, bool]:
    task_l = normalize_benchmark_task(task)
    if task_l == 'classification':
        return False, bool(mini_cls)
    if task_l == 'detection':
        return bool(mini_coco), False
    return False, False




def _infer_task_from_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "auto"
    blobs: list[str] = []
    for key in (
        "task",
        "benchmark_task",
        "model_task",
        "model_name",
        "model_id",
        "model",
        "model_source",
        "source_model",
        "source_model_path",
        "source_onnx",
        "onnx",
        "onnx_path",
        "family",
    ):
        val = payload.get(key)
        if val:
            blobs.append(str(val))
    model_obj = payload.get("model")
    if isinstance(model_obj, dict):
        for key in ("task", "family", "id", "name", "path", "onnx", "onnx_path", "source_model", "source_model_path"):
            val = model_obj.get(key)
            if val:
                blobs.append(str(val))
    # Some benchmark_set.json files store model entries under lists/dicts.
    for key in ("models", "model_suite", "primary", "reserve"):
        val = payload.get(key)
        if isinstance(val, dict):
            blobs.append(json.dumps(val, sort_keys=True)[:4000])
        elif isinstance(val, list):
            blobs.append(json.dumps(val[:10], sort_keys=True)[:4000])
    blob = " ".join(blobs).lower()
    if any(tok in blob for tok in ("yolo", "coco", "detect", "detection", "object_detection")):
        return "detection"
    if any(tok in blob for tok in ("resnet", "mobilenet", "regnet", "efficientnet", "convnext", "imagenet", "imagenette", "classification", "classify")):
        return "classification"
    return "auto"

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
        if task != 'classification' and str(raw_images).strip().lower() in {'coco', 'coco_50', 'coco50', 'coco_200', 'coco200'}:
            looks_ok = True
        else:
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
            max_images = 200 if str(os.environ.get('ONNX_SPLITPOINT_DETECTION_VALIDATION_PRESET') or 'coco_200').lower() in {'coco_200','coco200'} else 50
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
    task_override_explicit = bool(_effective_override(benchmark_task))
    images_override_explicit = bool(_effective_override(validation_images))
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
    classification_rel_cache: Dict[tuple[str, int, str], Optional[str]] = {}
    detection_rel_cache: Dict[tuple[str, int, str], Optional[str]] = {}
    effective_return_images: Optional[str] = desired_images
    effective_return_max: int = int(desired_max or 0)
    changed = False
    patched_files: list[str] = []

    # v52f: old/manual suites may have per-run benchmark_task=auto while the
    # suite-level model name clearly identifies ResNet/MobileNet/RegNet.
    # Keep this as a hint so classification runs are not left on COCO-50.
    suite_task_hint = "auto"
    try:
        bench_hint_path = benchmark_set_json if benchmark_set_json and benchmark_set_json.exists() else (suite_dir / "benchmark_set.json")
        hint_payload = _read_json_safe(bench_hint_path)
        suite_task_hint = _infer_task_from_payload(hint_payload)
    except Exception:
        suite_task_hint = "auto"

    def _normalize_one_run(run: Dict[str, Any]) -> bool:
        nonlocal embedded_rel, embedded_created, effective_return_images, effective_return_max
        local_changed = False
        try:
            old_max = int(run.get('validation_max_images') or 0)
        except Exception:
            old_max = 0
        budget_declared = bool(run.get('validation_budget_authoritative')) or ('validation_items_requested' in run)
        try:
            authoritative_budget = max(0, int(run.get('validation_items_requested') or 0))
        except Exception:
            authoritative_budget = 0

        # v52d: when the CLI/profile task is "auto", preserve each run's
        # benchmark_task instead of overwriting classification plans with COCO
        # defaults.  This is the core guard against accidental COCO validation on
        # ResNet/MobileNet/RegNet runs.
        current_task = normalize_benchmark_task(run.get('benchmark_task') or run.get('task') or 'auto', log=log)
        run_task = desired_task if task_override_explicit and desired_task != 'auto' else current_task
        if run_task == 'auto' and suite_task_hint != 'auto':
            run_task = suite_task_hint
        if run_task == 'auto':
            rid_blob = ' '.join(str(run.get(k) or '') for k in ('id', 'name', 'backend', 'provider'))
            if 'yolo' in rid_blob.lower() or 'detect' in rid_blob.lower():
                run_task = 'detection'

        # Per-run dataset routing.  A generated Evaluation Workflow run binds an
        # exact, already materialised subset through
        # ``validation_budget_authoritative``/``validation_items_requested``.
        # That binding covers the source *and* its cardinality.  Treating a
        # target-level legacy preset (for example ``coco_50``) as a later source
        # override created a second alias manifest on remote workers while the
        # management CPU reference kept the originally selected manifest.  The
        # two files described the same Image IDs and ground truth but had
        # different byte hashes, so the strict quality-contract comparison
        # correctly rejected every pair.  Preserve the exact per-run artefact;
        # ad-hoc/manual plans without the authoritative marker retain the legacy
        # override behaviour below.
        existing_images = str(run.get('validation_images') or '').strip()
        if budget_declared and existing_images:
            effective_images = existing_images
            effective_max = authoritative_budget
            local_use_embedded = existing_images.replace('\\', '/').startswith('resources/validation/')
        elif images_override_explicit or (task_override_explicit and desired_task != 'auto'):
            effective_images = desired_images
            effective_max = desired_max
            local_use_embedded = use_embedded
        else:
            if run_task == 'classification':
                if 'coco_50' in existing_images.lower() or not existing_images:
                    default_cls = default_available_classification_validation_preset(base_dir=suite_dir) or 'imagenette_val_mini_200'
                    effective_images = default_cls
                    effective_max = desired_max or old_max or (500 if default_cls.endswith('_500') else 200)
                else:
                    effective_images = existing_images
                    effective_max = desired_max or old_max or 200
                local_use_embedded = False
            elif run_task == 'detection':
                if existing_images and existing_images.lower() not in {'coco_50', 'coco50', 'coco_50_data', 'coco_200', 'coco200', 'coco_200_data'}:
                    effective_images = existing_images
                    effective_max = desired_max or old_max or 50
                    local_use_embedded = False
                else:
                    effective_images = existing_images if existing_images else (desired_images if desired_images else 'coco_50')
                    effective_max = desired_max or old_max or (200 if '200' in str(effective_images) else 50)
                    local_use_embedded = True
            else:
                effective_images = existing_images or desired_images
                effective_max = old_max or desired_max
                local_use_embedded = False

        # v60r: the per-run value materialised from Smoke/Standard/Final is
        # authoritative.  This is deliberately applied after all legacy
        # source-routing branches so COCO-50/Imagenette defaults cannot expand
        # a 12/16 item run back to 50.
        if budget_declared:
            effective_max = authoritative_budget

        if run_task == 'classification':
            if str(effective_images or '').strip():
                manifest_ref = str(run.get('validation_manifest') or '')
                cache_key = (str(effective_images), int(effective_max or 0), manifest_ref)
                if cache_key not in classification_rel_cache:
                    classification_rel_cache[cache_key] = provision_classification_validation_source_to_suite(
                        suite_dir,
                        effective_images,
                        base_dir=suite_dir,
                        max_images=max(0, int(effective_max or 0)),
                        manifest_path=manifest_ref,
                    )
                    if classification_rel_cache[cache_key]:
                        _log(log, f"[info] Using suite-local classification validation dataset: {classification_rel_cache[cache_key]}")
                if classification_rel_cache.get(cache_key):
                    effective_images = classification_rel_cache[cache_key]
                elif 'coco_50' in str(effective_images or '').lower():
                    # Strict guard: never leave COCO wired into a classification run.
                    effective_images = ''
            local_use_embedded = False
        elif run_task == 'detection':
            manifest_ref = str(run.get('validation_manifest') or '')
            request = str(effective_images or desired_images or 'coco_50')
            cache_key = (request, int(effective_max or 0), manifest_ref)
            if cache_key not in detection_rel_cache:
                before_root = suite_dir / 'resources' / 'validation' / 'detection'
                before_names = {p.name for p in before_root.iterdir()} if before_root.is_dir() else set()
                detection_rel_cache[cache_key] = provision_detection_validation_source_to_suite(
                    suite_dir,
                    request,
                    base_dir=suite_dir,
                    max_images=max(0, int(effective_max or 0)),
                    manifest_path=manifest_ref,
                )
                rel_now = detection_rel_cache[cache_key]
                if rel_now:
                    target_now = suite_dir / str(rel_now)
                    embedded_created = embedded_created or (target_now.name not in before_names)
                    _log(log, f"[info] Using suite-local detection validation dataset: {rel_now}")
            if detection_rel_cache.get(cache_key):
                effective_images = detection_rel_cache[cache_key]
                embedded_rel = str(effective_images)
            local_use_embedded = bool(effective_images and str(effective_images).replace('\\', '/').startswith('resources/validation/'))

        norm_images = str(effective_images or '')
        norm_max = int(max(0, int(effective_max or 0)))
        if norm_images:
            effective_return_images = norm_images
            effective_return_max = norm_max
        mini_coco_run, mini_cls_run = _task_gated_flags(
            run_task,
            desired_mini_coco_ap50 if task_override_explicit else normalize_mini_coco_ap50(run.get('mini_coco_ap50'), log=log),
            desired_mini_classification_eval if task_override_explicit else normalize_mini_classification_eval(run.get('mini_classification_eval'), log=log),
        )
        if run_task == 'classification' and not mini_cls_run and norm_images:
            mini_cls_run = True
        if run_task == 'detection' and not mini_coco_run and norm_images:
            mini_coco_run = True

        if str(run.get('validation_images') or '') != norm_images:
            run['validation_images'] = norm_images
            local_changed = True
        if int(run.get('validation_max_images') or 0) != norm_max:
            run['validation_max_images'] = norm_max
            local_changed = True
        if budget_declared and not bool(run.get('validation_budget_authoritative')):
            run['validation_budget_authoritative'] = True
            local_changed = True
        if str(run.get('validation_reference_mode') or 'auto') != desired_reference_mode:
            run['validation_reference_mode'] = desired_reference_mode
            local_changed = True
        if bool(run.get('mini_coco_ap50')) != bool(mini_coco_run):
            run['mini_coco_ap50'] = bool(mini_coco_run)
            local_changed = True
        if str(run.get('benchmark_task') or 'auto') != run_task:
            run['benchmark_task'] = run_task
            local_changed = True
        if bool(run.get('mini_classification_eval')) != bool(mini_cls_run):
            run['mini_classification_eval'] = bool(mini_cls_run)
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
        'validation_images': effective_return_images,
        'validation_max_images': int(effective_return_max),
        'validation_patched_files': patched_files,
        'validation_reference_mode': desired_reference_mode,
        'mini_coco_ap50': bool(desired_mini_coco_ap50 if desired_task == 'detection' else False),
        'benchmark_task': str(desired_task),
        'mini_classification_eval': bool(desired_mini_classification_eval if desired_task == 'classification' else False),
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

    if "class HailoInferModelSession" in text:
        from ..split_export_runners import assert_generated_hailo_layout_current
        assert_generated_hailo_layout_current(path)

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

    suite_runtime_import = text.find(
        "from splitpoint_runners.native_split_quality_runtime import"
    )
    suite_path_bootstrap = text.find(
        "\n_maybe_add_suite_runtime_to_syspath()\n"
    )
    if suite_runtime_import >= 0 and (
        suite_path_bootstrap < 0 or suite_path_bootstrap > suite_runtime_import
    ):
        raise RuntimeError(
            f"{path} imports the suite-owned quality runtime before the suite root "
            "is added to sys.path. Refusing to keep a remotely unimportable runner."
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
        "scientific_reporter_updated": False,
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

        # write_benchmark_suite_script also vendors the self-contained v60
        # reporter.  Older refresh logic copied only benchmark_suite.py, which
        # made remote suites fail with ModuleNotFoundError.
        src_reporter = tmp_dir / "scientific_reporter_v60.py"
        dst_reporter = suite_dir / "scientific_reporter_v60.py"
        if src_reporter.is_file() and _copy_if_changed(src_reporter, dst_reporter):
            stats["scientific_reporter_updated"] = True
            stats["changed"] = True
            _log(log, f"[info] Refreshed scientific_reporter_v60.py: {dst_reporter}")
        if not dst_reporter.is_file():
            raise RuntimeError(
                "Suite refresh did not produce scientific_reporter_v60.py; "
                "the generated benchmark suite would be incomplete."
            )

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

                if fname == "run_split_onnxruntime.py" and dst.exists():
                    # v53g: older benchmark suites may contain a syntactically
                    # self-consistent runner that predates DeepX provider tokens.
                    # The old runner passes the generic self-consistency check,
                    # but remote TensorRT->DeepX dispatch then fails with
                    # argparse: invalid choice 'deepx_m1'.  Treat that as stale
                    # and force a current runner copy.
                    try:
                        _dst_text = dst.read_text(encoding="utf-8", errors="ignore")
                        required_runtime_markers = (
                            "deepx_m1",
                            "dx_m1",
                            "--phase-runs",
                            "--external-output-dump",
                            "run_stage2_prefix_map",
                            "_run_full_variant_outputs_map",
                        )
                        if any(marker not in _dst_text for marker in required_runtime_markers):
                            dst_needs_repair = True
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
