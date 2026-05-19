from __future__ import annotations

"""Versioned evaluation profile helpers for benchmark generation.

Profiles describe compact model suites and associated run matrices. They are
implemented as data rather than as hardcoded logic so GUI/CLI workflows can
reuse the same presets for thesis evaluation, smoke regression, and future
campaigns.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml
from jsonschema import Draft202012Validator

_PROFILE_ALIASES = {
    'final': 'final_splitpoint_evaluation_v1',
    'final_v1': 'final_splitpoint_evaluation_v1',
    'final-splitpoint-evaluation-v1': 'final_splitpoint_evaluation_v1',
    'final_splitpoint_evaluation_v1': 'final_splitpoint_evaluation_v1',
    'thesis': 'final_splitpoint_evaluation_v1',
    'thesis_final': 'final_splitpoint_evaluation_v1',
    'smoke': 'smoke_regression_v1',
    'smoke_v1': 'smoke_regression_v1',
    'smoke-regression-v1': 'smoke_regression_v1',
    'smoke_regression_v1': 'smoke_regression_v1',
    'smoke_regression_evaluation_v1': 'smoke_regression_v1',
    'regression': 'smoke_regression_v1',
}


@dataclass
class LoadedEvaluationProfile:
    profile_id: str
    profile_path: str
    source: str
    raw_profile: Dict[str, Any]


@dataclass
class EvaluationProfileComparison:
    profiles: List[Dict[str, Any]]
    model_rows: List[Dict[str, Any]]
    run_rows: List[Dict[str, Any]]
    summary_markdown: str


@dataclass
class EvaluationProfileResolution:
    requested: str
    profile_id: str
    profile_path: str
    source: str
    profile_name: str
    version: Optional[int]
    matched: bool
    matched_tier: Optional[str]
    matched_model_id: Optional[str]
    matched_model_family: Optional[str]
    matched_task: Optional[str]
    model_tokens: List[str]
    run_profile_ids: List[str]
    selection_policy: Dict[str, Any]
    overrides: Dict[str, Any]
    raw_profile: Dict[str, Any]
    summary: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            'requested': self.requested,
            'profile_id': self.profile_id,
            'profile_name': self.profile_name,
            'profile_path': self.profile_path,
            'source': self.source,
            'version': self.version,
            'matched': bool(self.matched),
            'matched_tier': self.matched_tier,
            'matched_model_id': self.matched_model_id,
            'matched_model_family': self.matched_model_family,
            'matched_task': self.matched_task,
            'model_tokens': list(self.model_tokens),
            'run_profile_ids': list(self.run_profile_ids),
            'selection_policy': dict(self.selection_policy),
            'overrides': {k: v for k, v in dict(self.overrides).items() if k != 'raw_profile'},
            'summary': self.summary,
        }


def evaluation_profile_default_root() -> Path:
    return Path(__file__).resolve().parent.parent / 'resources' / 'evaluation_profiles'


def evaluation_profile_schema_path() -> Path:
    return Path(__file__).resolve().parent.parent / 'resources' / 'schemas' / 'evaluation_profile.schema.json'


def _load_profile_schema() -> Dict[str, Any]:
    schema_path = evaluation_profile_schema_path()
    try:
        return json.loads(schema_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise RuntimeError(f'Could not load evaluation profile schema: {schema_path} ({exc})') from exc


def _profile_validator() -> Draft202012Validator:
    return Draft202012Validator(_load_profile_schema())


def _norm_token(value: Any) -> str:
    s = str(value or '').strip().lower()
    s = Path(s).stem.lower() if any(ch in s for ch in '/\\.') else s
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')


def normalize_evaluation_profile_request(value: Any) -> str:
    tok = _norm_token(value)
    return _PROFILE_ALIASES.get(tok, tok)


def list_available_evaluation_profiles(base_dir: Optional[Path] = None) -> List[str]:
    root = Path(base_dir or evaluation_profile_default_root())
    names: List[str] = []
    if root.is_dir():
        for p in sorted(root.glob('*.yml')) + sorted(root.glob('*.yaml')):
            names.append(normalize_evaluation_profile_request(p.stem))
    out: List[str] = []
    seen = set()
    for name in names:
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def resolve_evaluation_profile_source(request: Any, *, base_dir: Optional[Path] = None) -> Optional[Path]:
    raw = str(request or '').strip()
    if not raw:
        return None
    cand = Path(raw).expanduser()
    if cand.is_file():
        return cand.resolve()
    root = Path(base_dir or evaluation_profile_default_root())
    tok = normalize_evaluation_profile_request(raw)
    for ext in ('.yaml', '.yml'):
        p = root / f'{tok}{ext}'
        if p.is_file():
            return p.resolve()
    if root.is_dir():
        for p in sorted(root.glob('*.yaml')) + sorted(root.glob('*.yml')):
            if normalize_evaluation_profile_request(p.stem) == tok:
                return p.resolve()
    return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    return dict(data or {}) if isinstance(data, Mapping) else {}


def _format_schema_error(err: Any) -> str:
    loc = ''.join(f'[{repr(x)}]' if isinstance(x, int) else f'.{x}' for x in list(err.path or []))
    if loc.startswith('.'):
        loc = loc[1:]
    where = f' at {loc}' if loc else ''
    return f'{err.message}{where}'


def validate_evaluation_profile_payload(
    payload: Mapping[str, Any],
    *,
    source: str = '<memory>',
    source_path: str | Path | None = None,
    strict_external: Optional[bool] = None,
) -> Dict[str, Any]:
    """Validate an evaluation profile payload.

    `strict_external` is accepted for backwards compatibility with older GUI
    code paths. Validation is hard for all profiles now, so the flag is ignored.
    """
    source_label = str(source_path or source)
    data = dict(payload or {}) if isinstance(payload, Mapping) else {}
    errors = sorted(_profile_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        head = '; '.join(_format_schema_error(e) for e in errors[:6])
        more = f' (+{len(errors) - 6} more)' if len(errors) > 6 else ''
        raise ValueError(f'Invalid evaluation profile {source_label}: {head}{more}')
    return data


def load_evaluation_profile(
    request: Any,
    *,
    base_dir: Optional[Path] = None,
    validate: bool = True,
    strict_external: Optional[bool] = None,
):
    """Load an evaluation profile.

    Default return value is `LoadedEvaluationProfile`. For backwards
    compatibility, when `strict_external` is provided this function returns a
    tuple `(raw_profile_dict, validation_result)` where `validation_result`
    exposes `.warnings`.
    """
    src = resolve_evaluation_profile_source(request, base_dir=base_dir)
    if src is None:
        return None
    raw = _load_yaml(src)
    if validate:
        raw = validate_evaluation_profile_payload(raw, source=str(src), strict_external=strict_external)
    root = Path(base_dir or evaluation_profile_default_root()).resolve()
    source = 'builtin'
    try:
        src.relative_to(root)
    except Exception:
        source = 'file'
    profile_id = str(raw.get('name') or src.stem).strip() or src.stem
    loaded = LoadedEvaluationProfile(profile_id=profile_id, profile_path=str(src), source=source, raw_profile=raw)
    if strict_external is not None:
        return dict(loaded.raw_profile), SimpleNamespace(warnings=[])
    return loaded


def save_evaluation_profile_yaml(path: str | Path, payload: Mapping[str, Any], *, validate: bool = True) -> Path:
    data = dict(payload or {}) if isinstance(payload, Mapping) else {}
    if validate:
        data = validate_evaluation_profile_payload(data, source=str(path))
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    dst.write_text(text, encoding='utf-8')
    return dst


def _export_metadata_candidates(model_path: Path) -> List[Path]:
    base = model_path.with_suffix('')
    return [
        base.with_suffix('.export.json'),
        model_path.with_name(base.name + '.export.json'),
        model_path.with_suffix(model_path.suffix + '.export.json'),
    ]


def load_export_metadata_for_model(model_path: str | Path) -> Dict[str, Any]:
    p = Path(model_path).expanduser()
    for cand in _export_metadata_candidates(p):
        try:
            if cand.is_file():
                data = json.loads(cand.read_text(encoding='utf-8'))
                return dict(data or {}) if isinstance(data, Mapping) else {}
        except Exception:
            continue
    return {}


def _task_from_export_meta(export_meta: Mapping[str, Any], model_path: Path) -> str:
    task = str(export_meta.get('task_type') or '').strip().lower()
    if task in {'classification', 'detection'}:
        return task
    ul_task = str(export_meta.get('ultralytics_task') or '').strip().lower()
    if ul_task == 'detect':
        return 'detection'
    if ul_task in {'classify', 'classification'}:
        return 'classification'
    name = _norm_token(model_path.stem)
    if name.startswith('yolo'):
        return 'detection'
    return 'auto'


def _infer_validation_images(task: str, entry: Mapping[str, Any]) -> tuple[str, int, str]:
    if task == 'classification':
        subset = str(entry.get('development_subset') or 'imagenet_val_mini_200').strip() or 'imagenet_val_mini_200'
        max_images = 500 if subset == 'imagenet_val_mini_500' else 200
        return subset, int(max_images), 'imagenet'
    return '', 50, 'auto'


def _derive_run_flags(run_profiles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    acc_cpu = False
    acc_cuda = False
    acc_trt = False
    acc_h8 = False
    acc_h10 = False
    matrix_trt_to_hailo = False
    matrix_hailo_to_trt = False
    needs_hailo_same_backend = False
    run_ids: List[str] = []
    for raw in list(run_profiles or []):
        rp = dict(raw or {})
        rid = str(rp.get('id') or '').strip()
        if rid:
            run_ids.append(rid)
        typ = str(rp.get('type') or '').strip().lower()
        full = _norm_token(rp.get('full') or rp.get('full_reference') or '')
        st1 = _norm_token(rp.get('stage1') or '')
        st2 = _norm_token(rp.get('stage2') or '')
        if full == 'cpu' or rid == 'ort_cpu':
            acc_cpu = True
        if full == 'cuda' or st1 == 'cuda' or st2 == 'cuda' or rid == 'ort_cuda':
            acc_cuda = True
        if full in {'tensorrt', 'trt'} or st1 in {'tensorrt', 'trt'} or st2 in {'tensorrt', 'trt'} or rid == 'ort_tensorrt':
            acc_trt = True
        hailo_tokens = {full, st1, st2, _norm_token(rid)}
        if any(tok.startswith('hailo8') for tok in hailo_tokens if tok):
            acc_h8 = True
        if any(tok.startswith('hailo10') for tok in hailo_tokens if tok):
            acc_h10 = True
        if typ == 'same_backend_reference' and any(tok.startswith('hailo') for tok in hailo_tokens if tok):
            needs_hailo_same_backend = True
        if st1 in {'tensorrt', 'trt'} and any(tok.startswith('hailo') for tok in (st2, full, _norm_token(rid)) if tok):
            matrix_trt_to_hailo = True
        if any(tok.startswith('hailo') for tok in (st1, full, _norm_token(rid)) if tok) and st2 in {'tensorrt', 'trt'}:
            matrix_hailo_to_trt = True
    hailo_preset = 'End-to-end compare' if needs_hailo_same_backend else 'Custom'
    return {
        'acc_cpu': bool(acc_cpu),
        'acc_cuda': bool(acc_cuda),
        'acc_trt': bool(acc_trt),
        'acc_h8': bool(acc_h8),
        'acc_h10': bool(acc_h10),
        'matrix_trt_to_hailo': bool(matrix_trt_to_hailo),
        'matrix_hailo_to_trt': bool(matrix_hailo_to_trt),
        'hailo_preset': hailo_preset,
        'hailo_custom_full': bool(needs_hailo_same_backend),
        'hailo_custom_composed': True,
        'hailo_custom_part1': False,
        'hailo_custom_part2': False,
        'run_profile_ids': run_ids,
    }


def _profile_version(profile_id: str) -> Optional[int]:
    m = re.search(r'_v(\d+)$', _norm_token(profile_id))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def profile_model_entries(profile: Mapping[str, Any], *, include_reserve: bool = False) -> List[Dict[str, Any]]:
    suite = profile.get('model_suite') if isinstance(profile, Mapping) else None
    suite = suite if isinstance(suite, Mapping) else {}
    entries: List[Dict[str, Any]] = []
    for tier in ('primary', 'reserve'):
        if tier == 'reserve' and not include_reserve:
            continue
        for item in list(suite.get(tier) or []):
            if isinstance(item, Mapping):
                row = dict(item)
            else:
                row = {'id': str(item)}
            row.setdefault('id', str(row.get('id') or '').strip())
            row['_tier'] = tier
            if str(row.get('id') or '').strip():
                entries.append(row)
    return entries


def _match_model_entry(entry: Mapping[str, Any], model_path: Path, export_meta: Mapping[str, Any]) -> bool:
    entry_id = _norm_token(entry.get('id'))
    if not entry_id:
        return False
    candidates = {
        _norm_token(model_path.stem),
        _norm_token(export_meta.get('model_name')),
        _norm_token(export_meta.get('model_ref')),
    }
    return entry_id in {tok for tok in candidates if tok}


def find_profile_model_paths(profile: Mapping[str, Any], model_root: str | Path, *, include_reserve: bool = False) -> List[Dict[str, Any]]:
    root = Path(model_root).expanduser().resolve()
    files = sorted(root.rglob('*.onnx')) if root.is_dir() else []
    entries = profile_model_entries(profile, include_reserve=include_reserve)
    out: List[Dict[str, Any]] = []
    taken: set[str] = set()
    for entry in entries:
        matched_path: Optional[Path] = None
        matched_meta: Dict[str, Any] = {}
        for p in files:
            if str(p) in taken:
                continue
            meta = load_export_metadata_for_model(p)
            if _match_model_entry(entry, p, meta):
                matched_path = p
                matched_meta = meta
                break
        row = dict(entry)
        row['resolved_path'] = str(matched_path) if matched_path is not None else ''
        row['resolved'] = bool(matched_path is not None)
        row['export_metadata'] = dict(matched_meta or {})
        out.append(row)
        if matched_path is not None:
            taken.add(str(matched_path))
    return out


def resolve_evaluation_profile(
    request: Any,
    *,
    model_path: str | Path | None = None,
    export_metadata: Optional[Mapping[str, Any]] = None,
    base_dir: Optional[Path] = None,
) -> Optional[EvaluationProfileResolution]:
    loaded = load_evaluation_profile(request, base_dir=base_dir, validate=True)
    if loaded is None or isinstance(loaded, tuple):
        return None
    raw = dict(loaded.raw_profile or {})
    profile_id = loaded.profile_id
    requested = str(request or profile_id).strip()
    model_p = Path(model_path).expanduser().resolve() if model_path else None
    export_meta = dict(export_metadata or {}) if export_metadata else (load_export_metadata_for_model(model_p) if model_p else {})
    tokens: List[str] = []
    for raw_tok in (
        export_meta.get('model_name'),
        export_meta.get('model_ref'),
        export_meta.get('weights'),
        export_meta.get('source'),
        model_p.stem if model_p else None,
        model_p.name if model_p else None,
    ):
        tok = _norm_token(raw_tok)
        if tok and tok not in tokens:
            tokens.append(tok)
    task_from_meta = _task_from_export_meta(export_meta, model_p or Path('model.onnx'))

    primary = list(((raw.get('model_suite') or {}).get('primary') or [])) if isinstance(raw.get('model_suite'), Mapping) else []
    reserve = list(((raw.get('model_suite') or {}).get('reserve') or [])) if isinstance(raw.get('model_suite'), Mapping) else []

    matched_entry: Dict[str, Any] = {}
    matched_tier: Optional[str] = None
    for item in primary:
        if not isinstance(item, Mapping):
            continue
        mid = _norm_token(item.get('id'))
        if mid and mid in tokens:
            matched_entry = dict(item)
            matched_tier = 'primary'
            break
    if not matched_entry:
        for item in reserve:
            if isinstance(item, Mapping):
                mid = _norm_token(item.get('id'))
                item_map = dict(item)
            else:
                mid = _norm_token(item)
                item_map = {'id': str(item)}
            if mid and mid in tokens:
                matched_entry = item_map
                matched_tier = 'reserve'
                break

    matched = bool(matched_entry)
    matched_task = str(matched_entry.get('task') or task_from_meta or 'auto').strip().lower() if matched_entry else None
    if matched_task not in {'classification', 'detection'}:
        matched_task = task_from_meta if task_from_meta in {'classification', 'detection'} else None
    run_profiles = list(raw.get('run_profiles') or []) if isinstance(raw.get('run_profiles'), list) else []
    run_flags = _derive_run_flags(run_profiles)
    selection_policy = dict(raw.get('selection_policy') or {}) if isinstance(raw.get('selection_policy'), Mapping) else {}
    model_preparation = dict(raw.get('model_preparation') or {}) if isinstance(raw.get('model_preparation'), Mapping) else {}
    overrides: Dict[str, Any] = dict(run_flags)
    prep_mode = str(model_preparation.get('mode') or '').strip()
    if prep_mode:
        overrides['model_preparation_mode'] = prep_mode
    if matched_task:
        val_images, val_max, scale = _infer_validation_images(matched_task, matched_entry)
        overrides.update({
            'benchmark_task': matched_task,
            'image_scale': scale,
            'validation_images': val_images,
            'validation_max_images': int(val_max),
            'validation_reference_mode': str(((raw.get('validation') or {}).get('split_fidelity_reference_mode') or 'auto')).strip() or 'auto',
            'mini_coco_ap50': bool(matched_task == 'detection' and 'mini_coco_ap50' in set((raw.get('validation') or {}).get('detection_metrics') or [])),
            'mini_classification_eval': bool(matched_task == 'classification'),
        })
    for key in ('max_accepted_cases_per_model', 'preferred_shortlist', 'min_gap'):
        if key in selection_policy:
            try:
                mapped = {
                    'max_accepted_cases_per_model': 'requested_cases',
                    'preferred_shortlist': 'preferred_shortlist',
                    'min_gap': 'min_gap',
                }[key]
                overrides[mapped] = int(selection_policy.get(key) or 0)
            except Exception:
                pass
    if 'candidate_search_pool' in selection_policy:
        pool_val = selection_policy.get('candidate_search_pool')
        if isinstance(pool_val, int):
            overrides['candidate_search_pool'] = int(pool_val)
    preflight_policy = str(selection_policy.get('full_model_hailo_preflight_policy') or '').strip()
    if preflight_policy:
        overrides['full_model_preflight_policy'] = preflight_policy

    matched_model_id = str(matched_entry.get('id') or '').strip() or None
    matched_model_family = str(matched_entry.get('family') or '').strip() or None
    prep_summary = f", prep={prep_mode}" if prep_mode else ""
    if matched:
        summary = f"{profile_id}: matched {matched_model_id} ({matched_tier}, task={matched_task or 'auto'}{prep_summary})"
    else:
        summary = f"{profile_id}: model not in profile suite{prep_summary}"
    return EvaluationProfileResolution(
        requested=requested,
        profile_id=profile_id,
        profile_path=loaded.profile_path,
        source=loaded.source,
        profile_name=str(profile_id),
        version=_profile_version(profile_id),
        matched=matched,
        matched_tier=matched_tier,
        matched_model_id=matched_model_id,
        matched_model_family=matched_model_family,
        matched_task=matched_task,
        model_tokens=list(tokens),
        run_profile_ids=list(run_flags.get('run_profile_ids') or []),
        selection_policy=selection_policy,
        overrides=overrides,
        raw_profile=raw,
        summary=summary,
    )


def profile_brief_text(resolved: Optional[EvaluationProfileResolution]) -> str:
    if resolved is None:
        return 'No evaluation profile selected.'
    if not resolved.matched:
        return f"Profile {resolved.profile_id} selected, but the current model is not part of its suite."
    runs = ', '.join(resolved.run_profile_ids) or 'custom runs'
    req = resolved.overrides.get('requested_cases')
    pref = resolved.overrides.get('preferred_shortlist')
    detail = []
    if req:
        detail.append(f"cases={req}")
    if pref:
        detail.append(f"shortlist={pref}")
    tail = f" ({', '.join(detail)})" if detail else ''
    return f"Profile {resolved.profile_id} → {resolved.matched_model_id} [{resolved.matched_task}] · runs: {runs}{tail}"


def _comparison_from_loaded_profiles(loaded_profiles: Sequence[LoadedEvaluationProfile], *, include_reserve: bool = True) -> EvaluationProfileComparison:
    summaries: List[Dict[str, Any]] = []
    model_map: Dict[str, Dict[str, Any]] = {}
    run_map: Dict[str, Dict[str, Any]] = {}
    profile_keys: List[str] = []
    for loaded in list(loaded_profiles or []):
        profile = dict(loaded.raw_profile or {})
        pid = loaded.profile_id
        profile_keys.append(pid)
        entries = profile_model_entries(profile, include_reserve=include_reserve)
        run_profiles = [dict(x or {}) for x in list(profile.get('run_profiles') or []) if isinstance(x, Mapping)]
        summaries.append({
            'profile_id': pid,
            'profile_path': loaded.profile_path,
            'source': loaded.source,
            'purpose': str(profile.get('purpose') or '').strip(),
            'version': _profile_version(pid),
            'primary_count': sum(1 for e in entries if str(e.get('_tier')) == 'primary'),
            'reserve_count': sum(1 for e in entries if str(e.get('_tier')) == 'reserve'),
            'run_ids': [str(r.get('id') or '').strip() for r in run_profiles if str(r.get('id') or '').strip()],
        })
        for ent in entries:
            mid = str(ent.get('id') or '').strip()
            if not mid:
                continue
            row = model_map.setdefault(mid, {
                'model_id': mid,
                'family': str(ent.get('family') or '').strip(),
                'task': str(ent.get('task') or '').strip(),
            })
            row[pid] = str(ent.get('_tier') or 'primary')
        for run in run_profiles:
            rid = str(run.get('id') or '').strip()
            if not rid:
                continue
            row = run_map.setdefault(rid, {
                'run_id': rid,
                'type': str(run.get('type') or '').strip(),
                'full': str(run.get('full') or run.get('full_reference') or '').strip(),
                'stage1': str(run.get('stage1') or '').strip(),
                'stage2': str(run.get('stage2') or '').strip(),
            })
            row[pid] = True
    model_rows = [model_map[k] for k in sorted(model_map)]
    run_rows = [run_map[k] for k in sorted(run_map)]
    lines: List[str] = ['# Evaluation profile comparison', '']
    for item in summaries:
        purpose = str(item.get('purpose') or '').strip()
        base = f"- **{item['profile_id']}**: primary={item['primary_count']}, reserve={item['reserve_count']}, runs={', '.join(item['run_ids']) or '—'}"
        if purpose:
            base += f"\n  - purpose: {purpose}"
        lines.append(base)
    lines.append('')
    lines.append('## Model coverage')
    lines.append('')
    for row in model_rows:
        presence = ', '.join(f"{pid}={row.get(pid, '—')}" for pid in profile_keys)
        lines.append(f"- `{row['model_id']}` ({row.get('task') or 'task?'}, {row.get('family') or 'family?'}) · {presence}")
    lines.append('')
    lines.append('## Run coverage')
    lines.append('')
    for row in run_rows:
        presence = ', '.join(f"{pid}={'yes' if row.get(pid) else '—'}" for pid in profile_keys)
        lines.append(f"- `{row['run_id']}` ({row.get('type') or 'run'}) · {presence}")
    return EvaluationProfileComparison(
        profiles=summaries,
        model_rows=model_rows,
        run_rows=run_rows,
        summary_markdown='\n'.join(lines).strip() + '\n',
    )


def format_profile_comparison_text(*profiles_or_requests: Any, base_dir: Optional[Path] = None, include_reserve: bool = True) -> str:
    if len(profiles_or_requests) == 1 and isinstance(profiles_or_requests[0], EvaluationProfileComparison):
        return str(profiles_or_requests[0].summary_markdown or '').strip() + '\n'
    specs: List[Any] = []
    loaded_profiles: List[LoadedEvaluationProfile] = []
    raw_payloads: List[Mapping[str, Any]] = []
    for item in list(profiles_or_requests or []):
        if item is None:
            continue
        if isinstance(item, EvaluationProfileComparison):
            return str(item.summary_markdown or '').strip() + '\n'
        if isinstance(item, LoadedEvaluationProfile):
            loaded_profiles.append(item)
            continue
        if isinstance(item, Mapping):
            raw_payloads.append(item)
            continue
        if str(item or '').strip():
            specs.append(item)
    if raw_payloads and not loaded_profiles and not specs:
        tmp: List[LoadedEvaluationProfile] = []
        for idx, payload in enumerate(raw_payloads, start=1):
            prof = dict(payload or {})
            pid = str(prof.get('name') or f'profile_{idx}').strip() or f'profile_{idx}'
            tmp.append(LoadedEvaluationProfile(profile_id=pid, profile_path=f'<memory:{pid}>', source='memory', raw_profile=prof))
        return _comparison_from_loaded_profiles(tmp, include_reserve=include_reserve).summary_markdown
    if loaded_profiles and not specs:
        return _comparison_from_loaded_profiles(loaded_profiles, include_reserve=include_reserve).summary_markdown
    if specs:
        return compare_evaluation_profiles(specs, base_dir=base_dir, include_reserve=include_reserve).summary_markdown
    return 'No evaluation profiles selected.\n'


def compare_evaluation_profiles(requests: Sequence[Any], *, base_dir: Optional[Path] = None, include_reserve: bool = True) -> EvaluationProfileComparison:
    loaded_profiles: List[LoadedEvaluationProfile] = []
    for req in list(requests or []):
        if not str(req or '').strip():
            continue
        loaded = load_evaluation_profile(req, base_dir=base_dir, validate=True)
        if loaded is not None:
            loaded_profiles.append(loaded)
    return _comparison_from_loaded_profiles(loaded_profiles, include_reserve=include_reserve)
