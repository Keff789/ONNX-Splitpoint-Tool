"""Hailo split stages consumed by a generated executable benchmark plan.

This is a projection of existing run directions, not another artifact identity.
None means a legacy/incomplete plan cannot safely narrow existing requests.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _token(value: Any) -> str:
    return str(value or '').strip().lower().replace('-', '_')


def hailo_scope_backend(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ('hw_arch', 'accelerator', 'backend', 'provider', 'target', 'type', 'id', 'name'):
            backend = hailo_scope_backend(value.get(key))
            if backend:
                return backend
        return ''
    token = _token(value)
    if token in {'hailo8', 'hailo8l', 'hailo8r', 'hailo_8'}:
        return 'hailo8'
    if token in {'hailo10', 'hailo10h', 'hailo10p', 'hailo_10'}:
        return 'hailo10'
    return ''


def _mentions_hailo(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_mentions_hailo(item) for item in value.values())
    return 'hailo' in _token(value)


def _cases(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        values = [values]
    result = []
    for value in values:
        if isinstance(value, Mapping):
            value = (value.get('id') or value.get('case_id') or value.get('case')
                     or value.get('case_dir') or value.get('folder')
                     or value.get('boundary') or value.get('split_index'))
        token = _token(value)
        if token.isdigit() or token.startswith('b') and token[1:].isdigit():
            token = f"b{int(token.lstrip('b')):03d}"
        if token and token not in result:
            result.append(token)
    return result


def selected_hailo_artifact_stages(
    plan: Mapping[str, Any], selected_cases: Sequence[Any],
) -> set[tuple[str, str, str]] | None:
    """Return exact (Hailo family, case, part) requirements, if resolvable.

    Nonempty generated rows define execution scope even when all are disabled.
    Empty/malformed/unknown legacy plans retain the prior conservative behavior.
    Full requests are deliberately handled by the existing Full contract.
    """
    rows = plan.get('runs') or plan.get('planned_runs')
    if not isinstance(rows, list) or not rows:
        return None
    accepted = set(_cases(selected_cases))
    required: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        if (row.get('enabled') is False or _token(row.get('enabled')) in {'false', 'off', 'no', '0'}
                or bool(row.get('deferred'))
                or _token(row.get('status') or row.get('build_status')) in {'disabled', 'deferred', 'not_selected'}):
            continue
        run_id = _token(row.get('id') or row.get('run_id'))
        run_type = _token(row.get('type'))
        variants_raw = row.get('variants') or row.get('variant') or []
        variants = {_token(value) for value in (variants_raw if isinstance(variants_raw, list) else [variants_raw])}
        fallback = run_id.split('_to_', 1) if '_to_' in run_id else []
        declared_stages = [row.get('stage1'), row.get('stage2')]
        stages = [hailo_scope_backend(value or (fallback[index] if fallback else ''))
                  for index, value in enumerate(declared_stages)]
        same_backend = hailo_scope_backend(row.get('hw_arch') or row.get('backend') or row.get('provider') or run_id)
        if not any(stages) and same_backend and run_type in {'hailo', 'same_backend_reference'}:
            stages = [same_backend, same_backend]
        for index, value in enumerate(declared_stages):
            raw_stage = value or (fallback[index] if fallback else '')
            if _mentions_hailo(raw_stage) and not stages[index]:
                return None
        if run_type == 'hailo' and not same_backend and not any(stages):
            return None
        split = bool(fallback or run_type in {'matrix', 'split', 'mixed_backend'}
                     or variants.intersection({'part1', 'part2', 'composed', 'split'})
                     or row.get('same_backend_split_diagnostics_enabled') is True)
        if not split:
            # Known Full/provider recipes contain no accelerator split work.
            known = bool(same_backend or hailo_scope_backend(row.get('full'))
                         or any(declared_stages) or row.get('provider') or row.get('backend')
                         or run_type in {'full', 'hailo', 'onnxruntime', 'deepx', 'same_backend_reference'}
                         or run_id in {'ort_cpu', 'ort_cuda', 'ort_tensorrt', 'tensorrt', 'trt', 'cpu', 'deepx_m1_full'})
            if not known:
                return None
            continue
        if not fallback and not any(declared_stages) and not any(stages):
            return None
        row_cases: list[str] = []
        explicit_cases = False
        for key in ('case_id', 'case', 'cases', 'case_ids', 'selected_cases'):
            if row.get(key) not in (None, '', [], {}):
                explicit_cases = True
                row_cases.extend(_cases(row[key]))
        cases = set(row_cases) & accepted if explicit_cases else accepted
        for index, stage in enumerate(('part1', 'part2')):
            if variants and not variants.intersection({stage, 'composed', 'split'}):
                continue
            backend = stages[index]
            if backend:
                required.update((backend, case, stage) for case in cases)
    return required
