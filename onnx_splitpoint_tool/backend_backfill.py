"""Finite backend-local selection using the existing ranked candidate pool.

This is run selection state, not an artifact cache or a new artifact identity.
Exact negative build evidence and proven native score collapse admit replacement.
"""
from __future__ import annotations

import copy
import threading
import contextvars
import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_BACKFILL = {
    'enabled': True, 'max_candidates_per_backend': 16,
    'max_cold_builds': 8, 'max_hailo_part1_builds': 4,
    'max_trt_part2_builds': 4, 'hailo_build_timeout_s': 3600,
    'trt_build_timeout_s': 7200,
    'technical_output_contract_version': 1,
}

_external_reservation = contextvars.ContextVar('backend_selection_build_reservation', default=None)


def reserve_active_build(command: list[str]) -> None:
    """Called only at the existing DeepX compiler's real process boundary."""
    reserve = _external_reservation.get()
    if reserve is not None:
        reserve(command)


def call_with_build_budget(function: Any, *, state: Any = None, persist: Any = None,
                           state_path: Any = None, stage: str, **kwargs: Any) -> Any:
    """Bind the existing generation checkpoint to deferred/local DX-COM work."""
    if state_path is not None:
        import fcntl
        path = Path(state_path)
        with path.with_name('.' + path.name + '.build-budget.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            saved = json.loads(path.read_text()) if path.is_file() else {}
            state = saved.get('backend_backfill')
            from .benchmark.generation_state import write_json_atomic
            return call_with_build_budget(function, state=state,
                persist=lambda: write_json_atomic(path, saved), stage=stage, **kwargs)
    if not state:
        return function(**kwargs)
    def reserve(command: list[str]) -> None:
        if state['cold_builds_started'] >= state['policy']['max_cold_builds']:
            raise RuntimeError('build_budget_exhausted:deepx')
        state['build_starts'].append({'backend': 'deepx', 'stage': stage, 'command': command,
                                     'started': True, 'status': 'reserved_before_dispatch'})
        state['cold_builds_started'] += 1
        persist()
    token = _external_reservation.set(reserve)
    try:
        return function(**kwargs)
    finally:
        _external_reservation.reset(token)


def backfill_policy(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = (profile.get('selection_policy') or {}).get('backend_backfill')
    if value is None:
        return {}  # Historical snapshots never acquire new selection behavior.
    if not isinstance(value, Mapping) or type(value.get('enabled', False)) is not bool:
        raise ValueError('backend_backfill.enabled must be boolean')
    if not value.get('enabled'):
        return {}
    result = {**DEFAULT_BACKFILL, **value}
    # Historical explicit profiles/resumes do not acquire a new policy.
    result['technical_output_contract_version'] = value.get('technical_output_contract_version', 0)
    if type(result['technical_output_contract_version']) is not int or result['technical_output_contract_version'] not in (0, 1):
        raise ValueError('backend_backfill.technical_output_contract_version must be 0 or 1')
    for key in DEFAULT_BACKFILL:
        if key == 'enabled':
            continue
        if type(result[key]) is not int or result[key] < (1 if 'timeout' in key or 'candidates' in key else 0):
            raise ValueError(f'backend_backfill.{key} must be a finite integer')
    if 'technical_output_contract_version' not in value:
        result.pop('technical_output_contract_version')  # Preserve saved resume bytes.
    return result


def backend_key(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get('hw_arch') or value.get('backend') or value.get('provider') or value.get('type') or ''
    value = str(value or '').lower().replace('-', '_')
    if 'hailo10' in value:
        return 'hailo10h'
    if 'hailo8' in value:
        return 'hailo8'
    if 'deepx' in value:
        return 'deepx'
    return value


def selection_contracts(runs: Any, targets: Any = ()) -> list[dict[str, Any]]:
    contracts = []
    for run in runs or []:
        if not isinstance(run, Mapping) or run.get('required') is False or run.get('deferred'):
            continue
        rid = str(run.get('id') or run.get('run_id') or '')
        if run.get('semantic_reference_only') or run.get('canonical_cpu_reference'):
            continue
        stage = run.get('stage1') or ''
        if not stage and '_to_' not in rid:
            continue
        if str(run.get('variants') or '').lower() in {'full', "['full']"}:
            continue
        backend = backend_key(stage or rid)
        setup = str(run.get('expected_setup_id') or run.get('setup_id') or '')
        endpoint = str(run.get('measurement_endpoint') or '')
        contract = {'id': '|'.join((rid, setup, endpoint)), 'run_id': rid,
                    'backend': backend, 'setup_id': setup, 'measurement_endpoint': endpoint,
                    'stage': 'part2' if backend_key(run.get('stage2')) in {'hailo8', 'hailo10h', 'deepx'} else 'part1'}
        if contract['stage'] == 'part2':
            contract['backend'] = backend_key(run.get('stage2'))
        descriptors = run.get('physical_identity_descriptors') or [{}]
        for descriptor in descriptors:
            if descriptor.get('variant') and descriptor['variant'] != 'split':
                continue
            scoped = dict(contract)
            scoped['setup_id'] = str(descriptor.get('expected_setup_id') or setup)
            scoped['measurement_endpoint'] = str(descriptor.get('measurement_endpoint') or endpoint)
            scoped['execution_contract'] = str(descriptor.get('global_run_descriptor_sha256') or '')
            scoped['id'] = '|'.join((rid, scoped['setup_id'], scoped['measurement_endpoint'], scoped['execution_contract']))
            if scoped not in contracts:
                contracts.append(scoped)
    if not contracts:
        contracts = [{'id': backend_key(t), 'run_id': '', 'backend': backend_key(t),
                      'setup_id': '', 'measurement_endpoint': '', 'stage': 'part1'} for t in targets]
    return contracts


def selected_cases_for_backend(state: Mapping[str, Any], backend: str, fallback: list[str], *, setup_id: str = '', run_id: str = '') -> list[str]:
    if not state or state.get('enabled') is not True:
        return list(fallback)
    from .workflow.required_run_scope import canonical_run_id
    result = []
    for contract in state.get('contracts', []):
        if (contract['backend'] == backend_key(backend)
                and (not setup_id or contract.get('setup_id', '') in {'', setup_id})
                and (not run_id or canonical_run_id(contract.get('run_id')) == canonical_run_id(run_id))):
            for case in contract.get('selected_case_ids', []):
                if case in fallback and case not in result:
                    result.append(case)
    return result


def bind_plan_cases(plan: dict[str, Any], state: Mapping[str, Any]) -> None:
    if not state:
        return
    plan['backend_backfill'] = copy.deepcopy(state)
    plan['backend_backfill_build_budget'] = {
        **state['policy'],
        'remaining_cold_builds': max(0, state['policy']['max_cold_builds'] - state['cold_builds_started']),
    }
    # Assign the remaining split-build starts before remote dispatch. Each
    # setup receives a fixed finite share; copies on separate hosts cannot
    # each spend the entire model budget.
    slots = {}
    remaining = min(plan['backend_backfill_build_budget']['remaining_cold_builds'],
                    state['policy']['max_trt_part2_builds'])
    setups = list(dict.fromkeys(c.get('setup_id', '') for c in state['contracts']
                               if c['stage'] == 'part1' and c['selected_case_ids']))
    # Divide the shared allowance across distinct setups. Giving the entire
    # remainder to the first setup silently makes every later setup warm-only,
    # even when the model budget covers all selected split chains.
    share, extra = divmod(remaining, len(setups)) if setups else (0, 0)
    for index, setup in enumerate(setups):
        slots[setup] = share + int(index < extra)
    plan['backend_backfill_build_budget']['trt_starts_by_setup'] = slots

    for run in plan.get('runs') or plan.get('planned_runs') or []:
        matches = [c for c in state['contracts'] if c['run_id'] == str(run.get('id') or run.get('run_id') or '')]
        if matches:
            run['case_ids'] = list(dict.fromkeys(case for c in matches for case in c['selected_case_ids']))
            run['backend_selection_contracts'] = copy.deepcopy(matches)
            for descriptor in run.get('physical_identity_descriptors') or []:
                if descriptor.get('variant') == 'split':
                    scoped = [c for c in matches if c['setup_id'] == str(descriptor.get('expected_setup_id') or '')
                              and c['measurement_endpoint'] == str(descriptor.get('measurement_endpoint') or '')]
                    descriptor['case_ids'] = list(dict.fromkeys(case for c in scoped for case in c['selected_case_ids']))
            run['selection_comparison_group'] = 'backend_optimized_selection'


class BackendBackfill:
    def __init__(self, *, state: dict[str, Any], policy: Mapping[str, Any], pool: list[int],
                 initial: list[int], quota: int, contracts: list[dict[str, Any]], persist: Any):
        self.state = state
        self.persist = persist
        self.build_lock = threading.RLock()
        frozen = {'policy': dict(policy), 'candidate_order': list(pool), 'initial_selection': list(initial), 'quota': quota, 'contract_definitions': copy.deepcopy(contracts)}
        if state:
            if any(state.get(k) != v for k, v in frozen.items()):
                raise ValueError('backend_backfill_resume_contract_changed')
        else:
            state.update(enabled=True, **frozen, cold_builds_started=0, build_starts=[], audit_cases=[],
                         contracts=[dict(c, selected_case_ids=[], considered=[], status='pending') for c in contracts])
            self.persist()

    def bind_builder(self, builder: Any) -> Any:
        from types import SimpleNamespace
        from .workflow.deferred_hailo_builds import _probe_classification, _result_negative_evidence

        def dispatch_serial(source: Any, **kwargs: Any) -> Any:
            probe = builder(source, **dict(kwargs, cache_only=True, force=False))
            status, reason = _probe_classification(probe)
            if status in {'HIT', 'KNOWN_INFEASIBLE'} or kwargs.get('cache_only'):
                return probe
            stage = str((kwargs.get('build_evidence_context') or {}).get('stage') or '')
            policy = self.state['policy']
            started = self.state['build_starts']
            part1_count = sum(r['stage'] == 'part1' for r in started)
            exhausted = (self.state['cold_builds_started'] >= policy['max_cold_builds']
                         or stage == 'part1' and part1_count >= policy['max_hailo_part1_builds'])
            if status != 'MISS' or exhausted or stage not in {'part1', 'part2'}:
                return SimpleNamespace(ok=False, skipped=True, timed_out=False, hef_path=None,
                    failure_kind='build_budget_exhausted' if exhausted else 'infrastructure_blocked',
                    error='build_budget_exhausted' if exhausted else f'cache_probe_not_dispatchable:{status}:{reason}',
                    elapsed_s=0, details={'cache_probe_status': status}, calib_info={})
            observation = {'backend': backend_key(kwargs.get('hw_arch')), 'stage': stage,
                           'boundary': (kwargs.get('build_evidence_context') or {}).get('boundary'),
                           'cache_status': status, 'started': True, 'status': 'reserved_before_dispatch'}
            started.append(observation)
            self.state['cold_builds_started'] += 1
            self.persist()  # A failure here MUST prevent dispatch.
            kwargs.update(force=False, wsl_timeout_s=min(int(kwargs.get('wsl_timeout_s') or policy['hailo_build_timeout_s']), policy['hailo_build_timeout_s']))
            result = builder(source, **kwargs)
            observation['status'] = 'completed' if getattr(result, 'ok', False) else 'failed'
            self.persist()
            if not getattr(result, 'ok', False):
                # A fresh exact lookup is required even for a newly recorded
                # deterministic failure. Unknown/timeout failures cannot backfill.
                refreshed = builder(source, **dict(kwargs, cache_only=True, force=False))
                if _result_negative_evidence(refreshed):
                    return refreshed
            return result
        def dispatch(source: Any, **kwargs: Any) -> Any:
            # Existing schedulers may invoke different Hailo targets in
            # parallel. This finite selection budget serializes its own work.
            with self.build_lock:
                return dispatch_serial(source, **kwargs)
        return dispatch

    def active(self) -> list[dict[str, Any]]:
        return [c for c in self.state['contracts'] if c['status'] == 'pending'
                and len(c['selected_case_ids']) < self.state['quota']]

    def begin(self, boundary: int, min_gap: int = 0) -> list[dict[str, Any]]:
        active = []
        for contract in self.active():
            if min_gap > 0 and any(abs(boundary - int(case[1:])) < min_gap for case in contract['selected_case_ids']):
                continue
            if boundary in contract['considered']:
                terminal = any(row['contract_id'] == contract['id'] and int(row['case_id'][1:]) == boundary
                               for row in self.state['audit_cases'])
                if terminal:
                    continue
                # Resume an interrupted candidate; interruption is never a
                # deterministic exclusion permitting a different candidate.
                active.append(contract)
                continue
            if len(contract['considered']) >= self.state['policy']['max_candidates_per_backend']:
                contract['status'] = 'pool_exhausted'
                continue
            contract['considered'].append(boundary)
            active.append(contract)
        self.persist()
        return active

    def observe(self, boundary: int, active: list[dict[str, Any]], hefs: Mapping[str, Any], *, case_id: str = '') -> list[str]:
        from .workflow.artifact_cache_preflight import known_negative_build_evidence
        selected = []
        case = case_id or f'b{boundary:03d}'
        for contract in active:
            stage = contract['stage']
            meta = next((v for k, v in hefs.items() if backend_key(k) == contract['backend']), {})
            build = meta.get(stage+'_build') or {}
            negative = known_negative_build_evidence(build) if not meta.get(stage) else {}
            suitability = meta.get(stage+'_output_suitability') or {}
            observation = {'case_id': case, 'contract_id': contract['id'], 'backend': contract['backend'],
                           'stage': stage, 'build_evidence': negative, 'cache_hit': build.get('cache_hit') is True}
            technical_policy = (self.state['policy'].get('technical_output_contract_version') == 1
                                and contract['backend'] == 'hailo10h' and stage == 'part1')
            if technical_policy:
                observation['output_suitability'] = copy.deepcopy(suitability)
            if technical_policy and meta.get(stage) and suitability.get('binding_verified') is True:
                from .native_split_quality import probability_quantization_suitability
                proof = probability_quantization_suitability(suitability.get('semantic') or {}, suitability.get('native_tensor') or {})
                if (proof.get('status') == 'INCOMPATIBLE'
                        and suitability.get('status') == 'INCOMPATIBLE'
                        and suitability.get('artifact') == meta.get(stage+'_output_artifact')):
                    observation.update(status='excluded', reason='TECHNICAL_OUTPUT_INCOMPATIBLE',
                                       artifact=meta.get(stage), build_status='BUILT')
                    self.state['audit_cases'].append(observation)
                    continue
            if negative:
                observation.update(status='excluded', reason=negative['state'])
                self.state['audit_cases'].append(observation)
                continue
            if technical_policy and suitability.get('required') and suitability.get('status') == 'UNKNOWN':
                contract['status'] = 'output_contract_unknown'
                observation.update(status=contract['status'], reason=suitability.get('reason'))
                self.state['audit_cases'].append(observation)
                continue
            elif contract['backend'] in {'hailo8', 'hailo10h'} and not meta.get(stage):
                contract['status'] = 'build_budget_exhausted' if build.get('failure_kind') == 'build_budget_exhausted' else 'infrastructure_blocked'
                observation.update(status=contract['status'], reason=build.get('error') or 'required_artifact_unavailable')
            else:
                observation.update(status='selected', reason='next_ranked_nonexcluded_candidate')
            contract['selected_case_ids'].append(case)
            if contract['status'] == 'pending' and len(contract['selected_case_ids']) >= self.state['quota']:
                contract['status'] = 'complete'
            selected.append(contract['id'])
            self.state['audit_cases'].append(observation)
        self.persist()
        return selected

    def finish(self) -> None:
        for contract in self.active():
            contract['status'] = 'pool_exhausted'
        self.state['final_backend_matrix'] = [dict(c) for c in self.state['contracts']]
        self.state['comparison_group'] = 'backend_optimized_selection'
        self.state['audit_case_count'] = sum(c['status'] == 'excluded' for c in self.state['audit_cases'])
        self.state['execution_case_count'] = sum(len(c['selected_case_ids']) for c in self.state['contracts'])
        self.persist()
