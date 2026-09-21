"""R9I regression table (offline; no device, GUI or compiler starts).

Axis              Positive evidence             Negative evidence
Transport         302 original receive records  inside/outside gap, duplicate, reorder, end
TRT classification real simulated SDK dispatch  missing Top-k and corrupt Top-k
Full/Split input  shared bound reference         foreign/missing/tampered binding
Normalization     16 verified / 48 pairs         missing repeat, wrong idle, false claim
Pad projection    exact bound contract           foreign source, image, hash, 0/114 conflict
Accounting        302 physical / 301 / 300       1 started failure / 4 never started
Generic runtime   explicit build success         SIGSEGV after report remains runtime fail
Hailo reuse       Full/Part1 exact cache lookup   cache miss never dispatches a compiler
Acceptance        required coverage              failed energy/crash; policy remains separate
"""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get('R9I_TASK_DIR', '/home/kmika/.local/share/onnx-splitpoint-codex/v283_R9I_20260919_140340_uirbnhi_'))
ORIGINAL = Path('/home/kmika/Models/EvaluationRuns/v283_R9H_20260918_233209_748qhzjo/eval_02/r9h_eval_20260919_064323')


def read(path):
    return json.loads(Path(path).read_text())


def checker():
    spec = importlib.util.spec_from_file_location('r9i_postcheck', TASK/'operator/postcheck.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def original():
    if not ORIGINAL.is_dir():
        pytest.skip('R9H original reports are local acceptance evidence')
    return ORIGINAL


@pytest.fixture(scope='module')
def energy(original):
    from onnx_splitpoint_tool.native_energy_reporting import collect_native_energy, build_native_energy_pairs
    rows = collect_native_energy(original)
    return rows, build_native_energy_pairs(rows)


def test_original_normalization_is_available_but_scientific_gates_stay_closed(energy):
    rows, pairs = energy
    assert (len(rows), len(pairs)) == (105, 126)
    normalized = [r for r in rows if r['energy_comparison_status'] == 'host_normalized_verified']
    assert len(normalized) == 16
    verified = [p for p in pairs if p['baseline_energy_comparison_status'] == 'host_normalized_verified']
    assert len(verified) == 48
    for pair in verified:
        assert 'baseline_required_host_normalization_unavailable' not in pair['comparison_reasons']
        assert not pair['claim_comparable']
        assert pair['energy_ratio'] is None
    assert not any(r['claim_eligible'] for r in rows)
    assert not any(p['comparable'] for p in pairs)


def test_original_pad_projection_removes_only_false_missing_reasons(energy):
    rows, pairs = energy
    full = [r for r in rows if r['backend'] == 'native_full_deepx' and r['task'] == 'detection']
    assert len(full) == 4
    assert {r['prepared_feed_letterbox_pad_value'] for r in full} == {'114'}
    affected = [p for p in pairs if p['baseline_backend'] == 'native_full_deepx' and p['task'] == 'detection']
    assert len(affected) == 12
    assert all('prepared_feed_pad_value_mismatch_or_missing' not in p['comparison_reasons'] for p in affected)


def test_real_zero_114_pad_conflict_remains_incomparable(energy):
    from onnx_splitpoint_tool.native_energy_reporting import build_native_energy_pairs
    rows=deepcopy(energy[0])
    full=next(r for r in rows if r['backend']=='native_full_deepx' and r['task']=='detection')
    full['prepared_feed_letterbox_pad_value']='0'
    pairs=[p for p in build_native_energy_pairs(rows) if p['model']==full['model'] and p['baseline_backend']=='native_full_deepx']
    assert len(pairs)==3
    assert all('prepared_feed_pad_value_mismatch_or_missing' in p['comparison_reasons'] and not p['comparable'] for p in pairs)


@pytest.mark.parametrize('damage', [None, 'source', 'image', 'model', 'hash', 'missing', 'ambiguous', 'conflict', 'zero'])
def test_pad_requires_exact_existing_contract(original, damage):
    from onnx_splitpoint_tool.native_energy_reporting import _project_bound_preprocessing_pad
    rows = read(original/'reports/native_energy_measurements/native_producer_energy_results.json')['rows']
    plan = deepcopy(next(i['row'] for i in rows if i['row']['backend'] == 'native_full_deepx' and i['row']['task'] == 'detection'))
    validations = read(original/'reports/native_validation/native_producer_validation_summary.json')['rows']
    validation = deepcopy(next(r for r in validations if all(r.get(k) == plan.get(k) for k in ('model', 'backend', 'case', 'setup_id', 'comparison_backend'))))
    if damage in {'source', 'image', 'model', 'hash'}:
        key = {'source':'source_request_sha256', 'image':'input_image_sha256', 'model':'model_sha256', 'hash':'preprocessing_contract_sha256'}[damage]
        validation[key] = 'f'*64
    if damage == 'missing': validation.pop('preprocessing_contract')
    if damage == 'conflict': plan['prepared_feed_letterbox_pad_value'] = '0'
    if damage == 'zero':
        from onnx_splitpoint_tool.preprocessing_contract import preprocessing_contract_sha256
        validation['preprocessing_contract'].update(letterbox_pad_value=0, pad_value=0)
        plan['preprocessing_contract_sha256'] = validation['preprocessing_contract_sha256'] = preprocessing_contract_sha256(validation['preprocessing_contract'])
    _project_bound_preprocessing_pad(plan, validation, 'ambiguous' if damage == 'ambiguous' else 'exact_unique')
    assert plan['prepared_feed_letterbox_pad_value'] == ('114' if damage is None else '0' if damage in {'zero','conflict'} else '')


def test_plan_projects_letterbox_alias_without_changing_pixels(original):
    from scripts.native_producer_energy_plan import _energy_preprocess_identity
    rows = read(original/'reports/native_energy_measurements/native_producer_energy_results.json')['rows']
    for item in rows:
        if item['row']['backend'] == 'native_full_deepx' and item['row']['task'] == 'detection':
            contract = read(item['row']['command_contract_file'])
            before = deepcopy(contract)
            assert _energy_preprocess_identity(contract)['prepared_feed_letterbox_pad_value'] == '114'
            assert contract == before


def test_original_physical_and_logical_counts_are_separate(original):
    from scripts.run_native_producer_energy_from_summary import _measurement_accounting
    result = read(original/'reports/native_energy_measurements/native_producer_energy_results.json')
    counts = _measurement_accounting(result['rows'])
    assert counts['physical_collector_attempt_count'] == 302
    assert counts['failed_physical_collector_attempt_count'] == 2
    assert counts['selected_logical_repeat_count'] == 301
    assert counts['valid_logical_repeat_count'] == 300
    assert counts['started_failed_measurement_row_count'] == 1
    assert counts['not_started_measurement_row_count'] == 4


def test_all_original_receive_evidence_preserves_inside_and_outside_gaps(original):
    from onnx_splitpoint_tool.energy.task_budget import source_completion
    budget = read(original/'energy_task_budget.json')
    chains = [c for t in budget['tasks'].values() for c in t['chains'] if c.get('collector_started')]
    assert len(chains) == 302
    gaps = []
    for chain in chains:
        folder = Path(chain['run_directory'])
        records = [json.loads(line) for line in (folder/'collector_storage/receive_diagnostics.jsonl').read_text().splitlines()]
        packets = [r['details']['counters'] for r in records if r['event'] == 'datagram' and r['details'].get('counters')]
        missing = sum(((b['first']-a['last']) % 65536)-1 for a,b in zip(packets,packets[1:]))
        assert all((p['last']-p['first']) % 65536 == 63 for p in packets)
        assert missing == sum(r['details']['missing_samples'] for r in records if r['event'] == 'counter_gap')
        assert source_completion(folder)['verified'] is True
        assert any(r['event'] == 'writer_close_end' and r['details']['ok'] for r in records)
        if missing:
            marker = read(folder/'collector_storage/command_window_markers.json')
            gaps.append((missing, marker['stream']['dropped_samples_in_window'], chain['valid']))
    assert Counter(gaps) == Counter({(64,64,False):2, (64,0,True):1})


@pytest.mark.parametrize('end', [True, False])
def test_source_completion_requires_actual_end(tmp_path, end):
    from onnx_splitpoint_tool.energy.task_budget import source_completion
    from test_v283_campaign_energy_budget import LIFECYCLE
    text = LIFECYCLE if end else '\n'.join(l for l in LIFECYCLE.splitlines() if 'protocol end verified' not in l)
    (tmp_path/'collector_stdout.log').write_text(text)
    assert source_completion(tmp_path)['verified'] is end


@pytest.mark.parametrize('name', ['natural_u16_wrap_is_not_a_drop', 'forward_gap_counts_missing_samples',
    'duplicate_and_reordered_counters_keep_u64_identity_monotonic', 'row_identity_is_always_zero_based_and_contiguous'])
def test_existing_r6_counter_rust_unit_tests(name, tmp_path):
    binary=Path('/home/kmika/.local/share/onnx-splitpoint-codex/v283_tagreparatur_R6_20260915/cargo_target/debug/deps/urecs_data_collector-618bbb2bf958e338')
    if not binary.is_file(): pytest.skip('local R6 Rust unit-test binary not installed')
    result=subprocess.run([str(binary),'--exact','network_firmware_fast::tests::'+name,'--nocapture'],capture_output=True,text=True,timeout=30)
    (tmp_path/'rust.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0 and '1 passed; 0 failed' in result.stdout


@pytest.mark.parametrize('drop', [0,64])
def test_packet_gap_window_gate_stays_strict(tmp_path, drop):
    from test_energy_command_window_binding_v2 import _binding_fixture
    from onnx_splitpoint_tool.energy.collector import _command_window_binding
    result, timing, path = _binding_fixture(tmp_path, dropped_samples=drop)
    binding = _command_window_binding(tmp_path, result, timing, result_path=path)
    assert binding['verified'] is (drop == 0)


@pytest.mark.parametrize('limit', [1,2,0,3,True])
def test_remaining_budget_reaches_snapshot_and_normal_command(tmp_path, monkeypatch, limit):
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile, save_evaluation_profile_yaml
    from onnx_splitpoint_tool.energy.task_budget import campaign_budget_profile_args, campaign_budget_forward_args
    from types import SimpleNamespace
    p = load_evaluation_profile(TASK/'profiles/A.yaml').raw_profile
    p['native_producers']['energy']['task_budget']['max_transport_failures'] = limit
    if type(limit) is not int or limit not in (1,2):
        with pytest.raises(ValueError): campaign_budget_profile_args(p['native_producers'], tmp_path)
        return
    monkeypatch.setenv('R9I_CELL','A'); monkeypatch.setenv('R9I_REMAINING_SOURCE_FAILURES',str(limit))
    path = save_evaluation_profile_yaml(tmp_path/'A.yaml', p)
    loaded = load_evaluation_profile(path)
    checker().validate_scope(loaded.raw_profile, 'smoke')
    snapshot = loaded.start_snapshot['resolved_profile']
    assert snapshot['native_producers']['energy']['task_budget']['max_transport_failures'] == limit
    args = campaign_budget_profile_args(snapshot['native_producers'], tmp_path)
    assert args[-1] == str(limit)
    forwarded = campaign_budget_forward_args(SimpleNamespace(campaign_budget_file=str(tmp_path/'budget.json'), campaign_max_retries=1, campaign_max_transport_failures=limit))
    assert forwarded[-1] == str(limit)


def test_existing_checkpoint_cannot_reset_or_raise_budget(tmp_path):
    from onnx_splitpoint_tool.energy.task_budget import EnergyTaskBudget
    path = tmp_path/'budget.json'
    limits = dict(max_chains=6,max_retries=1,max_transport_failures=1)
    with path.open('w+') as f:
        budget = EnergyTaskBudget(f, limits, campaign_row='one', source_id='source')
        budget.save(); before = path.read_bytes(); f.seek(0)
        with pytest.raises(ValueError, match='differs from existing checkpoint'):
            EnergyTaskBudget(f, {**limits,'max_transport_failures':2}, campaign_row='two', source_id='source')
    assert path.read_bytes() == before


def test_remaining_one_failure_stops_without_retry_or_next_row(tmp_path,monkeypatch):
    from test_v283_campaign_energy_budget import campaign_rig,measure
    rig=campaign_rig(tmp_path,monkeypatch,['drop','success'])
    first=measure(rig,'first',row='first',campaign_max_transport_failures=1)
    assert first['error'] == 'campaign_source_transport_failure_limit'
    second=measure(rig,'second',row='second',campaign_max_transport_failures=1)
    assert second['execution_status'] == 'NOT_RUN'
    assert rig.counts() == {'preflight':1,'collector':1,'workload':1}


def test_original_sigsegv_is_runtime_failure_even_after_report(original):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
    source = original/'models/yolo11l/benchmark_results/benchmark_results_hailo8_auto.json'
    raw = read(source)[0]
    assert raw['runner_returncode'] == -11 and raw['build_pass'] is True
    rows, _ = normalize_benchmark_files(model_id='renamed_detector',source_paths=[source])
    row = next(r for r in rows if r['variant'] == 'full')
    assert row['buildable'] is True
    assert row['runtime_ok'] is False and row['execution_ok'] is False
    assert row['validation_verdict'] == 'runtime_not_executable'
    assert row['error_class'] == 'case_runner_signal_sigsegv'
    assert row['runner_returncode'] == -11


def test_full_companion_does_not_inherit_split_structure(original):
    from onnx_splitpoint_tool.workflow.results import expand_normalized_benchmark_rows
    source=original/'models/yolo11l/benchmark_results/benchmark_results_ort_tensorrt_auto.json'
    rows=expand_normalized_benchmark_rows(read(source)[0],model_id='yolo11l',source_path=source,tag='ort_tensorrt_auto')
    full=next(r for r in rows if r['variant']=='full')
    assert full['structural_contract_pass'] is True
    assert full['numerical_similarity_pass'] is False
    assert full['ranking_eligible'] is False


def test_generic_original_groups_and_exact_runner_projection(original):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
    old=[];new=[]
    for path in sorted((original/'models').glob('*/benchmark_results/normalized_results.json')):
        old.extend(read(path)['results'])
        rows,_=normalize_benchmark_files(model_id=path.parents[1].name,source_paths=sorted(path.parent.glob('benchmark_results_*_auto.json')))
        new.extend(rows)
    assert Counter(r['validation_verdict'] for r in old) == {
        'contract_fail_or_unavailable':57,'numerical_similarity_failed':42,'screening_only':12,'not_buildable':1}
    assert len(new) == len(old) == 112
    assert Counter(r['structural_contract_reason'] for r in new if r['validation_verdict']=='contract_fail_or_unavailable') == {
        'endpoint_contract_incomplete':7,'raw_head_host_tail_missing':7}
    # The same runner's exact request now precedes the structural gate.
    projected=[r for r in new if r['structural_contract_reason']=='endpoint_contract_complete']
    assert len(projected) == 39
    assert all(r['quality_identity_valid'] and r['endpoint_contract_complete'] for r in projected)
    assert sum(r['validation_verdict']=='numerical_similarity_failed' for r in new) == 70
    assert not any(r['ranking_eligible'] for r in new)


@pytest.mark.parametrize('rc', [-11,-9,0])
def test_negative_process_returncode_overrides_stale_runtime_success(rc):
    from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
    row = dict(compile_ok=True,runtime_ok=True,runner_returncode=rc,task='classification',contract_consistent=True)
    apply_accuracy_gate_to_row(row)
    assert row['buildable'] is True
    assert row['runtime_executable'] is (rc == 0)


@pytest.mark.parametrize('structure', [True,False,None])
def test_same_backend_numeric_failure_does_not_define_structure(structure):
    from onnx_splitpoint_tool.workflow.results import _apply_validation_level_fields_v59j
    row = dict(variant='split',backend='tensorrt',runtime_ok=True,semantic_e2e_pass=False,
        interface_stage1_structural_pass=structure,interface_stage1_mapping_pass=structure,interface_stage1_shape_pass=structure)
    _apply_validation_level_fields_v59j(row)
    assert row['interface_contract_pass'] is structure
    assert row['numerical_similarity_pass'] is False
    assert row['ranking_eligible'] is False


def test_postcheck_original_rejects_real_technical_failures(original):
    result = checker().check_run(original, check_scope=False)
    assert result['pass'] is False
    assert {'generic_runtime_failure','energy_source_transport_limit','energy_not_started_source_blocked','energy_replicate_technical_failure'} <= set(result['defect_ids'])
    assert result['details']['unknown_ids'] == []
    checks = result['details']['checks']
    assert checks[0]['id'] == 'energy_transport_capture_failure'
    summary = next(c for c in checks if c['id'] == 'execution_summary')
    assert (summary['energy_failed_started_rows'],summary['energy_not_started_rows']) == (1,4)


@pytest.mark.parametrize('never_started', [True,False])
def test_postcheck_never_started_is_not_damaged_aggregate(never_started):
    execution = dict(energy_aggregate_import_status='rejected_fail_closed',execution_status='NOT_RUN' if never_started else 'INCOMPLETE',energy_collector_attempt_count=0 if never_started else 1)
    aggregate = dict(execution_status='BLOCKED',terminal=True,task_budget={'stop_reason':'campaign_source_transport_failure_limit'},runs=[])
    failed, checks = checker().energy_attempt_check(execution,aggregate,3)
    assert failed
    assert checks[0][1] == 'defect'
    assert checks[0][0] == ('energy_not_started_source_blocked' if never_started else 'energy_attempt_technical_failure')
    if never_started: assert len(checks) == 1


def test_policy_negative_complete_energy_is_not_technical_failure():
    aggregate = dict(execution_status='COMPLETED',runs=[dict(run_index=i) for i in range(3)],claim_eligible=False,screening_only=True)
    assert checker().energy_attempt_check({'energy_aggregate_import_status':'verified'},aggregate,3) == (False,[])


def test_postcheck_isolated_cli_contract(tmp_path):
    run = tmp_path/'empty';run.mkdir();out=tmp_path/'check'
    process = subprocess.run([sys.executable,'-I','-B',str(TASK/'operator/postcheck.py'),'--run',str(run),'--out',str(out)],capture_output=True,text=True,timeout=30)
    assert process.returncode == 1, process.stderr
    result = read(out/'POSTCHECK.json')
    assert result['pass'] is False and isinstance(result['defect_ids'],list) and isinstance(result['details'],dict)
    assert 'checker_input_unreadable' not in result['details']['unknown_ids']


@pytest.mark.parametrize('cell', ['A','B','C'])
def test_profiles_and_immutable_guards(cell, monkeypatch):
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
    monkeypatch.setenv('R9I_CELL',cell);monkeypatch.setenv('R9I_REMAINING_SOURCE_FAILURES','2')
    profile = load_evaluation_profile(TASK/'profiles'/f'{cell}.yaml')
    checker().validate_scope(profile.start_snapshot['resolved_profile'],'smoke')
    p=profile.raw_profile
    models=[m for m in p['model_suite']['primary'] if m.get('enabled',True)]
    assert len(models)*p['selection_policy']['backend_backfill']['max_trt_part2_builds'] <= 8
    assert p['hailo_build']['mode'] == p['deepx_build']['mode'] == 'reuse_only'
    assert p['native_producers']['build_missing_engines'] is True
    for name in ('scope_contract.py','strict_postcheck.py'):
        assert (TASK/'operator'/name).read_bytes() == (TASK/'before/task/operator'/name).read_bytes()


def test_text_collector_isolated_import_and_symlink_exclusion(tmp_path):
    from scripts.collect_cancelled_diagnostic import _priority_members
    from onnx_splitpoint_tool.workflow.zip_utils import iter_safe_pack_files
    run=tmp_path/'run';run.mkdir();(run/'run_manifest.json').write_text('{}')
    outside=tmp_path/'outside';outside.mkdir();(outside/'foreign.json').write_text('{}')
    (run/'alias.json').symlink_to(run/'run_manifest.json')
    (run/'outside').symlink_to(outside,target_is_directory=True)
    assert iter_safe_pack_files(run,run) == [run/'run_manifest.json']
    assert _priority_members(run) == ['run_manifest.json']
    process=subprocess.run([sys.executable,'-I','-B',str(REPO/'scripts/collect_cancelled_diagnostic.py'),'--help'],capture_output=True,text=True,timeout=30)
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize('missing', [False,True])
def test_host_text_collector_isolated_sibling_import(tmp_path,missing):
    host=Path('/home/kmika/Downloads/Codex_v283_R9I_Gezielte_Abnahme/run_targeted.py')
    if not host.is_file():pytest.skip('R9I HOST package is external local evidence')
    run=tmp_path/'run';run.mkdir()
    required=['run_manifest.json','profile.yaml','reports/run_status_summary.json','reports/native_producer_combined_summary.json']
    for rel in required[1:] if missing else required:
        path=run/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    outside=tmp_path/'outside';outside.mkdir();(outside/'private.json').write_text('must not be collected')
    (run/'external').symlink_to(outside,target_is_directory=True)
    (run/'alias.json').symlink_to(run/'profile.yaml')
    code="import runpy,sys,json;from pathlib import Path;h=runpy.run_path(sys.argv[1]);c=h['sibling']('collect_text_debug');print(json.dumps(c.collect(Path(sys.argv[2]),Path(sys.argv[3]))))"
    process=subprocess.run([sys.executable,'-I','-B','-c',code,str(host),str(run),str(tmp_path/'text.zip')],capture_output=True,text=True,timeout=30)
    assert process.returncode == 0,process.stderr
    result=json.loads(process.stdout)
    assert len(result['symlink_exclusions']) == 2
    assert result['read_issues'] == []
    assert result['missing_required'] == (['run/run_manifest.json'] if missing else [])
    assert result['status'] == ('INCOMPLETE_TEXT_CAPTURE' if missing else 'TEXT_CAPTURE_WITH_SYMLINK_EXCLUSIONS')
    import zipfile
    with zipfile.ZipFile(tmp_path/'text.zip') as archive:
        assert not any('private' in name or 'alias' in name or 'external' in name for name in archive.namelist())


# These existing tests exercise the actual task paths with a simulated SDK,
# without invoking accelerator runtimes or replacing their scientific gates.
from test_v283_r9h_energy_reporting import (
    test_real_trt_performance_contract_preflight_and_duration_energy_dispatch as test_trt_classification_dispatch,
    test_normalization_missing_or_conflicting_replicates_stay_invalid as test_normalization_negative_bindings,
    test_model_reference_is_shared_before_full_and_three_split_fanout as test_full_split_common_image,
    test_fast_detection_projection_requires_each_fresh_bound_energy_repeat as test_detection_exact_repeat_source,
)


@pytest.mark.parametrize('mode,run_mode', [('reuse_only','standard'), ('reuse_only','smoke'),
    ('reuse_and_build_missing','standard'), ('disabled','standard'), ('skip','standard')])
def test_hailo_profile_generation_preserves_cache_lookup(tmp_path, monkeypatch, mode, run_mode):
    from types import SimpleNamespace
    from tests import test_v2802_cpu_reference_binding as generator
    from onnx_splitpoint_tool.gui import benchmark_workflow
    from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationOrchestrationService
    from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import materialize_legacy_benchmark_set

    captured = {}
    class ConfigReady(BaseException): pass
    def builder(*args, **kwargs):
        pytest.fail('configuration test must not dispatch a builder')
    def resolve(**kwargs):
        captured['helper_request'] = kwargs
        return benchmark_workflow.ResolvedHailoBenchmarkHelpers(
            hailo_build_hef_fn=builder if kwargs['need_build'] else None)
    def stop(self, cfg):
        captured['config'] = cfg
        raise ConfigReady
    def invoke(**kwargs):
        p = kwargs['profile_payload']
        p['execution_preset'] = {'id':run_mode}
        p['hailo_build'].update(mode=mode, full_baseline_cold_build_policy='cache_or_defer')
        p['native_producers'] = {'enabled':True, 'split_backends':['hailo8'],
            'full_baselines':{'enabled':True, 'backends':['hailo8','tensorrt']}}
        p['run_profiles'] = [{'id':'hailo8','full':'hailo8','enabled':True},
            {'id':'hailo8_to_trt','stage1':'hailo8','stage2':'tensorrt','enabled':True}]
        kwargs['targets'] = ['hailo8','tensorrt']
        return materialize_legacy_benchmark_set(**kwargs)
    monkeypatch.setattr(generator, 'materialize_legacy_benchmark_set', invoke)
    monkeypatch.setattr(benchmark_workflow, 'resolve_hailo_benchmark_helpers', resolve)
    monkeypatch.setattr(BenchmarkGenerationOrchestrationService, 'run', stop)
    with pytest.raises(ConfigReady):
        generator.make_generated_reference_suite(tmp_path/'generated', 'classification')
    cfg = captured['config']
    enabled = mode not in {'disabled','skip'}
    assert captured['helper_request']['need_build'] is enabled
    assert (cfg.hailo_build_hef_fn is not None) is enabled
    if mode == 'reuse_only': assert cfg.full_model_preflight_policy == 'skip'
    for policy in (cfg, cfg.execution_cfg):
        assert policy.hailo_cache_only is (mode == 'reuse_only')
        if mode == 'reuse_only': assert policy.hailo_full_cache_only is True


@pytest.mark.parametrize('raw_hit', [True, False])
def test_full_cache_miss_probes_existing_raw_endpoint_without_cold_build(tmp_path, monkeypatch, raw_hit):
    from types import SimpleNamespace
    from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationOrchestrationService
    model=tmp_path/'renamed_detector.onnx';model.write_bytes(b'synthetic-source')
    hef=tmp_path/'raw.hef';hef.write_bytes(b'existing-raw-artifact')
    calls=[]
    def builder(source, **kwargs):
        assert kwargs['cache_only'] is True and kwargs['force'] is False
        assert str(source) == str(model)
        nodes=kwargs.get('end_node_names');calls.append(nodes)
        hit=bool(nodes and raw_hit)
        return SimpleNamespace(ok=hit, skipped=True, timed_out=False, elapsed_s=0,
            hef_path=str(hef) if hit else '', fixed_onnx_path=None,
            failure_kind='' if hit else 'deferred_cold_full_cache_miss',
            unsupported_reason='' if hit else 'cache_only_policy',
            error='' if hit else 'exact cache miss', details={}, calib_info={'cache_hit':hit})
    cfg=SimpleNamespace(hef_targets=['hailo8'], hef_full=True, hef_part1=False, hef_part2=False,
        hailo_build_hef_fn=builder, hailo_build_unavailable=None, should_cancel=None,
        hailo_full_end_node_names=[], hailo_full_endpoint_mode='', hailo_full_output_contract=None,
        out_dir=tmp_path/'suite', full_model_src=str(model), full_model_dst=str(model),
        base='renamed_detector', hailo_full_timeout_explicit=False, hailo_full_timeout_s=0,
        hef_timeout_s=60, hef_backend='auto', hef_fixup=True, hef_opt_level=1,
        hef_calib_dir=None, hef_calib_count=500, hef_calib_bs=8, hef_force=False,
        hef_keep=False, hef_wsl_distro=None, hef_wsl_venv='auto',
        hailo_cache_only=True, hailo_full_cache_only=True, hailo_run_mode='standard',
        execution_cfg=SimpleNamespace(benchmark_task='detection', build_scheduler_config={}, hailo_run_mode='standard'),
        analysis_payload={}, analysis_params_payload={}, bench_log_path=str(tmp_path/'generation.log'))
    service=BenchmarkGenerationOrchestrationService()
    monkeypatch.setattr(service, '_infer_suite_raw_head_end_nodes', lambda cfg:['head/boxes','head/scores'])
    result={}
    service._build_suite_full_hefs(cfg, log=lambda *a,**k:None, queue_put=lambda *a:None,
        errors=[], suite_hailo_hefs=result, publish_hailo_diagnostics=lambda *a:None)
    assert calls == [None, ['head/boxes','head/scores']]
    assert bool(result['hailo8'].get('full')) is raw_hit
    assert result['hailo8']['full_build']['ok'] is raw_hit
    assert result['hailo8']['full_endpoint_mode'] == 'raw_detection_head'
    if not raw_hit: assert result['hailo8'].get('full_error')


from tests.test_v27922_negative_backend import harness as hailo_cache_fixture


@pytest.mark.parametrize('stage', ['full','part1'])
@pytest.mark.parametrize('hit', [True,False])
def test_hailo_real_cache_receipts_without_compiler_dispatch(hailo_cache_fixture, monkeypatch, stage, hit):
    from onnx_splitpoint_tool import hailo_backend as backend
    from tests.test_v27933_runtime_reuse import receipt_bytes
    h=hailo_cache_fixture
    h.behavior['error']=''
    h.context['stage']=stage
    h.kwargs['net_name']='renamed_'+stage
    # Only the vendor compiler is simulated; receipt/cache publication is real.
    seed=backend.hailo_build_hef_auto(h.source, backend='local', outdir=h.root/'seed',
        build_evidence_context=h.context, **h.kwargs)
    assert seed.ok and h.calls.count('compile') == 1
    before=receipt_bytes(h.root/'cache'); calls=list(h.calls)
    def forbidden(*args,**kwargs): pytest.fail('cache-only lookup reached a compiler dispatcher')
    for name in ('hailo_build_hef_via_venv','hailo_build_hef_via_wsl','hailo_sdk_available','auto_prefers_subprocess'):
        monkeypatch.setattr(backend,name,forbidden)
    if not hit:
        h.source.write_bytes(b'different-source-no-compatible-cache')
        h.context['full_source_onnx_sha256']=hashlib.sha256(h.source.read_bytes()).hexdigest()
    result=backend.hailo_build_hef_auto(h.source, backend='auto', outdir=h.root/'lookup',
        cache_only=True, build_evidence_context=h.context, **h.kwargs)
    assert result.ok is hit
    assert h.calls == calls
    assert receipt_bytes(h.root/'cache') == before
    if hit:
        assert result.skipped and result.calib_info['cache_hit'] is True
        assert Path(result.hef_path).read_bytes() == Path(seed.hef_path).read_bytes()
    else:
        assert result.failure_kind == 'deferred_cold_full_cache_miss'
        assert result.calib_info['compiler_dispatch_count'] == 0


@pytest.mark.parametrize('cell,passed', [('A',True),('B',False)])
def test_host_cell_evidence_keeps_original_technical_result(cell, passed, monkeypatch):
    # Pin the first attempt; a later successful HOST retry must not rewrite
    # the historical regression or become its new expected failure.
    root=Path('/home/kmika/Models/EvaluationRuns')/TASK.name/f'cell_{cell}_01'
    runs=sorted(root.glob('r9i_*'))
    if not runs: pytest.skip('HOST cell evidence not present')
    monkeypatch.setenv('R9I_CELL',cell)
    monkeypatch.setenv('R9I_REMAINING_SOURCE_FAILURES','2')
    result=checker().check_run(Path(runs[0]))
    assert result['pass'] is passed
    if cell == 'B':
        run=Path(runs[0]);b=run/'models/yolo11l/benchmark_results'
        assert read(b/'benchmark_results_hailo8_auto.json') == []
        assert {'native_execution_failed','native_validation_technical_failure','generic_runtime_failure'} <= set(result['defect_ids'])
        native=read(run/'native_producers/hailo8/analysis_tables/native_full_baseline_eval.json')
        full=next(r for r in native['rows'] if r['backend']=='native_full_hailo8')
        assert full['primary_repetition_failure_reason']=='missing_hailo_full_hef'
        split=read(b/'benchmark_results_hailo8_to_trt_auto.json')[0]
        assert split['run_cfg']['variants']=='part2' and split['_runner_rc']==0
