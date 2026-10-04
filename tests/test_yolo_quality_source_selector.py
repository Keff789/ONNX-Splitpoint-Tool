"""Real generated-suite selector regression; all outputs belong to external pytest tmp.

YOLO_SELECTOR_USE_FAILED_SUITE=1 reproduces the preserved productive failure.
The default uses the normal suite writer against the same immutable run metadata.
YOLO_SELECTOR_USE_INSTALLED_SUITE=1 reads the generated suite without rewriting;
YOLO_SELECTOR_SUITE_RUN can point at an isolated normal workflow-writer run copy.
"""
from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

import pytest

OUT = (Path(os.environ.get('THESIS20_FIXTURE_HOME', Path.home())) / 'Reports/thesis20-yolo-min3-augmentation-20261002_222904')
RUN = OUT / 'runs/yolo_min3_20261002_222904'
SESSION = '56d3d5e7b2c4485c81d52b8d660170b5'
PLAN = RUN / 'jobs/supplement_attempts' / SESSION / 'supplement_plan.json'
pytestmark = pytest.mark.skipif(not PLAN.is_file(), reason='requires preserved actual YOLO augmentation run')
SETUPS = ('orin_nx_hailo8_01', 'orin_nx_hailo10_01', 'orin_nx_deepx_m1_01')


def _load_suite(root: Path, model='yolo11l'):
    source_run = Path(os.environ.get('YOLO_SELECTOR_SUITE_RUN') or RUN)
    source = source_run / 'models' / model / 'benchmark_set/legacy_suite'
    root.mkdir(parents=True)
    for name in ('benchmark_suite.py', 'benchmark_set.json', 'benchmark_plan.json', 'scientific_reporter_v60.py'):
        if (source / name).exists():
            shutil.copy2(source / name, root / name)
    if os.environ.get('YOLO_SELECTOR_USE_INSTALLED_SUITE') == '1':
        shutil.copytree(source / 'splitpoint_runners', root / 'splitpoint_runners',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if (os.environ.get('YOLO_SELECTOR_USE_FAILED_SUITE') != '1'
            and os.environ.get('YOLO_SELECTOR_USE_INSTALLED_SUITE') != '1'):
        from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
        write_benchmark_suite_script(root)
    spec = importlib.util.spec_from_file_location('selector_' + root.parent.name.replace('-', '_'), root / 'benchmark_suite.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source


def _productive_args(setup):
    metas = RUN / 'models/yolo11l/benchmark_results/attempts' / SESSION / 'remote_runs/Results/legacy_suite/1'
    matches = list(metas.glob('*_' + setup + '_completion_' + SESSION + '/run_meta.json'))
    assert len(matches) == 1
    meta = json.loads(matches[0].read_text())
    tokens = shlex.split(meta['args']['add_args'])
    rows = json.loads(tokens[tokens.index('--selective-completion-rows') + 1])
    plan = json.loads(PLAN.read_text())
    for row in rows:
        pool = plan['quality_source_rows'] if row.get('stage') == 'quality_source' else plan['generic_rows']
        assert any(all(candidate.get(key) == value for key, value in row.items()) for candidate in pool)
    return matches[0], tokens, rows


RUN_IDS = dict(zip(SETUPS, ('hailo8_to_trt', 'hailo10_to_tensorrt', 'deepx_m1_to_tensorrt')))
BACKENDS = dict(zip(SETUPS, ('hailo8_to_trt', 'hailo10h_to_trt', 'deepx_to_trt')))
MODELS = ('yolo11l', 'yolo26s', 'yolov7_paper')


def _selector_rows(model, setup):
    from onnx_splitpoint_tool.workflow.execution_binding import _scheduler_owned_remote_add_args
    plan = json.loads(PLAN.read_text())
    selected = [row for name in ('generic_rows', 'quality_source_rows') for row in plan[name]
                if row['model_id'] == model and row['setup_id'] == setup]
    argv = shlex.split(_scheduler_owned_remote_add_args('', run_ids=[RUN_IDS[setup]],
                         selective_completion_rows=selected))
    return argv, json.loads(argv[argv.index('--selective-completion-rows') + 1])


def _install_software_cache(root, source, model, setup, rows):
    """Small SDK artifacts only; normal local receipt verification and writer run.

    Original remote engine files are not on this controller. This fixture does
    not claim their identity. Real plan/scope/dataset/CPU reference stay intact.
    """
    from tests.test_yolo_augmentation_quality_release import binding
    from onnx_splitpoint_tool.split_export_runners import write_runner_skeleton_onnxruntime
    from onnx_splitpoint_tool.native_split_quality import validate_native_split_quality_binding
    cache = root.parent / 'software-cache'
    cases = [case['case_dir'] for case in json.loads((source / 'benchmark_set.json').read_text())['cases']]
    for case in cases:
        case_dir = root / case
        case_dir.mkdir()
        shutil.copy2(source / case / 'split_manifest.json', case_dir / 'split_manifest.json')
        write_runner_skeleton_onnxruntime(case_dir, target='cpu')
        bound = binding(root.parent / 'software-artifacts' / case, model, case, setup, BACKENDS[setup], RUN.name)
        verified, status = validate_native_split_quality_binding(bound, verification_mode='local')
        assert verified is not None, status
        p1 = {'hailo8_to_trt':'hailo/hailo8/part1/compiled.hef',
              'hailo10h_to_trt':'hailo/hailo10h/part1/compiled.hef',
              'deepx_to_trt':'deepx/deepx_m1/part1/model.dxnn'}[BACKENDS[setup]]
        (case_dir / p1).parent.mkdir(parents=True)
        shutil.copy2(bound['artifacts']['part1_runtime']['path'], case_dir / p1)
        shutil.copy2(bound['artifacts']['source_part2_onnx']['path'], case_dir / 'source_part2.onnx')
        target = cache / 'native_split_quality' / setup / model / case / BACKENDS[setup] / 'software-fixture'
        target.mkdir(parents=True)
        (target / 'native_split_quality_binding.json').write_text(json.dumps(bound))
    # This link is exclusively read by the normal dataset admission code. All
    # generated case/provenance/cache outputs above are regular fixture files.
    dataset = Path('resources/validation/detection/val2017_n5000_s20260710')
    (root / dataset).parent.mkdir(parents=True, exist_ok=True)
    (root / dataset).symlink_to(source / dataset, target_is_directory=True)
    return cache


def _execute_main(tmp_path, monkeypatch, model, setup, *, change=None):
    module, source = _load_suite(tmp_path / 'suite', model)
    if model == 'yolo11l' and change is None:
        meta_path, tokens, rows = _productive_args(setup)
    else:
        tokens, rows = _selector_rows(model, setup)
        meta_path = None
    if change is not None:
        rows = change(copy.deepcopy(rows))
        tokens[tokens.index('--selective-completion-rows') + 1] = json.dumps(rows)
    case_calls, child_calls, run_calls = [], [], []
    original_run_filter = module._completion_run_rows
    def traced_run_filter(*args, **kwargs):
        value = original_run_filter(*args, **kwargs)
        run_calls.append({'run_id': args[1], 'stage': kwargs.get('stage', 'generic'),
                          'selected': copy.deepcopy(value)})
        return value
    monkeypatch.setattr(module, '_completion_run_rows', traced_run_filter)
    original_filter = module._completion_case_variants
    def traced_filter(*args, **kwargs):
        value = original_filter(*args, **kwargs)
        case_calls.append({'run_id': args[1], 'case_id': args[2].get('case_dir'),
                           'include_quality_source': kwargs.get('include_quality_source', True), 'selected': value})
        return value
    monkeypatch.setattr(module, '_completion_case_variants', traced_filter)
    old = os.environ.get('YOLO_SELECTOR_USE_FAILED_SUITE') == '1'
    if not old:
        cache = _install_software_cache(tmp_path / 'suite', source, model, setup, rows)
        monkeypatch.syspath_prepend(str(tmp_path / 'suite'))
        monkeypatch.setenv('ONNX_SPLITPOINT_ARTIFACT_POLICY', 'cache_verify_only')
        tokens += ['--quality-evidence-eval-id', RUN.name,
                   '--quality-evidence-model-id', model, '--quality-evidence-setup-id', setup,
                   '--trt-cache-root', str(cache), '--no-native-trt-build']
    def child_leaf(cmd, **kwargs):
        if old or not isinstance(cmd, list) or not any(str(arg).endswith('/run_split_onnxruntime.py') for arg in cmd):
            raise AssertionError('Unexpected non-inference process: ' + str(cmd)[:200])
        assert Path(kwargs['cwd']).is_relative_to(tmp_path)
        child_calls.append({'case_id': Path(kwargs['cwd']).name, 'argv': cmd})
        # Controlled unsuccessful SDK/child boundary: normal _run_case retains
        # its TimeoutExpired path, and main continues the independent cases.
        raise module.subprocess.TimeoutExpired(cmd, 0)
    monkeypatch.setattr(module.subprocess, 'run', child_leaf)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Unexpected Popen'))
    monkeypatch.setattr(sys, 'argv', [str(tmp_path / 'suite/benchmark_suite.py'), *tokens, '--no-csv', '--no-plot'])
    capture = io.StringIO()
    error = None
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        try:
            rc = module.main()
        except (ValueError, RuntimeError) as exc:
            rc, error = None, str(exc)
    bench = json.loads((source / 'benchmark_set.json').read_text())
    run_id = RUN_IDS[setup]
    main_run_calls = copy.deepcopy(run_calls)
    direct = [original_filter(rows, run_id, case, ['composed']) for case in bench['cases']]
    evidence = {'source_suite': str(source / 'benchmark_suite.py'), 'productive_meta': str(meta_path),
                'supplement_plan': str(PLAN), 'rows_unchanged': rows, 'return_code': rc, 'error': error,
                'model':model, 'setup':setup, 'main_run_filter_calls': main_run_calls, 'main_case_filter_calls': case_calls,
                'child_calls': child_calls, 'second_filter_direct_on_exact_original_rows': direct,
                'console': capture.getvalue(), 'software_runtime_artifacts': not old}
    (tmp_path / 'selector-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (tmp_path / 'console.txt').write_text(capture.getvalue())
    return evidence


@pytest.mark.parametrize('setup', SETUPS)
def test_actual_productive_main_and_case_filter_select_quality_source(tmp_path, monkeypatch, setup):
    evidence = _execute_main(tmp_path, monkeypatch, 'yolo11l', setup)
    assert evidence['main_case_filter_calls'], evidence
    assert evidence['second_filter_direct_on_exact_original_rows'] == [['composed']]
    assert len(evidence['child_calls']) == 1, evidence
    _assert_quality_child(evidence['child_calls'][0])


def _assert_quality_child(child):
    argv = child['argv']
    assert '--quality-source-only' in argv
    assert argv[argv.index('--variants') + 1] == 'composed'
    assert argv[argv.index('--validation-max-images') + 1] == '5000'
    policy = json.loads(argv[argv.index('--quality-gate-json') + 1])
    assert policy['statistics']['bootstrap_repetitions'] == 1000
    assert '--existing-management-cpu-reference-json' in argv
    assert '--native-split-quality-binding' in argv
    emitted = json.loads(Path(argv[argv.index('--native-split-quality-binding') + 1]).read_text())
    assert emitted['cache_verify_replay']['compiler_dispatched'] is False
    assert emitted['eval_run_id'] == RUN.name
    assert emitted['preselection']['case_id'] == child['case_id']
    assert '--no-native-trt-build' in argv
    assert '--energy-measurement-only' not in argv
    assert '--quality-evidence-only' not in argv


@pytest.mark.parametrize('model', MODELS[1:])
@pytest.mark.parametrize('setup', SETUPS)
def test_other_actual_primary_sources_reach_only_quality_child(tmp_path, monkeypatch, model, setup):
    evidence = _execute_main(tmp_path, monkeypatch, model, setup)
    expected = {'b038'} if model == 'yolo26s' else {'b044', 'b066'}
    assert {child['case_id'] for child in evidence['child_calls']} == expected, evidence
    assert len(evidence['child_calls']) == len(expected)
    for child in evidence['child_calls']:
        _assert_quality_child(child)
    assert all(row['action']=='preserve' for row in evidence['rows_unchanged'] if row.get('stage','generic')=='generic')


def _negative_rows(rows, mutation):
    for row in rows:
        if row.get('stage') != 'quality_source':
            continue
        if mutation in {'preserve', 'negative_completed'}:
            row['action'] = mutation
        elif mutation == 'foreign_case':
            row['case_id'] = 'b999'
        elif mutation == 'foreign_run':
            row['run_id'] = 'foreign_backend'
        elif mutation == 'foreign_model':
            row['model_id'] = 'yolo26m'
        elif mutation == 'foreign_setup':
            row['setup_id'] = 'foreign_setup'
        elif mutation == 'full_variant':
            row['variant'] = 'full'
        else:
            raise AssertionError(mutation)
    return rows


@pytest.mark.parametrize('mutation', ('preserve', 'negative_completed', 'foreign_case',
                                      'foreign_run', 'foreign_model', 'foreign_setup', 'full_variant'))
def test_completed_and_foreign_sources_never_reach_child(tmp_path, monkeypatch, mutation):
    evidence = _execute_main(tmp_path, monkeypatch, 'yolo11l', SETUPS[0],
                             change=lambda rows: _negative_rows(rows, mutation))
    assert evidence['child_calls'] == [], evidence
    if mutation in {'foreign_model','foreign_setup','full_variant'}:
        assert evidence['error'] == 'quality_source_selection_scope_mismatch', evidence
    else:
        assert evidence['error'] is None, evidence
        assert all(not row['selected'] for row in evidence['main_case_filter_calls'])


def test_finished_quality_source_stays_finished_while_other_case_continues(tmp_path, monkeypatch):
    def completed_b044(rows):
        for row in rows:
            if row.get('stage') == 'quality_source' and row['case_id'] == 'b044':
                row['action'] = 'preserve'
        return rows
    evidence = _execute_main(tmp_path, monkeypatch, 'yolov7_paper', SETUPS[2], change=completed_b044)
    assert [child['case_id'] for child in evidence['child_calls']] == ['b066'], evidence
    _assert_quality_child(evidence['child_calls'][0])
