"""Current release gates require actual new-suite execution and honest scope."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = ('test_v2801_cpu_reference_dispatch', 'test_v2801_native_remote_closure', 'test_v2801_reference_errors', 'test_v2801_reference_debug_export', 'test_v2801_workflow_integration', 'test_v2801_release_scope', 'test_v2801_release_closure')


def _write_junit(path, modules, outcome='pass'):
    suites = ET.Element('testsuites')
    suite = ET.SubElement(suites, 'testsuite', tests=str(len(modules)),
                          failures=str(int(outcome == 'failure')),
                          errors=str(int(outcome == 'error')),
                          skipped=str(int(outcome in ('skip', 'xfail'))))
    for index, module in enumerate(modules):
        row = ET.SubElement(suite, 'testcase', classname=f'tests.{module}', name='boundary')
        if index == 0 and outcome != 'pass':
            tag = 'skipped' if outcome in ('skip', 'xfail') else outcome
            ET.SubElement(row, tag, type='pytest.xfail' if outcome == 'xfail' else outcome)
    ET.ElementTree(suites).write(path, encoding='unicode')


def _report_python():
    gate = (ROOT/'scripts/run_v2801_small_acceptance.sh').read_text()
    body = gate.split('write_report() {', 1)[1].split('\nPY\n}', 1)[0]
    return body.split('"$PYTHON" -B - <<\'PY\'\n', 1)[1]


@pytest.mark.parametrize('case', ('pass', 'missing_module', 'gui_only_module', 'failure', 'error', 'skip', 'xfail'))
def test_standard_gate_requires_every_new_module_in_main_junit(tmp_path, case):
    main, gui = tmp_path/'main.xml', tmp_path/'gui.xml'
    modules = REQUIRED_MODULES[:-1] if case in ('missing_module', 'gui_only_module') else REQUIRED_MODULES
    _write_junit(main, modules, case if case in ('failure', 'error', 'skip', 'xfail') else 'pass')
    _write_junit(gui, REQUIRED_MODULES if case == 'gui_only_module' else ('test_gui',))
    report = tmp_path/'report.json'
    env = {**os.environ, 'REPORT': str(report), 'STARTED': 'fixture', 'FINAL_RC': '0',
           'PYTEST_JUNIT_MAIN': str(main), 'PYTEST_JUNIT_GUI': str(gui),
           **{key: 'PASS' for key in ('SMOKE', 'PYTEST', 'MANIFEST', 'COMPILE', 'SHELL', 'DEPENDENCIES')}}
    result = subprocess.run([sys.executable, '-I', '-B', '-c', _report_python()],
                            env=env, text=True, capture_output=True, timeout=10)
    payload = json.loads(report.read_text())
    assert result.returncode == (0 if case == 'pass' else 70), result.stdout + result.stderr
    assert payload['status'] == ('PASS' if case == 'pass' else 'FAIL')
    assert payload['required_new_modules'] == list(REQUIRED_MODULES)
    if case in ('missing_module', 'gui_only_module'):
        assert payload['missing_required_new_modules'] == [REQUIRED_MODULES[-1]]
    assert payload['real_evalrun_status'] == 'NOT_RUN'


@pytest.mark.parametrize('case', ('pass', 'missing_module', 'skip', 'xfail', 'source_drift', 'profile_drift'))
def test_short_gate_validates_junit_and_preservation(tmp_path, case):
    script = (ROOT/'scripts/run_v2801_short_tests.sh').read_text()
    code = script.rsplit('"$TEST_PY" -B - "$TEST_OUT" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
    _write_junit(tmp_path/'short_tests.xml', REQUIRED_MODULES[:-1] if case == 'missing_module' else REQUIRED_MODULES,
                 case if case in ('skip', 'xfail') else 'pass')
    before = {'ok': True, 'user_profiles': [{'path': 'profiles/custom.yaml', 'sha256': 'original'}]}
    after = json.loads(json.dumps(before))
    if case == 'source_drift':
        after['ok'] = False
    if case == 'profile_drift':
        after['user_profiles'][0]['sha256'] = 'changed'
    (tmp_path/'source_before.json').write_text(json.dumps(before))
    (tmp_path/'source_after.json').write_text(json.dumps(after))
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(tmp_path)],
                            text=True, capture_output=True, timeout=10)
    assert (result.returncode == 0) is (case == 'pass'), result.stdout + result.stderr
    if case == 'pass':
        assert 'HARDWARE_EXECUTION=NOT_RUN' in result.stdout


def test_current_gates_select_all_new_modules_and_preserve_prior_behavior():
    for current, previous in (
        ('run_v2801_short_tests.sh', 'run_v280_short_tests.sh'),
        ('run_v2801_small_acceptance.sh', 'run_v280_small_acceptance.sh'),
    ):
        current_text = (ROOT/'scripts'/current).read_text()
        previous_text = (ROOT/'scripts'/previous).read_text()
        selected = set(re.findall(r'tests/[^\s\\]+', current_text))
        previous_selected = set(re.findall(r'tests/[^\s\\]+', previous_text))
        previous_selected.discard('tests/test_v280_release_closure.py')
        assert previous_selected <= selected
        assert {'tests/'+name+'.py' for name in REQUIRED_MODULES} <= selected
        assert 'tests/test_v280_release_closure.py' not in selected
        assert 'xfail_strict=true' in current_text
        assert '-p no:cacheprovider' in current_text



def test_new_required_modules_are_real_selected_files():
    for module in REQUIRED_MODULES:
        assert (ROOT/'tests'/(module+'.py')).is_file(), module
    for gate in ('run_v2801_short_tests.sh', 'run_v2801_small_acceptance.sh'):
        text = (ROOT/'scripts'/gate).read_text()
        for path in (ROOT/'tests').glob('test_v2801_*.py'):
            assert 'tests/'+path.name in text, path.name


def test_previous_smoke_feature_contract_is_not_reduced():
    from onnx_splitpoint_tool import v280_smoke, v2801_smoke, __build_features__
    assert v280_smoke.REQUIRED_FEATURES < v2801_smoke.REQUIRED_FEATURES
    assert v2801_smoke.REQUIRED_FEATURES <= set(__build_features__)


def test_previous_v280_entrypoints_still_exist():
    pyproject = (ROOT/'pyproject.toml').read_text()
    updater = (ROOT/'scripts/update_source_release.sh').read_text()
    for alias in ('v280', 'v2-80'):
        assert f'onnx-splitpoint-smoke-{alias} = "onnx_splitpoint_tool.v280_smoke:main"' in pyproject
        assert f'onnx-splitpoint-smoke-{alias}=onnx_splitpoint_tool.v280_smoke:main' in updater


def test_dependency_failure_is_reported_as_environment_blocked(tmp_path):
    main, gui = tmp_path/'main.xml', tmp_path/'gui.xml'
    _write_junit(main, REQUIRED_MODULES)
    _write_junit(gui, ('test_gui',))
    report = tmp_path/'report.json'
    env = {**os.environ, 'REPORT': str(report), 'STARTED': 'fixture', 'FINAL_RC': '78',
           'PYTEST_JUNIT_MAIN': str(main), 'PYTEST_JUNIT_GUI': str(gui),
           'DEPENDENCIES': 'BLOCKED',
           **{key: 'PASS' for key in ('SMOKE', 'PYTEST', 'MANIFEST', 'COMPILE', 'SHELL')}}
    result = subprocess.run([sys.executable, '-I', '-B', '-c', _report_python()],
                            env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 78, result.stdout + result.stderr
    payload = json.loads(report.read_text())
    assert payload['status'] == 'environment_blocked'
    assert payload['real_evalrun_status'] == 'NOT_RUN'


def test_current_scope_docs_distinguish_software_and_model_quality():
    docs = '\n'.join((ROOT/name).read_text() for name in (
        'TESTANLEITUNG_2.80.1.md', 'VERSION_2.80.1_BUILD_AND_TEST_REPORT.md'))
    for term in ('NOT_RUN', 'Force', 'v2.80.1-cpu-reference-remote-closure-debugexport',
                 'CPU', 'Native', 'Debug', 'FAIL', 'INCONCLUSIVE'):
        assert term in docs
