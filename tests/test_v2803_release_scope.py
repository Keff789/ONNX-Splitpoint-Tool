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
REQUIRED_MODULES = ('test_v2803_deferred_build_readiness', 'test_v2803_native_not_started', 'test_v2803_debug_pack_large_sources', 'test_v2803_runtime_diagnostic_projection', 'test_v2803_reference_workflow_gate', 'test_v2803_scope_and_diagnostic_claims', 'test_v2803_measurement_configuration', 'test_v2803_release_scope', 'test_v2803_release_closure')


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
    gate = (ROOT/'scripts/run_v2803_small_acceptance.sh').read_text()
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
    script = (ROOT/'scripts/run_v2803_short_tests.sh').read_text()
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
        ('run_v2803_short_tests.sh', 'run_v2802_short_tests.sh'),
        ('run_v2803_small_acceptance.sh', 'run_v2802_small_acceptance.sh'),
    ):
        current_text = (ROOT/'scripts'/current).read_text()
        previous_text = (ROOT/'scripts'/previous).read_text()
        selected = set(re.findall(r'tests/[^\s\\]+', current_text))
        previous_selected = set(re.findall(r'tests/[^\s\\]+', previous_text))
        previous_selected.discard('tests/test_v2802_release_closure.py')
        assert previous_selected <= selected
        assert {'tests/'+name+'.py' for name in REQUIRED_MODULES} <= selected
        assert 'tests/test_v2802_release_closure.py' not in selected
        assert 'xfail_strict=true' in current_text
        assert '-p no:cacheprovider' in current_text



def test_new_required_modules_are_real_selected_files():
    from onnx_splitpoint_tool.release_identity import VERSION

    for module in REQUIRED_MODULES:
        assert (ROOT/'tests'/(module+'.py')).is_file(), module
    for gate in ('run_v2803_short_tests.sh', 'run_v2803_small_acceptance.sh'):
        text = (ROOT/'scripts'/gate).read_text()
        # The original .3 launchers retain their original acceptance contract.
        # FIX2--5 suites were delivered later and run in the current release
        # gate; requiring them retroactively in the original script conflates
        # the archived .3 gate with the effective current selection.
        for module in REQUIRED_MODULES:
            assert 'tests/'+module+'.py' in text, module

    current_tag = 'v' + VERSION.replace('.', '')
    for suffix in ('short_tests', 'small_acceptance'):
        text = (ROOT/'scripts'/f'run_{current_tag}_{suffix}.sh').read_text()
        for path in (ROOT/'tests').glob('test_v2803_*.py'):
            if current_tag != 'v2803' and path.name == 'test_v2803_release_closure.py':
                # Current identity closure carries the historical behavior;
                # the old fixed version assertion must not run on a new build.
                assert 'tests/test_'+current_tag+'_*.py' in text or 'tests/test_'+current_tag+'_release_closure.py' in text
                continue
            assert 'tests/'+path.name in text, path.name


def test_previous_smoke_feature_contract_is_not_reduced():
    from onnx_splitpoint_tool import v2802_smoke, v2803_smoke, __build_features__
    assert v2802_smoke.REQUIRED_FEATURES < v2803_smoke.REQUIRED_FEATURES
    assert v2803_smoke.REQUIRED_FEATURES <= set(__build_features__)


def test_previous_v2802_entrypoints_still_exist():
    pyproject = (ROOT/'pyproject.toml').read_text()
    updater = (ROOT/'scripts/update_source_release.sh').read_text()
    for alias in ('v2802', 'v2-80-2'):
        assert f'onnx-splitpoint-smoke-{alias} = "onnx_splitpoint_tool.v2802_smoke:main"' in pyproject
        assert f'onnx-splitpoint-smoke-{alias}=onnx_splitpoint_tool.v2802_smoke:main' in updater


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
        'TESTANLEITUNG_2.80.3.md', 'VERSION_2.80.3_BUILD_AND_TEST_REPORT.md'))
    for term in ('NOT_RUN', 'Force', 'v2.80.3-build-readiness-native-not-started-debug-export',
                 'CPU', 'Native', 'Debug', 'FAIL', 'INCONCLUSIVE'):
        assert term in docs


def test_previous_closure_checks_are_all_carried_forward():
    import ast

    previous = ast.parse((ROOT/'tests/test_v2802_release_closure.py').read_text())
    current = ast.parse((ROOT/'tests/test_v2803_release_closure.py').read_text())
    previous_tests = {node.name: node for node in previous.body
                      if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
    current_tests = {node.name: node for node in current.body
                     if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
    assert previous_tests.keys() <= current_tests.keys()
    for name, previous_test in previous_tests.items():
        assert sum(isinstance(node, ast.Assert) for node in ast.walk(current_tests[name])) >= sum(
            isinstance(node, ast.Assert) for node in ast.walk(previous_test)), name


def test_current_scope_document_is_allowlisted_by_every_verifier():
    import ast
    from onnx_splitpoint_tool import source_integrity
    from scripts import build_source_manifest

    relative = Path('docs/RELEASE_SCOPE_V2803.md')
    assert (ROOT/relative).is_file()
    assert build_source_manifest._source_candidate(ROOT/relative, ROOT, package_version='2.80.3')
    assert source_integrity._source_candidate(relative, package_version='2.80.3')
    updater = (ROOT/'scripts/update_source_release.sh').read_text()
    embedded = updater[updater.index('ALLOWED_DOC_FILES = {'):]
    embedded_policy = ast.literal_eval(embedded.split('=', 1)[1].split('}', 1)[0] + '}')
    assert relative.as_posix() in embedded_policy
    assert embedded_policy == build_source_manifest.ALLOWED_DOC_FILES == source_integrity._ALLOWED_DOC_FILES


def _stdlib_python_launcher(path):
    """A real interpreter without site packages; no dependency-success stubs."""
    import shlex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -I -S "$@"\n')
    path.chmod(0o755)
    return path


def test_actual_standard_launcher_preserves_dependency_failure_evidence(tmp_path):
    launcher = _stdlib_python_launcher(tmp_path/'stdlib-python')
    report = tmp_path/'acceptance.json'
    result = subprocess.run(['bash', str(ROOT/'scripts/run_v2803_small_acceptance.sh'),
                             '--report', str(report)],
                            env={**os.environ, 'PY': str(launcher)},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 78, result.stdout + result.stderr
    payload = json.loads(report.read_text())
    assert payload['status'] == 'environment_blocked'
    assert payload['return_code'] == 78
    assert payload['real_evalrun_status'] == 'NOT_RUN'
    assert payload['pytest_totals']['tests'] == 0
    assert payload['checks']['acceptance_environment'] == 'BLOCKED'
    assert 'V2803_ACCEPTANCE_REPORT=' + str(report) in result.stdout
    assert 'PASS v2.80.3 small acceptance' not in result.stdout


def test_actual_short_launcher_archives_dependency_failure(tmp_path):
    import shutil
    import zipfile

    isolated_home = tmp_path/'isolated-home'
    isolated_home.mkdir()
    tool = tmp_path/'tool'
    _stdlib_python_launcher(tool/'.venv/bin/python')
    (tool/'scripts').mkdir()
    shutil.copyfile(ROOT/'scripts/check_acceptance_environment.py',
                    tool/'scripts/check_acceptance_environment.py')
    result = subprocess.run(['bash', str(ROOT/'scripts/run_v2803_short_tests.sh')],
                            env={**os.environ, 'HOME': str(isolated_home),
                                 'ONNX_SPLITPOINT_TOOL_DIR': str(tool)},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 78, result.stdout + result.stderr
    archive_line = next(line for line in result.stdout.splitlines() if line.startswith('EVIDENCE_ZIP='))
    archive = Path(archive_line.split('=', 1)[1])
    assert archive.is_relative_to(isolated_home/'Downloads')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        status_name = next(name for name in z.namelist() if name.endswith('/status.txt'))
        status = z.read(status_name).decode()
        environment_name = next(name for name in z.namelist() if name.endswith('/acceptance_environment.json'))
        environment = json.loads(z.read(environment_name))
    assert 'SHORT_TESTS_RC=78' in status
    assert 'ACCEPTANCE_ENVIRONMENT=environment_blocked' in status
    assert 'HARDWARE_EXECUTION=NOT_RUN' in status
    assert environment['status'] == 'environment_blocked'
    assert environment['packages_installed'] is False
    assert 'SHORT_TESTS=PASS' not in result.stdout
