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
REQUIRED_MODULES = (
    'test_v280_hailo8_compute_env', 'test_v280_runtime_cleanup',
    'test_v280_artifact_reuse', 'test_v280_release_scope', 'test_v280_release_closure',
)


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
    gate = (ROOT/'scripts/run_v280_small_acceptance.sh').read_text()
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
    script = (ROOT/'scripts/run_v280_short_tests.sh').read_text()
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
        ('run_v280_short_tests.sh', 'run_v27934_short_tests.sh'),
        ('run_v280_small_acceptance.sh', 'run_v27934_small_acceptance.sh'),
    ):
        current_text = (ROOT/'scripts'/current).read_text()
        previous_text = (ROOT/'scripts'/previous).read_text()
        selected = set(re.findall(r'tests/[^\s\\]+', current_text))
        previous_selected = set(re.findall(r'tests/[^\s\\]+', previous_text))
        previous_selected.discard('tests/test_v27934_release_closure.py')
        assert previous_selected <= selected
        assert {'tests/'+name+'.py' for name in REQUIRED_MODULES} <= selected
        assert 'tests/test_v27934_release_closure.py' not in selected
        assert 'xfail_strict=true' in current_text
        assert '-p no:cacheprovider' in current_text


def test_legacy_runtime_entrypoint_and_current_wrapper_share_one_main(monkeypatch):
    import importlib.util
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import hailo_model_runtime_probe_v27934 as retained
    spec = importlib.util.spec_from_file_location('runtime_current_wrapper', ROOT/'scripts/hailo_model_runtime_probe_v280.py')
    current = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(current)
    assert current.main is retained.main


def test_current_gpu_launcher_uses_shared_collector_without_new_worker_copy():
    script = (ROOT/'scripts/run_hailo_gpu_compute_v280.sh').read_text()
    assert 'hailo_gpu_diagnostics/run_hailo_gpu_smoke.sh' in script
    assert '--compiler-context' in script
    assert not (ROOT/'scripts/hailo_gpu_diagnostics/worker_context_v280.py').exists()


def test_current_scope_docs_distinguish_software_and_hardware_acceptance():
    docs = '\n'.join((ROOT/name).read_text() for name in (
        'TESTANLEITUNG_2.80.md', 'VERSION_2.80_BUILD_AND_TEST_REPORT.md'))
    for term in ('first_selected_model_only', 'quality_first_trt_binding_ready=False',
                 'not_available', 'NOT_RUN', 'Force', 'v2.80-hailo-reuse-env-cleanup'):
        assert term in docs
    assert 'Hailo8' in docs and 'Hailo10' in docs
