"""F3 process errors are real child failures, never synthesized successes."""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import threading

import pytest

import onnx_splitpoint_tool.management_reference as reference
from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner

ORIGINAL_CAUSE = (
    'generic Full central-quality dispatch identity is incomplete: '
    'quality-evidence-eval-id,quality-evidence-setup-id'
)


def _suite(root: Path, body: str) -> Path:
    suite = root / 'suite'
    suite.mkdir(parents=True)
    (suite / 'benchmark_plan.json').write_text(json.dumps({
        'runs': [{'id': 'ort_cpu', 'provider': 'cpu'}],
    }))
    (suite / 'benchmark_set.json').write_text(json.dumps({
        'cases': [{'case_id': 'b001'}],
    }))
    (suite / 'benchmark_suite.py').write_text(body)
    return suite


def _run(root: Path, body: str, *, model: str = 'failed_model', **kwargs):
    suite = _suite(root, body)
    out = root / 'quality_management' / 'references' / model
    logs = []
    registry = kwargs.pop('process_registry', ProcessTreeRegistry())
    status = reference.generate_management_cpu_reference(
        suite_dir=suite, output_dir=out, model_id=model, workers=1,
        log=logs.append, process_registry=registry, **kwargs,
    )
    assert registry.active_pids() == []
    assert list((out / 'workspaces').iterdir()) == []
    return status, logs, out


def _consumer(root: Path, status: dict):
    future = concurrent.futures.Future()
    future.set_result(status)
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = root
    runner.run_id = 'test_eval'  # Same run identity as the scoped fixture requests.
    runner._management_reference_futures = {status['model_id']: future}
    runner._management_reference_source_contracts = {
        status['model_id']: status['source_contract_sha256'],
    }
    # The service is never reached in a failed reference dependency.
    runner._central_quality_service = object()
    runner._central_quality_coord_executor = object()
    return runner


def test_original_exception_reaches_real_process_callback_and_central_result(tmp_path):
    status, logs, out = _run(tmp_path, f'raise RuntimeError({ORIGINAL_CAUSE!r})\n')
    assert status['status'] == 'failed'
    assert status['return_code'] == 1
    assert status['error'] == 'RuntimeError: ' + ORIGINAL_CAUSE
    assert status['errors'] == [status['error'], 'quality_reference_not_emitted']
    assert status['failure_stage'] == 'reference_process'
    assert status['exception_type'] == 'RuntimeError'
    assert status['exception_message'] == ORIGINAL_CAUSE
    assert status['source_contract_sha256']
    assert status['stdout_path'] == str(out / 'management_cpu_reference_stdout.txt')
    assert 'RuntimeError: ' + ORIGINAL_CAUSE in Path(status['stdout_path']).read_text()
    assert any(ORIGINAL_CAUSE in line and status['stdout_path'] in line and 'return_code=1' in line for line in logs)
    runner = _consumer(tmp_path, status)
    with pytest.raises(reference.ManagementCPUReferenceError, match=ORIGINAL_CAUSE) as caught:
        runner._management_reference_records('failed_model')
    assert caught.value.reference_status['return_code'] == 1
    for case_id in ('b001', 'b002'):
        request = tmp_path / 'quality_inputs' / 'device' / case_id / 'results_hailo8' / 'full_request.json'
        request.parent.mkdir(parents=True)
        request.write_text(json.dumps({'task': 'classification', 'variant': 'full', 'eval_run_id': 'test_eval'}))
        result = runner._evaluate_central_quality_request('failed_model', request)
        assert result['status'] == result['technical_status'] == 'failed'
        assert result['decision'] == result['scientific_status'] == 'unavailable'
        assert result['failure_stage'] == 'management_cpu_reference'
        assert result['case_id'] == case_id
        assert result['model_id'] == 'failed_model'
        assert result['source_setup_id'] == 'device'
        assert ORIGINAL_CAUSE in result['error']
        for key in ('return_code', 'stdout_path', 'source_contract_sha256', 'error'):
            assert result['management_cpu_reference'][key] == status[key]
        assert 'stdout_excerpt' not in result['management_cpu_reference']
        assert not {'latency_ms', 'fps', 'energy_j', 'primary'}.intersection(result)
    assert len(list(tmp_path.rglob('*_central_result.json'))) == 2


@pytest.mark.parametrize('body,rc,primary,secondary', [
    ('pass\n', 0, 'quality_reference_not_emitted', None),
    ('raise SystemExit(7)\n', 7, 'reference_runner_rc_7', 'quality_reference_not_emitted'),
    ("from pathlib import Path\nPath('task_quality_inputs').mkdir()\n"
     "Path('task_quality_inputs/canonical_classification_reference.json').write_text('{}')\n"
     'raise SystemExit(17)\n', 17, 'reference_runner_rc_17', None),
])
def test_no_output_or_nonzero_with_partial_output_never_publishes(tmp_path, body, rc, primary, secondary):
    status, logs, out = _run(tmp_path, body)
    assert status['status'] == 'failed'
    assert status['return_code'] == rc
    assert status['error'] == primary
    if secondary:
        assert secondary in status['errors']
    assert not list(out.rglob('canonical_cpu_reference.json'))


def test_invalid_generated_reference_reports_concrete_validation_error(tmp_path):
    status, logs, out = _run(tmp_path,
        "from pathlib import Path\nPath('task_quality_inputs').mkdir()\n"
        "Path('task_quality_inputs/canonical_classification_reference.json').write_text('{}')\n")
    assert status['status'] == 'failed'
    assert status['return_code'] == 0
    assert status['failure_stage'] == 'reference_validation'
    assert status['error'] == 'ValueError: canonical reference schema is invalid'
    assert any(status['error'] in line for line in logs)
    assert not list(out.rglob('canonical_cpu_reference.json'))


def test_timeout_is_observed_from_deadline_not_log_text(tmp_path):
    status, _, out = _run(tmp_path, "import time\nprint('running', flush=True)\ntime.sleep(60)\n", timeout_s=1)
    assert status['status'] == 'failed'
    assert status['timed_out'] is True
    assert status['cancelled'] is False
    assert status['return_code'] != 0
    assert status['error'] == 'reference_runner_timeout: budget_s=1'
    second, _, _ = _run(tmp_path / 'log_only', "print('[management-reference] timeout')\nraise SystemExit(9)\n", timeout_s=10)
    assert second['timed_out'] is False
    assert second['error'] == 'reference_runner_rc_9'


@pytest.mark.parametrize('before_start', [True, False])
def test_cancellation_has_own_cause_and_process_registry_is_empty(tmp_path, before_start):
    cancel = threading.Event()
    timer = None
    if before_start:
        cancel.set()
    else:
        timer = threading.Timer(.4, cancel.set)
        timer.start()
    try:
        status, logs, out = _run(tmp_path, 'import time\ntime.sleep(60)\n', cancel_event=cancel, timeout_s=10)
    finally:
        if timer:
            timer.join()
    assert status['status'] == 'cancelled'
    assert status['cancelled'] is True
    assert status['timed_out'] is False
    assert status['error'] == 'reference_cancelled'
    assert status['errors'] == ['reference_cancelled']
    if before_start:
        assert status['return_code'] == 130


def test_late_cancellation_does_not_replace_finished_runtime_exception(tmp_path):
    cancel = threading.Event()
    class CancelOnUnregister(ProcessTreeRegistry):
        def unregister(self, proc):
            super().unregister(proc)
            cancel.set()
    status, _, _ = _run(tmp_path, "raise RuntimeError('original failure')\n",
        cancel_event=cancel, process_registry=CancelOnUnregister())
    assert cancel.is_set()
    assert status['status'] == 'failed'
    assert status['cancelled'] is False
    assert status['error'] == 'RuntimeError: original failure'


def test_large_stdout_keeps_small_summary_and_full_original(tmp_path):
    status, logs, out = _run(tmp_path,
        "print('x' * 200000)\nraise RuntimeError('bounded cause')\n")
    assert Path(status['stdout_path']).stat().st_size > 200000
    assert len(status['stdout_excerpt']) <= 4096
    assert status['stdout_excerpt_is_tail'] is True
    assert status['error'] == 'RuntimeError: bounded cause'
    assert max(map(len, logs)) < 8192


def test_chained_exception_retains_last_cause(tmp_path):
    status, _, _ = _run(tmp_path,
        "try:\n    raise ValueError('earlier')\nexcept ValueError:\n    raise RuntimeError('final')\n")
    assert status['error'] == 'RuntimeError: final'


def test_ordinary_log_trailers_cannot_replace_traceback_exception(tmp_path):
    status, _, _ = _run(tmp_path,
        "import sys, traceback\n"
        "try:\n    raise RuntimeError('real cause')\n"
        "except RuntimeError:\n    traceback.print_exc()\n"
        "print('Done', file=sys.stderr)\nprint('STOP', file=sys.stderr)\n"
        "print('Summary: stopped', file=sys.stderr)\nraise SystemExit(1)\n")
    assert status['error'] == 'RuntimeError: real cause'


def test_controller_keyboard_interrupt_cancels_and_reaps_child(tmp_path, monkeypatch):
    original_sleep = reference.time.sleep
    triggered = False
    def interrupt_once(seconds):
        nonlocal triggered
        if not triggered:
            triggered = True
            raise KeyboardInterrupt()
        return original_sleep(seconds)
    monkeypatch.setattr(reference.time, 'sleep', interrupt_once)
    status, _, _ = _run(tmp_path, 'import time\ntime.sleep(60)\n', timeout_s=10)
    assert status['status'] == 'cancelled'
    assert status['error'] == 'reference_cancelled'
    assert status['return_code'] != 0


@pytest.mark.parametrize('model', [
    'mobilenet_v3_large', 'resnet50', 'regnet_x_1_6gf',
    'yolo11l', 'yolo26m', 'yolo26s',
])
def test_original_six_stdout_files_retain_same_terminal_exception(model):
    path = Path(__file__).parent / 'fixtures' / 'v2801_cpu_reference_details' / 'quality_management' / 'references' / model / 'management_cpu_reference_stdout.txt'
    original = path.read_bytes()
    diagnostic = reference._reference_stdout_diagnostic(path)
    assert diagnostic['exception_summary'] == 'RuntimeError: ' + ORIGINAL_CAUSE
    assert path.read_bytes() == original


def test_real_reference_publication_failure_keeps_older_immutable_reference(tmp_path, monkeypatch):
    from test_v2801_cpu_reference_dispatch import run_real_reference

    suite, run_dir, out, first, _ = run_real_reference(tmp_path, 'classification')
    assert first['status'] == 'completed', first
    previous = Path(first['reference_path'])
    original = previous.read_bytes()
    plan_path = suite / 'benchmark_plan.json'
    plan = json.loads(plan_path.read_text())
    plan['diagnostic_source_revision'] = 'publication_failure_probe'
    plan_path.write_text(json.dumps(plan))
    publications = []
    def fail_publication(source, target):
        # Real ORT already generated this reference and validation succeeds;
        # only the filesystem publication operation is fault-injected.
        metadata = reference._validate_generated_reference(source)
        assert metadata['record_count'] == 3
        publications.append(str(source))
        raise PermissionError('injected publication target denied')
    monkeypatch.setattr(reference, '_publish_immutable_reference', fail_publication)
    logs = []
    status = reference.generate_management_cpu_reference(
        suite_dir=suite, output_dir=out, model_id='tiny_classification',
        workers=1, timeout_s=60, log=logs.append,
    )
    assert len(publications) == 1
    assert status['status'] == 'failed'
    assert status['return_code'] == 0
    assert status['failure_stage'] == 'reference_publication'
    assert status['error'] == 'PermissionError: injected publication target denied'
    assert any(status['error'] in line for line in logs)
    assert status['source_contract_sha256'] != first['source_contract_sha256']
    assert previous.read_bytes() == original
    assert list(out.rglob('canonical_cpu_reference.json')) == [previous]


def test_shared_registry_cancellation_before_first_poll_is_retained(tmp_path):
    class CancelOnRegister(ProcessTreeRegistry):
        def register(self, proc, **kwargs):
            result = super().register(proc, **kwargs)
            self.terminate_all(grace_s=.1)
            return result
    status, _, _ = _run(tmp_path, 'import time\ntime.sleep(60)\n',
        process_registry=CancelOnRegister(), timeout_s=10)
    assert status['status'] == 'cancelled'
    assert status['error'] == 'reference_cancelled'
    assert status['cancelled'] is True
    assert status['timed_out'] is False
