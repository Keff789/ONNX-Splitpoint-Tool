"""Terminal evidence survives a downstream reporting failure; no hardware."""
import json
import tarfile

from onnx_splitpoint_tool.workflow import yolo_augmentation as workflow
from onnx_splitpoint_tool import yolo_augmentation_reporting
from tests.test_yolo_augmentation_workflow import frozen


def test_reporting_failure_preserves_terminal_exitcode_and_diagnostics(frozen, monkeypatch):
    path, plan = frozen

    def broken_report(*args, **kwargs):
        raise ValueError('unbound new energy result')

    monkeypatch.setattr(yolo_augmentation_reporting, 'write_augmentation_reports', broken_report)
    assert workflow.write_final_delivery(plan, 'ok', '', 0) == 1
    output = path.parent.parent
    terminal = json.loads((output / 'terminal-status.json').read_text())
    assert terminal['status'] == 'delivery_failed'
    assert terminal['exitcode'] == 1
    assert terminal['automatic_restart'] is False
    assert 'unbound new energy result' in terminal['error']
    assert (output / 'exitcode.txt').read_text() == '1\n'
    assert (output / 'finished_at.txt').read_text().strip()
    with tarfile.open(output / 'completion-diagnostics.tar.gz') as archive:
        assert 'terminal-status.json' in archive.getnames()
        assert 'delivery-error.json' in archive.getnames()


def test_technical_workflow_stop_is_not_upgraded_by_successful_reporting(frozen, monkeypatch):
    path, plan = frozen
    monkeypatch.setattr(yolo_augmentation_reporting, 'write_augmentation_reports',
                        lambda *a, **kw: {'new_completion_pair_count': 0})
    assert workflow.write_final_delivery(plan, 'failed', 'cleanup unproven', 1) == 1
    terminal = json.loads((path.parent.parent / 'terminal-status.json').read_text())
    assert terminal['status'] == 'failed'
    assert terminal['error'] == 'cleanup unproven'
    assert terminal['summary']['new_completion_pair_count'] == 0
