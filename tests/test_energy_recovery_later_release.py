"""Old recovery may follow a normally released later workflow, never a live owner.

Only temporary local lock inodes are used; no production locks or hardware.
"""
from datetime import datetime, timedelta, timezone
import copy
import json
import os
import socket

import pytest

from onnx_splitpoint_tool.workflow import energy_resume, run_control


RESOURCES = ['controller:capture', 'controller:nic', 'dut:192.0.2.145', 'source:192.0.2.176']


def check_release(tmp_path, monkeypatch, resource, fault=None):
    monkeypatch.setattr(run_control, 'platform_workflow_interlock_path',
                        lambda: tmp_path / 'interlock.lock')
    stamp = datetime.now(timezone.utc)
    identity = dict(run_id='old-parent', session_id='old-session', operation_id='old-capture')
    recovery = dict(identity, released_at=(stamp - timedelta(hours=2)).isoformat(),
                    resources=RESOURCES, old_attempt_validated=False, measurements_started=0)
    owner = dict(schema='onnx-splitpoint/evaluation-run-lock', schema_version=1,
                 physical_resource=resource, hostname=socket.gethostname(), state='released',
                 run_id='later-normal-parent', session_id='later-session', operation_id='later-operation',
                 pid=99999991, owner_pid=99999992,
                 released_at=(stamp - timedelta(hours=1)).isoformat())
    if fault == 'exact_recovery':
        owner['resource_recovery'] = dict(identity, resources=RESOURCES)
        owner['released_at'] = recovery['released_at']
    elif fault == 'quarantine': pass
    elif fault == 'running': owner['state'] = 'running'
    elif fault == 'resource': owner['physical_resource'] = 'dut:192.0.2.250'
    elif fault == 'outside': resource = 'source:192.0.2.250'; owner['physical_resource'] = resource
    elif fault == 'hostname': owner['hostname'] = 'foreign-controller'
    elif fault == 'live_pid': owner['pid'] = os.getpid()
    elif fault == 'live_owner': owner['owner_pid'] = os.getpid()
    elif fault == 'invalid_pid': owner['pid'] = True
    elif fault == 'missing_operation': owner.pop('operation_id')
    elif fault == 'schema': owner['schema'] = 'foreign'
    elif fault == 'version': owner['schema_version'] = 2
    elif fault == 'earlier': owner['released_at'] = (stamp - timedelta(hours=3)).isoformat()
    elif fault == 'equal': owner['released_at'] = recovery['released_at']
    elif fault == 'future': owner['released_at'] = (stamp + timedelta(hours=1)).isoformat()
    elif fault == 'naive': owner['released_at'] = stamp.replace(tzinfo=None).isoformat()
    elif fault == 'malformed': owner['released_at'] = 'not-a-time'
    elif fault == 'missing_recovery_time': recovery.pop('released_at')
    lease = run_control.EvaluationRunLock(out_root=tmp_path, run_dir=tmp_path / 'resource', owner={})
    lease.lock_path.parent.mkdir(parents=True)
    lease.lock_path.write_text(json.dumps(owner))
    if fault == 'quarantine': lease.quarantine_path.write_text('{"state":"quarantined"}')
    before = lease.lock_path.read_bytes()
    previous_recovery = copy.deepcopy(recovery)
    lease.acquire_for_recovery()
    try:
        energy_resume.released_recovery_owner(lease, resource, identity, RESOURCES, recovery)
    finally:
        lease.release(write_released_state=False)
        assert lease.lock_path.read_bytes() == before
        assert recovery == previous_recovery
        assert lease.quarantine_path.exists() == (fault == 'quarantine')


@pytest.mark.parametrize('resource', RESOURCES)
def test_later_normal_release_preserves_old_recovery_and_lock(tmp_path, monkeypatch, resource):
    check_release(tmp_path, monkeypatch, resource)


@pytest.mark.parametrize('resource', RESOURCES)
def test_original_recovery_owner_still_accepted(tmp_path, monkeypatch, resource):
    check_release(tmp_path, monkeypatch, resource, 'exact_recovery')


@pytest.mark.parametrize('resource', RESOURCES[2:])
@pytest.mark.parametrize('fault', ['quarantine', 'running', 'resource', 'outside', 'hostname',
    'live_pid', 'live_owner', 'invalid_pid', 'missing_operation', 'schema', 'version',
    'earlier', 'equal', 'future', 'naive', 'malformed', 'missing_recovery_time'])
def test_later_release_refuses_unproven_owner_without_mutation(tmp_path, monkeypatch, resource, fault):
    with pytest.raises(ValueError):
        check_release(tmp_path, monkeypatch, resource, fault)
