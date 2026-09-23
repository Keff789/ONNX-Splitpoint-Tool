"""Opt-in acceptance counters using real processes and real advisory locks.

Every journal and leaf marker belongs to pytest's external temporary directory.
No actual collector, workflow, network or live acceptance counter is invoked.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from onnx_splitpoint_tool.energy.task_budget import reserve_controlled_test_start
from onnx_splitpoint_tool.process_control import ProcessTreeRegistry


_LEAF = r'''
import fcntl, json, sys
from pathlib import Path
journal, marker, kind, identity = sys.argv[1:]
with open(journal) as handle:
    fcntl.flock(handle, fcntl.LOCK_SH)
    data = json.load(handle)
    matches = [row for row in data["entries"] if row["kind"] == kind and row["identity"] == identity]
    assert len(matches) == 1, "physical leaf was dispatched before its reservation was durable"
Path(marker).write_text(json.dumps(matches[0]))
'''


_CHILD = r'''
import json, os, subprocess, sys, time
from pathlib import Path
from onnx_splitpoint_tool.energy.task_budget import reserve_controlled_test_start
kind, identity, ready, release, marker, leaf_script = sys.argv[1:]
Path(ready).touch()
while not Path(release).exists():
    time.sleep(0.005)
try:
    row = reserve_controlled_test_start(kind, identity)
except (RuntimeError, ValueError) as error:
    print(json.dumps({"status": "blocked", "error": str(error)}), flush=True)
else:
    assert row is not None
    subprocess.run([sys.executable, "-B", leaf_script,
        os.environ["ONNX_SPLITPOINT_TEST_START_JOURNAL"], marker, kind, identity], check=True)
    print(json.dumps({"status": "started", "reservation": row}), flush=True)
'''


def _journal(path, *, previous=(), limits=None):
    value = {"schema": "onnx-splitpoint/controlled-test-starts",
             "limits": limits or {"collector": 36, "workflow": 3},
             "entries": list(previous)}
    path.write_text(json.dumps(value))
    return path


def _attempts(directory, journal, kind, identities):
    directory.mkdir()
    leaf = directory / "leaf.py"
    leaf.write_text(_LEAF)
    release = directory / "release"
    env = dict(os.environ, ONNX_SPLITPOINT_TEST_START_JOURNAL=str(journal),
               PYTHONDONTWRITEBYTECODE="1")
    registry = ProcessTreeRegistry()
    children = []
    try:
        for index, identity in enumerate(identities):
            ready, marker = directory / f"ready-{index}", directory / f"leaf-{index}.json"
            proc = subprocess.Popen([sys.executable, "-B", "-c", _CHILD,
                kind, identity, str(ready), str(release), str(marker), str(leaf)],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True)
            registry.register(proc, label="controlled-counter-fixture")
            children.append((proc, ready, marker))
        deadline = time.monotonic() + 5
        while not all(ready.exists() for _, ready, _ in children):
            assert time.monotonic() < deadline, "test children failed to reach the controlled start barrier"
            assert all(proc.poll() is None for proc, _, _ in children)
            time.sleep(0.005)
        release.touch()
        results = []
        for proc, _, marker in children:
            output, error = proc.communicate(timeout=5)
            assert proc.returncode == 0, error
            result = json.loads(output)
            assert marker.exists() is (result["status"] == "started")
            if marker.exists():
                assert json.loads(marker.read_text()) == result["reservation"]
            results.append(result)
        return results
    finally:
        for proc, _, _ in children:
            if proc.poll() is None:
                registry.terminate_registered(proc, grace_s=0.1)
            registry.unregister(proc)
        registry.assert_quiescent()


def test_normal_user_workflow_without_opt_in_has_no_counter_side_effect(tmp_path, monkeypatch):
    monkeypatch.delenv("ONNX_SPLITPOINT_TEST_START_JOURNAL", raising=False)
    assert reserve_controlled_test_start("collector", "ordinary-capture") is None
    assert reserve_controlled_test_start("workflow", "ordinary-workflow") is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind,limit", [("collector", 36), ("workflow", 3)])
def test_real_process_race_respects_last_remaining_start_before_leaf_dispatch(tmp_path, kind, limit):
    previous = [{"kind": kind, "identity": f"prior-{index}", "owner_pid": 1,
                 "reserved_at_unix": 1.0} for index in range(limit - 1)]
    journal = _journal(tmp_path / "starts.json", previous=previous)
    results = _attempts(tmp_path / "competing-processes", journal, kind,
                        [f"contender-{index}" for index in range(6)])
    assert sum(result["status"] == "started" for result in results) == 1
    assert sum(result["status"] == "blocked" for result in results) == 5
    assert all("budget exhausted:" + kind in result["error"]
               for result in results if result["status"] == "blocked")
    persisted = json.loads(journal.read_text())
    assert len(persisted["entries"]) == limit
    assert persisted["entries"][:-1] == previous
    assert len(list((tmp_path / "competing-processes").glob("leaf-*.json"))) == 1


def test_resume_preserves_workflow_count_and_duplicate_never_dispatches_again(tmp_path):
    journal = _journal(tmp_path / "starts.json")
    first = _attempts(tmp_path / "first", journal, "workflow", ["g1"])
    assert first[0]["status"] == "started"
    saved = journal.read_bytes()
    duplicate = _attempts(tmp_path / "duplicate", journal, "workflow", ["g1"])
    assert duplicate[0]["status"] == "blocked"
    assert "already reserved" in duplicate[0]["error"]
    assert journal.read_bytes() == saved
    resumed = _attempts(tmp_path / "resume", journal, "workflow", ["g2", "g3"])
    assert all(result["status"] == "started" for result in resumed)
    saved = journal.read_bytes()
    fourth = _attempts(tmp_path / "fourth", journal, "workflow", ["g4"])
    assert fourth[0]["status"] == "blocked"
    assert "budget exhausted:workflow" in fourth[0]["error"]
    assert journal.read_bytes() == saved
    assert {row["identity"] for row in json.loads(saved)["entries"]} == {"g1", "g2", "g3"}


def test_explicit_counter_path_is_used_without_environment_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("ONNX_SPLITPOINT_TEST_START_JOURNAL", raising=False)
    journal = _journal(tmp_path / "explicit.json")
    row = reserve_controlled_test_start("collector", "explicit-attempt", journal_path=journal)
    assert json.loads(journal.read_text())["entries"] == [row]
    assert row["identity"] == "explicit-attempt"
    assert row["owner_pid"] == os.getpid()
