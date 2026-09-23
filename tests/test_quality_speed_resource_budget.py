"""AP01 controller resource discovery; no processes or hardware are started."""
from pathlib import Path

import pytest

from onnx_splitpoint_tool import quality_statistics_config as settings


def fake_linux(monkeypatch, *, cpus=8, files=None):
    values = {
        '/proc/meminfo': 'MemTotal: 64000000 kB\nMemAvailable: 56000000 kB\n',
        '/proc/self/cgroup': '0::/user.slice/session.scope\n',
        **(files or {}),
    }
    monkeypatch.setattr(settings.os, 'sched_getaffinity', lambda pid: set(range(cpus)))
    def read(path, *args, **kwargs):
        if str(path) not in values:
            raise FileNotFoundError(path)
        return values[str(path)]
    monkeypatch.setattr(Path, 'read_text', read)


def test_cpu_budget_uses_tightest_ancestor_quota_and_reserves_controller(monkeypatch):
    fake_linux(monkeypatch, files={
        '/sys/fs/cgroup/user.slice/session.scope/cpu.max': 'max 100000\n',
        '/sys/fs/cgroup/user.slice/cpu.max': '250000 100000\n',
        '/sys/fs/cgroup/cpu.max': '800000 100000\n',
    })
    value = settings.resource_budget()
    assert value['affinity_cpus'] == 8
    assert value['quota_cpus'] == 2.5
    assert value['statistics_cpu_slots'] == 1
    assert value['controller_cpu_reserve'] == 1


def test_budget_uses_available_memory_and_remaining_hierarchical_memory(monkeypatch):
    fake_linux(monkeypatch, files={
        '/sys/fs/cgroup/user.slice/session.scope/memory.max': '40000000\n',
        '/sys/fs/cgroup/user.slice/session.scope/memory.current': '5000000\n',
        '/sys/fs/cgroup/user.slice/memory.max': '50000000\n',
        '/sys/fs/cgroup/user.slice/memory.current': '20000000\n',
    })
    assert settings.resource_budget()['available_memory_bytes'] == 30000000


@pytest.mark.parametrize('cpus,expected', [(1,1),(2,1),(8,7)])
def test_unlimited_or_unavailable_cgroup_uses_actual_affinity(monkeypatch,cpus,expected):
    fake_linux(monkeypatch, cpus=cpus)
    value = settings.resource_budget()
    assert value['statistics_cpu_slots'] == expected
    assert value['available_memory_bytes'] == 56000000 * 1024


def test_exhausted_memory_is_zero_not_negative_or_nominal_limit(monkeypatch):
    fake_linux(monkeypatch, files={
        '/sys/fs/cgroup/user.slice/session.scope/memory.max': '100\n',
        '/sys/fs/cgroup/user.slice/session.scope/memory.current': '101\n',
    })
    assert settings.resource_budget()['available_memory_bytes'] == 0
