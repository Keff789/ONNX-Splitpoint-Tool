from __future__ import annotations

import json
import threading
import time

import pytest

import onnx_splitpoint_tool.benchmark.services as benchmark_services
import onnx_splitpoint_tool.build_scheduler as build_scheduler_module
from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationCancelled,
    _hailo_case_builder_exception_outcome_v276,
    _hailo_full_builder_exception_outcome_v276,
    _hailo_pair_parallel_decision_v27550,
    _run_hailo_target_builds_v60s,
    _v60s_build_scheduler_config,
)
from onnx_splitpoint_tool.build_scheduler import BuildScheduler, BuildTaskSpec
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _profile_build_scheduler_config,
    _v60s_finish_deepx_prefetch,
    _v60s_start_deepx_prefetch,
)


PAIR_CONFIG = {
    "enabled": True,
    "max_workers": 3,
    "cpu_tokens": 8,
    "ram_mb": 12288,
    "family_limits": {"hailo8": 1, "hailo10": 1, "deepx": 1},
    "weights": {
        "hailo8": {"cpu_tokens": 4, "ram_mb": 6144},
        "hailo10": {"cpu_tokens": 4, "ram_mb": 6144},
        "deepx": {"cpu_tokens": 2, "ram_mb": 4096},
    },
}


@pytest.fixture(autouse=True)
def _stable_available_ram(monkeypatch):
    monkeypatch.setattr(
        benchmark_services,
        "_available_ram_mb_v27550",
        lambda: 65536,
    )


def test_resolved_profile_overrides_environment_and_pair_overlaps(monkeypatch):
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_BUILD_SCHEDULER_JSON",
        '{"enabled":false,"max_workers":1}',
    )
    barrier = threading.Barrier(2)
    active = 0
    peak = 0
    lock = threading.Lock()

    def builder(target: str, build_backend: str) -> str:
        nonlocal active, peak
        assert build_backend == "venv"
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=2)
        if target == "hailo8":
            time.sleep(0.03)
        with lock:
            active -= 1
        return target

    results = _run_hailo_target_builds_v60s(
        ["hailo8", "hailo10h"],
        builder,
        scheduler_config=PAIR_CONFIG,
        backend="venv",
    )

    # Completion order is intentionally different, but merge order remains the
    # declared target order.
    assert results == ["hailo8", "hailo10h"]
    assert peak == 2


def test_pair_falls_back_to_serial_when_cpu_budget_is_too_small():
    config = dict(PAIR_CONFIG, cpu_tokens=7)
    active = 0
    peak = 0
    lock = threading.Lock()
    logs: list[str] = []

    def builder(target: str, build_backend: str) -> str:
        nonlocal active, peak
        assert build_backend == "venv"
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return target

    assert _run_hailo_target_builds_v60s(
        ["hailo8", "hailo10h"],
        builder,
        scheduler_config=config,
        backend="venv",
        log=logs.append,
    ) == ["hailo8", "hailo10h"]
    assert peak == 1
    assert any("effective=False" in line and "insufficient_cpu_tokens" in line for line in logs)


def test_only_one_hailo8_hailo10_pair_is_parallelised():
    logs: list[str] = []
    seen: list[str] = []
    results = _run_hailo_target_builds_v60s(
        ["hailo8", "hailo10h", "hailo8l"],
        lambda target, _build_backend: seen.append(target) or target,
        scheduler_config=PAIR_CONFIG,
        backend="venv",
        log=logs.append,
    )
    assert results == ["hailo8", "hailo10h", "hailo8l"]
    assert seen == results
    assert any("effective=False" in line and "targets_are_not" in line for line in logs)


@pytest.mark.parametrize("backend", ["", "auto", "local", "wsl", "subprocess"])
def test_pair_is_serial_without_process_isolation(backend, monkeypatch):
    if backend == "auto":
        monkeypatch.setattr(
            benchmark_services,
            "_managed_venv_pair_available_v27550",
            lambda _targets: (False, "test-managed-venv-missing"),
        )
    active = 0
    peak = 0
    lock = threading.Lock()
    logs: list[str] = []

    def builder(target: str, build_backend: str) -> str:
        nonlocal active, peak
        assert build_backend == backend
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return target

    assert _run_hailo_target_builds_v60s(
        ["hailo8", "hailo10h"],
        builder,
        scheduler_config=PAIR_CONFIG,
        backend=backend,
        log=logs.append,
    ) == ["hailo8", "hailo10h"]
    assert peak == 1
    expected_reason = (
        "auto_managed_venv_pair_unavailable"
        if backend == "auto"
        else "backend_not_explicit_process_isolated_venv"
    )
    assert any("effective=False" in line and expected_reason in line for line in logs)


def test_auto_pair_with_two_managed_venvs_forces_venv(monkeypatch):
    monkeypatch.setattr(
        benchmark_services,
        "_managed_venv_pair_available_v27550",
        lambda _targets: (True, "hailo8:/venv8,hailo10:/venv10"),
    )
    barrier = threading.Barrier(2)
    seen_backends: list[str] = []
    lock = threading.Lock()

    def builder(target: str, build_backend: str) -> str:
        with lock:
            seen_backends.append(build_backend)
        barrier.wait(timeout=2)
        return target

    assert _run_hailo_target_builds_v60s(
        ["hailo8", "hailo10h"],
        builder,
        scheduler_config=PAIR_CONFIG,
        backend="auto",
    ) == ["hailo8", "hailo10h"]
    assert seen_backends == ["venv", "venv"]


def test_parallel_failure_joins_sibling_and_keeps_events(tmp_path, monkeypatch):
    event_path = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv("ONNX_SPLITPOINT_BUILD_SCHEDULER_LOG", str(event_path))
    sibling_finished = threading.Event()

    def builder(target: str, _build_backend: str) -> dict:
        if target == "hailo8":
            raise RuntimeError("expected-hailo8-failure")
        time.sleep(0.03)
        sibling_finished.set()
        return {
            "hw_arch": target,
            "target_output": {"part1": "compiled.hef"},
            "errors": [],
        }

    outcomes = _run_hailo_target_builds_v60s(
        ["hailo8", "hailo10h"],
        builder,
        scheduler_config=PAIR_CONFIG,
        backend="venv",
        exception_outcome_factory=lambda target, exc: (
            _hailo_case_builder_exception_outcome_v276(
                target,
                exc,
                label="hailo-targets:b044",
                boundary=44,
                folder="b044",
                hef_part1=True,
                hef_part2=True,
            )
        ),
    )
    assert sibling_finished.is_set()
    assert [outcome["hw_arch"] for outcome in outcomes] == [
        "hailo8",
        "hailo10h",
    ]
    failed, succeeded = outcomes
    assert failed["status"] == "terminal_failed"
    assert failed["terminal_reason"] == "hailo_target_builder_exception"
    assert failed["target_output"]["part1_error"] == (
        "RuntimeError: expected-hailo8-failure"
    )
    assert failed["target_output"]["part2_error"] == (
        "RuntimeError: expected-hailo8-failure"
    )
    assert failed["first_rejection"]["reason"] == (
        "hailo_target_builder_exception"
    )
    assert succeeded["target_output"]["part1"] == "compiled.hef"
    events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 2
    assert {event["status"] for event in events} == {"failed", "ok"}


def test_parallel_full_exception_is_terminal_and_keeps_successful_sibling():
    def builder(target: str, _build_backend: str) -> dict:
        if target == "hailo10h":
            raise ValueError("hailo10-full-failed")
        return {
            "hw_arch": target,
            "target_output": {"full": "hailo/hailo8/full/compiled.hef"},
            "errors": [],
        }

    outcomes = _run_hailo_target_builds_v60s(
        ["hailo8", "hailo10h"],
        builder,
        scheduler_config=PAIR_CONFIG,
        backend="venv",
        exception_outcome_factory=lambda target, exc: (
            _hailo_full_builder_exception_outcome_v276(
                target,
                exc,
                label="hailo-full-targets",
            )
        ),
    )

    assert outcomes[0]["target_output"]["full"].endswith("compiled.hef")
    assert outcomes[1]["status"] == "terminal_failed"
    assert outcomes[1]["target_output"]["full_required"] is True
    assert outcomes[1]["target_output"]["full_error"] == (
        "ValueError: hailo10-full-failed"
    )


def test_parallel_cancellation_still_joins_sibling_and_propagates():
    sibling_finished = threading.Event()

    def builder(target: str, _build_backend: str) -> str:
        if target == "hailo8":
            raise BenchmarkGenerationCancelled("cancelled-by-user")
        time.sleep(0.03)
        sibling_finished.set()
        return target

    with pytest.raises(
        BenchmarkGenerationCancelled,
        match="cancelled-by-user",
    ):
        _run_hailo_target_builds_v60s(
            ["hailo8", "hailo10h"],
            builder,
            scheduler_config=PAIR_CONFIG,
            backend="venv",
        )
    assert sibling_finished.is_set()


def test_pair_ram_gate_reserves_live_memory(monkeypatch):
    monkeypatch.setattr(
        benchmark_services,
        "_available_ram_mb_v27550",
        lambda: 13000,
    )
    decision = _hailo_pair_parallel_decision_v27550(
        ["hailo8", "hailo10h"],
        dict(PAIR_CONFIG, ram_mb=20000, ram_reserve_mb=2048),
        backend="venv",
    )
    assert decision["effective"] is False
    assert decision["ram_pool_mb"] == 10952
    assert decision["reason"].startswith("insufficient_ram_mb")


def test_impossible_scheduler_request_is_rejected_before_worker_wait():
    with BuildScheduler(max_workers=1, cpu_tokens=4, ram_mb=1024) as scheduler:
        with pytest.raises(ValueError, match="ram_request_exceeds_pool"):
            scheduler.submit(
                BuildTaskSpec("too-large", "hailo8", cpu_tokens=1, ram_mb=2048),
                lambda: None,
            )


def test_profile_scheduler_mapping_supports_resolved_and_nested_shapes(monkeypatch):
    monkeypatch.setenv("ONNX_SPLITPOINT_BUILD_SCHEDULER_JSON", '{"enabled":false}')
    assert _profile_build_scheduler_config({"build_scheduler": PAIR_CONFIG}) == PAIR_CONFIG
    assert _profile_build_scheduler_config({"build": {"scheduler": PAIR_CONFIG}}) == PAIR_CONFIG
    assert _v60s_build_scheduler_config(PAIR_CONFIG)["enabled"] is True


def test_deepx_prefetch_is_deferred_while_hailo_pair_is_requested(tmp_path):
    profile = {
        "run_profiles": [
            {"id": "hailo8", "enabled": True},
            {"id": "hailo10h", "enabled": True},
            {"id": "deepx_m1", "enabled": True},
        ],
        "build_scheduler": dict(PAIR_CONFIG, prefetch_deepx_full=True),
    }
    logs: list[str] = []
    handle = _v60s_start_deepx_prefetch(
        run_dir=tmp_path,
        model_id="model",
        model_path=str(tmp_path / "model.onnx"),
        model_row={"id": "model"},
        task="classification",
        profile_payload=profile,
        targets=["hailo8", "hailo10h", "deepx_m1"],
        suite_dir=tmp_path / "suite",
        log=logs.append,
        hailo_backend="venv",
    )
    assert handle is not None and handle.get("deferred") is True
    assert any("deferred" in line and "Hailo-8/Hailo-10" in line for line in logs)


def test_deepx_prefetch_is_not_deferred_for_serial_hailo_pair(
    tmp_path,
    monkeypatch,
):
    submitted: list[BuildTaskSpec] = []

    class FakeScheduler:
        def __init__(self, **_kwargs):
            pass

        def submit(self, spec, _fn, **_kwargs):
            submitted.append(spec)
            return object()

        def shutdown(self, _wait=True):
            pass

    monkeypatch.setattr(build_scheduler_module, "BuildScheduler", FakeScheduler)
    profile = {
        "run_profiles": [
            {"id": "hailo8", "enabled": True},
            {"id": "hailo10h", "enabled": True},
            {"id": "deepx_m1", "enabled": True},
        ],
        "build_scheduler": dict(
            PAIR_CONFIG,
            cpu_tokens=7,
            prefetch_deepx_full=True,
        ),
    }
    logs: list[str] = []
    handle = _v60s_start_deepx_prefetch(
        run_dir=tmp_path,
        model_id="model",
        model_path=str(tmp_path / "model.onnx"),
        model_row={"id": "model"},
        task="classification",
        profile_payload=profile,
        targets=["hailo8", "hailo10h", "deepx_m1"],
        suite_dir=tmp_path / "suite",
        log=logs.append,
        hailo_backend="venv",
    )
    assert handle is not None and handle.get("deferred") is not True
    assert len(submitted) == 1
    assert any("pair is serial" in line and "insufficient_cpu_tokens" in line for line in logs)


def test_cancelled_deferred_deepx_prefetch_is_not_started(tmp_path, monkeypatch):
    cancel_event = threading.Event()
    cancel_event.set()
    started = False

    def forbidden_start(**_kwargs):
        nonlocal started
        started = True
        raise AssertionError("deferred prefetch must not be started after cancellation")

    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding._v60s_start_deepx_prefetch",
        forbidden_start,
    )
    report = _v60s_finish_deepx_prefetch(
        {"deferred": True, "request": {"unused": True}},
        suite_dir=tmp_path,
        log=lambda _message: None,
        cancel_event=cancel_event,
    )
    assert started is False
    assert report["status"] == "cancelled_before_deferred_start"
    assert report["cancelled"] is True
    assert report["deferred_for_hailo_pair"] is True
