"""Parent setup-completion admission with controlled DUT/evaluator leaves.

Real setup dispatch, stage lifecycle, request discovery, service ownership and
coordinator threads run locally. Request contents are scheduling fixtures; no
SSH, model execution, process pool or statistics computation is permitted.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import pytest

from onnx_splitpoint_tool import quality_service, quality_statistics_blocks
from onnx_splitpoint_tool.process_control import bind_process_registry, current_process_registry
from onnx_splitpoint_tool.workflow.artifacts import write_json
from onnx_splitpoint_tool.workflow.execution_binding import PhysicalSetupDispatchManager
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from tests.test_ap08_stage_lifecycle import MODELS, _runner, _stage_path


SETUPS = ("fixture_fast", "fixture_slow")


def _wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("controlled setup/quality rendezvous timed out")
        threading.Event().wait(0.01)


class _ControlledSetupQuality(EvaluationWorkflowRunner):
    """Replace only external benchmark/evaluation leaves; observe real queues."""

    def _queue_central_quality_requests(self, model_id, *, results_root=None):
        self.queue_calls.append((threading.get_ident(), model_id, results_root))
        return super()._queue_central_quality_requests(model_id, results_root=results_root)

    def _ensure_central_quality_service(self):
        service = super()._ensure_central_quality_service()
        self.service_observations.append((threading.get_ident(), id(service),
            id(self._central_quality_coord_executor)))
        return service

    def _quality_root(self, model_id, setup):
        return self.run_dir / "models" / model_id / "benchmark_results/quality_inputs" / setup

    def _request(self, model_id, setup, case, *, ready):
        path = self._quality_root(model_id, setup) / case / "results_fixture/task_quality_inputs/composed_request.json"
        write_json(path, {"model_id": model_id, "setup_id": setup, "case_id": case, "ready": ready})
        return path

    def _stage_run_benchmarks_steps(self, model_id, row, *, dispatch_manager,
                                    on_setup_complete=None, dispatch_log=None):
        assert isinstance(dispatch_manager, PhysicalSetupDispatchManager)
        self.callback_observations.append((model_id, on_setup_complete))
        self.dispatch_manager_ids.add(id(dispatch_manager))

        def external_setup(setup):
            assert current_process_registry() is self._process_registry
            gate = self.release_slow if setup == SETUPS[1] else self.release_fast
            self.setup_started[(model_id, setup)].set()
            assert gate.wait(10), "test-owned DUT leaf was not released"
            cases = ("b001", "b002", "b003") if setup == SETUPS[0] else ("b004",)
            for case in cases:
                self._request(model_id, setup, case, ready=True)
            self.setup_downloaded[(model_id, setup)].set()
            if on_setup_complete is not None:
                self.callback_threads.append(threading.get_ident())
                # Remote retries can repeat a notification for identical files.
                on_setup_complete(model_id, setup, {})
                on_setup_complete(model_id, setup, {})
            return setup

        def dispatch():
            futures = {setup: dispatch_manager.submit_group(model_id, "dut:" + setup,
                external_setup, setup) for setup in SETUPS}
            self.setup_futures[model_id] = futures
            dispatch_manager.finish_registration(model_id)
            return [future.result(timeout=12) for future in futures.values()]

        completed = yield dispatch
        assert completed == list(SETUPS)
        # The ordinary model finalizer still scans after its complete download.
        self._queue_central_quality_requests(model_id)
        artifact = write_json(self.run_dir / "models" / model_id / "controlled_result.json",
            {"model_id": model_id, "diagnostic_only": True})
        return {"controlled_result": artifact}, {"controlled_leaf": True}, "controlled local DUTs", "ok"

    def _stage_validate_outputs(self, model_id, row):
        return {}, {"controlled_leaf": True}, "controlled local validation", "ok"

    _stage_hardware_smoke = _stage_validate_outputs

    def _evaluate_central_quality_request(self, model_id, request_path):
        payload = json.loads(request_path.read_text())
        with self.observation_lock:
            self.evaluated.append((threading.get_ident(), model_id, request_path))
            if not payload["ready"]:
                self.partial_reads.append(request_path)
                raise AssertionError("unfinished setup input reached quality evaluation")
            self.active_evaluations += 1
            self.max_active_evaluations = max(self.max_active_evaluations, self.active_evaluations)
            if self.active_evaluations == 2:
                self.two_evaluations_started.set()
        try:
            assert self.release_quality.wait(10), "test-owned evaluation leaf was not released"
            if payload["case_id"] == "b002":
                raise ValueError("controlled bad-case evaluation")
            result = {"status": "completed", "model_id": model_id,
                "setup_id": payload["setup_id"], "case_id": payload["case_id"]}
            self.good_finished.append((model_id, payload["setup_id"], payload["case_id"]))
            return result
        finally:
            with self.observation_lock:
                self.active_evaluations -= 1


@pytest.fixture
def controlled(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("setup callback test must not start a process pool or statistics")

    monkeypatch.setattr(quality_service, "ProcessPoolExecutor", forbidden)
    monkeypatch.setattr(quality_service, "_evaluate_payload", forbidden)
    monkeypatch.setattr(quality_statistics_blocks, "prepare_plan", forbidden)
    instances = []

    class ControlledManagementService:
        def __init__(self, cache_root, *, workers, pause_gate, statistics):
            self.cache_root, self.workers, self.pause_gate = cache_root, workers, pause_gate
            self.statistics = deepcopy(statistics)
            self.constructor_thread = threading.get_ident()
            self.shutdown_calls = []
            instances.append(self)

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    monkeypatch.setattr(quality_service, "ManagementQualityService", ControlledManagementService)
    runner, events = _runner(tmp_path, _ControlledSetupQuality)
    runner.profile_payload.update({
        "workflow_execution": {"setup_queue_mode": "per_setup", "native_release_mode": "global_barrier"},
        "parallel_remote": {"max_parallel_setups": 2},
        "hardware": {"selected_setups": list(SETUPS)},
        "native_producers": {"enabled": False}, "energy": {"enabled": False},
        "run_profiles": [{"id": setup, "enabled": True, "hardware_setup_id": setup,
            "quality_source_only": True, "variants": ["composed"]} for setup in SETUPS]
            + [{"id": "disabled-unsafe", "enabled": False, "quality_source_only": False,
                "variants": ["full"], "hardware_setup_id": "not-selected"}],
        "quality_gate": {"statistics": {"execution_location": "central_management",
            "engine": "optimized_coco_v1", "workers": 6, "max_active_requests": 2}},
    })
    runner.options.max_parallel_setups = 2
    runner.queue_calls, runner.service_observations, runner.callback_observations = [], [], []
    runner.dispatch_manager_ids, runner.callback_threads = set(), []
    runner.release_fast, runner.release_slow, runner.release_quality = (threading.Event() for _ in range(3))
    runner.two_evaluations_started = threading.Event()
    runner.setup_started = {(model, setup): threading.Event() for model in MODELS for setup in SETUPS}
    runner.setup_downloaded = {(model, setup): threading.Event() for model in MODELS for setup in SETUPS}
    runner.setup_futures, runner.evaluated, runner.partial_reads, runner.good_finished = {}, [], [], []
    runner.observation_lock = threading.Lock()
    runner.active_evaluations = runner.max_active_evaluations = 0
    runner.service_instances = instances
    for model in MODELS:
        runner._request(model, SETUPS[1], "b004", ready=False)
    try:
        yield runner, events
    finally:
        runner.release_fast.set()
        runner.release_slow.set()
        runner.release_quality.set()
        runner._shutdown_management_services()


def _start_parent(runner, executor):
    def parent():
        runner.controller_thread = threading.get_ident()
        with bind_process_registry(runner._process_registry):
            return runner._run_prepared_setup_queues(runner.profile_payload["models"])
    return executor.submit(parent)


def test_completed_setup_quality_finishes_while_other_dut_is_pending_without_duplicate_submits(controlled):
    runner, _ = controlled
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="controlled-parent") as controller:
        parent = _start_parent(runner, controller)
        try:
            assert runner.setup_started[(MODELS[0], SETUPS[1])].wait(8)
            runner.release_fast.set()
            assert runner.two_evaluations_started.wait(8)
            _wait_for(lambda: len(runner._central_quality_futures) == 6)
            service, coordinator = runner._central_quality_service, runner._central_quality_coord_executor
            assert len(runner.service_instances) == 1
            assert service.workers == 6 and service.statistics["max_active_requests"] == 2
            assert coordinator._max_workers == 2
            assert runner.max_active_evaluations == 2
            assert not parent.done()
            assert not runner.setup_futures[MODELS[0]][SETUPS[1]].done()
            assert not runner.setup_downloaded[(MODELS[0], SETUPS[1])].is_set()
            assert not runner.partial_reads
            assert all(root == runner._quality_root(model, SETUPS[0])
                       for _, model, root in runner.queue_calls)
            assert all(owner == runner.controller_thread for owner, _, _ in runner.queue_calls)
            assert service.constructor_thread == runner.controller_thread
            runner.release_quality.set()
            _wait_for(lambda: all(future.done() for future in runner._central_quality_futures.values()))
            fast_futures = dict(runner._central_quality_futures)
            assert sum(isinstance(future.exception(), ValueError) for future in fast_futures.values()) == 2
            assert {(model, case) for model, setup, case in runner.good_finished if setup == SETUPS[0]} == {
                (model, case) for model in MODELS for case in ("b001", "b003")}
            assert not parent.done(), "finished statistics must not require the slow DUT barrier"
            runner.release_slow.set()
            assert parent.result(timeout=10) == set(MODELS)
            _wait_for(lambda: len(runner._central_quality_futures) == 8
                and all(future.done() for future in runner._central_quality_futures.values()))
            assert all(runner._central_quality_futures[key] is future for key, future in fast_futures.items())
            assert len(runner.service_instances) == 1
            assert runner._central_quality_service is service
            assert runner._central_quality_coord_executor is coordinator
            assert {service_id for _, service_id, _ in runner.service_observations} == {id(service)}
            assert {coordinator_id for _, _, coordinator_id in runner.service_observations} == {id(coordinator)}
            assert len(runner.dispatch_manager_ids) == 1
            assert runner.callback_threads and all(owner != runner.controller_thread for owner in runner.callback_threads)
            assert Counter(str(path) for _, _, path in runner.evaluated) == Counter(runner._central_quality_futures.keys())
            assert all(sum(root is None and model == expected for _, model, root in runner.queue_calls) == 1
                       for expected in MODELS)
            assert not runner.partial_reads
            assert all(json.loads(_stage_path(runner, model).read_text())["complete"] for model in MODELS)
        finally:
            runner.release_fast.set()
            runner.release_slow.set()
            runner.release_quality.set()
            parent.result(timeout=12)


@pytest.mark.parametrize("mode", ["native", "energy", "non_source", "full_variant", "local"])
def test_other_workflow_modes_keep_model_finalizer_admission(controlled, mode):
    runner, _ = controlled
    if mode in {"native", "energy"}:
        runner.profile_payload["native_producers" if mode == "native" else "energy"]["enabled"] = True
    elif mode == "non_source":
        runner.profile_payload["run_profiles"][0]["quality_source_only"] = False
    elif mode == "full_variant":
        runner.profile_payload["run_profiles"][0]["variants"] = ["composed", "full"]
    elif mode == "local":
        runner.profile_payload["quality_gate"]["statistics"]["execution_location"] = "local"
    runner.release_fast.set()
    runner.release_slow.set()
    runner.release_quality.set()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="controlled-parent") as controller:
        assert _start_parent(runner, controller).result(timeout=12) == set(MODELS)
    assert len(runner.callback_observations) == len(MODELS)
    assert all(callback is None for _, callback in runner.callback_observations)
    assert len(runner.queue_calls) == len(MODELS)
    assert all(root is None for _, _, root in runner.queue_calls)
    assert not runner.partial_reads
    if mode == "local":
        assert not runner.service_instances
        assert not runner._central_quality_futures
