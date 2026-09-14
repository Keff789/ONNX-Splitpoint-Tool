from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.workflow import execution_binding
from onnx_splitpoint_tool.workflow.execution_binding import (
    ExecutionBindingResult,
    _remote_execution_if_requested,
    _scheduler_owned_remote_add_args,
)
from onnx_splitpoint_tool.workflow.setup_local_trt_dispatch import (
    build_setup_local_tensorrt_quality_dispatch,
    validate_setup_local_tensorrt_quality_dispatch,
)


SETUPS = {
    "hailo8": "orin_nx_hailo8_01",
    "hailo10h": "orin_nx_hailo10_01",
    "deepx": "orin_nx_deepx_m1_01",
}


def _run_profiles() -> list[dict[str, object]]:
    return [
        {
            "id": "ort_tensorrt", "type": "same_backend_reference",
            "full": "tensorrt", "stage1": "tensorrt",
            "stage2": "tensorrt",
        },
        {
            "id": "hailo8", "type": "same_backend_reference",
            "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8",
            "hardware_setup_id": SETUPS["hailo8"],
        },
        {
            "id": "hailo10", "type": "same_backend_reference",
            "full": "hailo10", "stage1": "hailo10", "stage2": "hailo10",
            "hardware_setup_id": SETUPS["hailo10h"],
        },
        {
            "id": "deepx_m1_full", "type": "same_backend_reference",
            "full": "deepx_m1", "stage1": "deepx_m1",
            "stage2": "deepx_m1",
            "hardware_setup_id": SETUPS["deepx"],
        },
        {
            "id": "ort_cpu", "type": "onnxruntime", "provider": "cpu",
            "full": "cpu", "stage1": "cpu", "stage2": "cpu",
            "semantic_reference_only": True,
            "execution_location": "central_management",
        },
    ]


def _targets(*, malicious_add_args: bool = False) -> list[dict[str, object]]:
    return [
        {
            "id": SETUPS["hailo8"], "accelerator": "hailo8", "enabled": True,
            "runtime": {
                "enabled": True, "host": "h8", "user": "nx",
                "add_args": "--run-ids evil --quality-only-run-ids evil"
                if malicious_add_args else "",
            },
        },
        {
            "id": SETUPS["hailo10h"], "accelerator": "hailo10h", "enabled": True,
            "runtime": {
                "enabled": True, "host": "h10", "user": "nx",
                "add_args": "--run-id evil" if malicious_add_args else "",
            },
        },
        {
            "id": SETUPS["deepx"], "accelerator": "deepx_m1", "enabled": True,
            "runtime": {
                "enabled": True, "host": "dx", "user": "nx",
                "add_args": "--quality-only-run-ids ort_tensorrt"
                if malicious_add_args else "",
            },
        },
    ]


def _profile(*, malicious_add_args: bool = False) -> dict[str, object]:
    return {
        "run_profiles": _run_profiles(),
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
        "hardware_targets": _targets(
            malicious_add_args=malicious_add_args,
        ),
        "native_producers": {
            "enabled": True,
            "remotes": {
                "hailo8": {"setup_id": SETUPS["hailo8"]},
                "hailo10h": {"setup_id": SETUPS["hailo10h"]},
                "deepx": {"setup_id": SETUPS["deepx"]},
            },
        },
    }


def _profile_with_frozen_hardware() -> dict[str, object]:
    profile = _profile()
    targets = _targets()
    target_sha = hashlib.sha256(json.dumps(
        targets,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")).hexdigest()
    profile["hardware"] = {
        "resolution_frozen_at_start": True,
        "resolved_targets": targets,
        "resolved_targets_sha256": f"sha256:{target_sha}",
    }
    return profile


def test_profile_preflight_replication_has_one_deepx_performance_owner() -> None:
    contract = validate_setup_local_tensorrt_quality_dispatch(_profile())

    assert contract["ok"] is True
    assert contract["performance_owner_producer"] == "deepx"
    assert contract["performance_owner_setup_id"] == SETUPS["deepx"]
    assert contract["remote_cpu_reference_run_ids"] == []
    assert contract["central_cpu_reference_run_ids"] == ["ort_cpu"]
    dispatches = {row["producer"]: row for row in contract["setup_dispatches"]}
    assert set(dispatches) == {"hailo8", "hailo10h", "deepx"}
    assert dispatches["hailo8"]["run_ids"] == ["hailo8", "ort_tensorrt"]
    assert dispatches["hailo10h"]["run_ids"] == ["hailo10", "ort_tensorrt"]
    assert dispatches["deepx"]["run_ids"] == ["deepx_m1_full", "ort_tensorrt"]
    assert dispatches["hailo8"]["quality_only_run_ids"] == ["ort_tensorrt"]
    assert dispatches["hailo10h"]["quality_only_run_ids"] == ["ort_tensorrt"]
    assert dispatches["deepx"]["quality_only_run_ids"] == []
    assert sum(row["performance_claims_emitted"] for row in dispatches.values()) == 1
    assert {
        producer: row["quality_companion_endpoint_id"]
        for producer, row in dispatches.items()
    } == {
        "hailo8": "tensorrt_at_hailo8_full",
        "hailo10h": "tensorrt_at_hailo10h_full",
        "deepx": "tensorrt_at_deepx_m1_full",
    }
    assert all(
        row["quality_companion_required"] is True
        and row["quality_companion_identity"]["setup_id"]
        == row["setup_id"]
        for row in dispatches.values()
    )


def test_normalized_same_backend_reference_is_executable_trt_full_recipe() -> None:
    normalized = BenchmarkGenerationService().build_run_plan(
        acc_cpu=False,
        acc_cuda=False,
        acc_trt=True,
        acc_h8=True,
        acc_h10=True,
        acc_deepx=True,
        hailo8_hw="hailo8",
        hailo10_hw="hailo10",
        hailo_preset="Custom",
        hailo_custom_full=True,
        hailo_custom_composed=False,
        hailo_custom_part1=False,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=False,
        matrix_deepx_to_trt=False,
    ).bench_plan_runs

    contract = build_setup_local_tensorrt_quality_dispatch(
        _profile(),
        hardware_targets=_targets(),
        plan_rows=normalized,
    )

    assert contract["ok"] is True
    assert contract["tensorrt_run_id"] == "ort_tensorrt"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda rows: rows.append(dict(rows[0])),
            "setup_local_tensorrt_recipe_count_not_one",
        ),
        (
            lambda rows: rows[0].update({
                "full": "cpu", "stage1": "cpu", "stage2": "cpu",
            }),
            "setup_local_tensorrt_recipe_not_executable_trt_full",
        ),
    ],
)
def test_preflight_rejects_duplicate_or_cpu_disguised_trt_recipe(
    mutation, expected_error: str,
) -> None:
    profile = _profile()
    rows = list(profile["run_profiles"])
    mutation(rows)
    profile["run_profiles"] = rows

    contract = build_setup_local_tensorrt_quality_dispatch(
        profile,
        hardware_targets=_targets(),
        plan_rows=rows,
    )

    assert contract["ok"] is False
    assert expected_error in contract["errors"]


def test_preflight_rejects_generic_native_setup_id_drift() -> None:
    profile = _profile()
    profile["native_producers"]["remotes"]["hailo10h"]["setup_id"] = "other_h10"

    contract = build_setup_local_tensorrt_quality_dispatch(
        profile,
        hardware_targets=_targets(),
        plan_rows=_run_profiles(),
    )

    assert contract["ok"] is False
    assert "generic_native_setup_id_mismatch:hailo10h" in contract["errors"]


def test_scheduler_owned_args_cannot_be_overridden_by_user_add_args() -> None:
    bound = _scheduler_owned_remote_add_args(
        "--run-id evil --run-ids evil --quality-only-run-ids evil "
        "--quality-evidence-setup-id forged --timeout 17",
        run_ids=["hailo8", "ort_tensorrt"],
        quality_only_run_ids=["ort_tensorrt"],
    )
    tokens = shlex.split(bound)

    assert tokens == [
        "--timeout", "17", "--run-ids", "hailo8,ort_tensorrt",
        "--quality-only-run-ids", "ort_tensorrt",
    ]


def test_execution_dispatches_three_setup_local_trt_producers_before_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(malicious_add_args=True)
    targets = _targets(malicious_add_args=True)
    monkeypatch.setattr(execution_binding, "matrix_for_runtime", lambda _p: targets)
    calls: list[dict[str, object]] = []

    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        return ExecutionBindingResult(
            artifacts={},
            metrics={"remote_result_files_copied": 1},
            status="ok",
            message="ok",
        )

    monkeypatch.setattr(execution_binding, "_run_remote_dispatch_once", fake_dispatch)
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "benchmark_plan.json").write_text(
        __import__("json").dumps({"runs": _run_profiles()}),
        encoding="utf-8",
    )
    (suite / "benchmark_set.json").write_text("{}", encoding="utf-8")
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    result = _remote_execution_if_requested(
        run_root=tmp_path,
        model_id="resnet50",
        options=SimpleNamespace(
            no_remote=False,
            benchmark_execution_backend="remote",
            parallel_remote_setups=False,
        ),
        profile_payload=profile,
        suite_dir=suite,
        benchmark_set_json=suite / "benchmark_set.json",
        result_dir=result_dir,
        contains_hailo=True,
        gates={},
        log=None,
    )

    assert result is not None and result.status == "ok"
    assert len(calls) == 3
    by_setup = {
        str(call["target_id"]): shlex.split(
            str(call["runtime_override"]["add_args"])
        )
        for call in calls
    }
    assert by_setup[SETUPS["hailo8"]][-4:] == [
        "--run-ids", "hailo8,ort_tensorrt",
        "--quality-only-run-ids", "ort_tensorrt",
    ]
    assert by_setup[SETUPS["hailo10h"]][-4:] == [
        "--run-ids", "hailo10,ort_tensorrt",
        "--quality-only-run-ids", "ort_tensorrt",
    ]
    assert by_setup[SETUPS["deepx"]][-2:] == [
        "--run-ids", "deepx_m1_full,ort_tensorrt",
    ]
    assert not any("ort_cpu" in token for tokens in by_setup.values() for token in tokens)
    assert (result_dir / "setup_local_tensorrt_dispatch_preflight.json").is_file()


def test_three_models_three_setups_bind_nine_concrete_companions_before_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regress the overnight 3x3 Standard/Quality dispatch shape."""

    profile = _profile()
    targets = _targets()
    monkeypatch.setattr(
        execution_binding, "matrix_for_runtime", lambda _p: targets,
    )
    calls: list[dict[str, object]] = []

    def fake_dispatch(**kwargs):
        calls.append(kwargs)
        return ExecutionBindingResult(
            artifacts={},
            metrics={"remote_result_files_copied": 1},
            status="ok",
            message="ok",
        )

    monkeypatch.setattr(
        execution_binding, "_run_remote_dispatch_once", fake_dispatch,
    )
    models = ("resnet50", "yolo26s", "yolov7_paper")
    for model_id in models:
        model_root = tmp_path / model_id
        suite = model_root / "suite"
        result_dir = model_root / "results"
        suite.mkdir(parents=True)
        result_dir.mkdir()
        (suite / "benchmark_plan.json").write_text(
            __import__("json").dumps({"runs": _run_profiles()}),
            encoding="utf-8",
        )
        (suite / "benchmark_set.json").write_text("{}", encoding="utf-8")

        result = _remote_execution_if_requested(
            run_root=tmp_path,
            model_id=model_id,
            options=SimpleNamespace(
                no_remote=False,
                benchmark_execution_backend="remote",
                parallel_remote_setups=False,
            ),
            profile_payload=profile,
            suite_dir=suite,
            benchmark_set_json=suite / "benchmark_set.json",
            result_dir=result_dir,
            contains_hailo=True,
            gates={},
            log=None,
        )
        assert result is not None and result.status == "ok"

    assert len(calls) == 9
    expected_ids = {
        SETUPS["hailo8"]: "tensorrt_at_hailo8_full",
        SETUPS["hailo10h"]: "tensorrt_at_hailo10h_full",
        SETUPS["deepx"]: "tensorrt_at_deepx_m1_full",
    }
    observed_models: set[str] = set()
    per_setup_count = {setup_id: 0 for setup_id in expected_ids}
    for call in calls:
        gates = dict(call["gates"])
        setup_id = str(call["target_id"])
        observed_models.add(str(call["model_id"]))
        per_setup_count[setup_id] += 1
        assert gates["quality_companion_required"] is True
        assert (
            gates["quality_companion_endpoint_id"]
            == expected_ids[setup_id]
        )
    assert observed_models == set(models)
    assert per_setup_count == {setup_id: 3 for setup_id in expected_ids}


def test_missing_companion_identity_fails_before_remote_service_or_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    result_dir = tmp_path / "results"
    suite.mkdir()
    result_dir.mkdir()
    benchmark_set = suite / "benchmark_set.json"
    benchmark_set.write_text("{}", encoding="utf-8")
    host_resolution_called = False

    def forbidden_host_resolution(_payload):
        nonlocal host_resolution_called
        host_resolution_called = True
        raise AssertionError("SSH host construction must not start")

    monkeypatch.setattr(
        execution_binding, "_make_ssh_host_config",
        forbidden_host_resolution,
    )

    result = execution_binding._run_remote_dispatch_once(
        run_root=tmp_path,
        model_id="resnet50",
        options=SimpleNamespace(),
        profile_payload={},
        suite_dir=suite,
        benchmark_set_json=benchmark_set,
        result_dir=result_dir,
        gates={
            "quality_companion_required": True,
            "quality_companion_endpoint_id": "",
        },
        log=None,
        runtime_override={"host": "must-not-be-resolved"},
        target_id=SETUPS["hailo8"],
    )

    assert result.status == "failed_to_dispatch"
    assert result.metrics["remote_dispatched"] is False
    assert result.metrics["pre_remote_mutation"] is True
    assert result.metrics["failure_kind"] == (
        "setup_local_tensorrt_quality_identity_missing"
    )
    assert host_resolution_called is False


def test_ambiguous_setup_identity_fails_before_remote_worker_or_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _run_profiles()
    targets = [*_targets(), dict(_targets()[0])]
    profile = _profile()
    monkeypatch.setattr(
        execution_binding, "matrix_for_runtime", lambda _p: targets,
    )
    dispatch_called = False

    def forbidden_dispatch(**_kwargs):
        nonlocal dispatch_called
        dispatch_called = True
        raise AssertionError("remote worker/SSH must not start")

    monkeypatch.setattr(
        execution_binding, "_run_remote_dispatch_once", forbidden_dispatch,
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "benchmark_plan.json").write_text(
        json.dumps({"runs": rows}), encoding="utf-8",
    )
    (suite / "benchmark_set.json").write_text("{}", encoding="utf-8")
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    with pytest.raises(
        RuntimeError,
        match="setup_local_tensorrt_dispatch_preflight_failed",
    ):
        _remote_execution_if_requested(
            run_root=tmp_path,
            model_id="resnet50",
            options=SimpleNamespace(
                no_remote=False,
                benchmark_execution_backend="remote",
                parallel_remote_setups=False,
            ),
            profile_payload=profile,
            suite_dir=suite,
            benchmark_set_json=suite / "benchmark_set.json",
            result_dir=result_dir,
            contains_hailo=True,
            gates={},
            log=None,
        )
    assert dispatch_called is False
    preflight = json.loads(
        (result_dir / "setup_local_tensorrt_dispatch_preflight.json")
        .read_text(encoding="utf-8")
    )
    assert "setup_local_target_count_not_one:hailo8" in preflight[
        "errors"
    ]


def test_execution_preflight_blocks_cpu_disguised_trt_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _run_profiles()
    rows[0].update({"full": "cpu", "stage1": "cpu", "stage2": "cpu"})
    profile = _profile()
    profile["run_profiles"] = rows
    targets = _targets()
    monkeypatch.setattr(execution_binding, "matrix_for_runtime", lambda _p: targets)
    called = False

    def forbidden_dispatch(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("remote dispatch must not start")

    monkeypatch.setattr(
        execution_binding, "_run_remote_dispatch_once", forbidden_dispatch,
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "benchmark_plan.json").write_text(
        __import__("json").dumps({"runs": rows}), encoding="utf-8",
    )
    (suite / "benchmark_set.json").write_text("{}", encoding="utf-8")
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    with pytest.raises(
        RuntimeError,
        match="setup_local_tensorrt_dispatch_preflight_failed",
    ):
        _remote_execution_if_requested(
            run_root=tmp_path,
            model_id="resnet50",
            options=SimpleNamespace(
                no_remote=False,
                benchmark_execution_backend="remote",
                parallel_remote_setups=False,
            ),
            profile_payload=profile,
            suite_dir=suite,
            benchmark_set_json=suite / "benchmark_set.json",
            result_dir=result_dir,
            contains_hailo=True,
            gates={},
            log=None,
        )
    assert called is False


def test_execution_plan_preview_matches_setup_local_dispatch_roles() -> None:
    profile = _profile_with_frozen_hardware()
    profile.update({
        "model_suite": {"primary": [{"id": "resnet50", "enabled": True}]},
        "selection_policy": {"max_accepted_cases_per_model": 1},
        "execution_preset": {
            "id": "smoke",
            "overrides": {"native_enabled": True, "energy_enabled": False},
            "snapshot": {
                "defaults": {"native_enabled": False, "energy_enabled": False},
            },
        },
    })

    plan = build_effective_execution_plan(profile)

    assert plan["setup_groups"] == {
        "hailo8_setup": ["hailo8", "ort_tensorrt"],
        "hailo10h_setup": ["hailo10", "ort_tensorrt"],
        "deepx_setup": ["deepx_m1_full", "ort_tensorrt"],
    }
    assert plan["tensorrt_performance_owner_group"] == "deepx_setup"
    assert set(plan["setup_local_tensorrt_quality_only_groups"]) == {
        "hailo8_setup", "hailo10h_setup",
    }
    assert plan["setup_local_tensorrt_quality_companions"] is True
    assert plan["quality_canary_enabled"] is False
    assert (
        plan["setup_local_tensorrt_quality_companion_identity_ready"]
        is True
    )
    assert {
        row["logical_setup_group"]: row[
            "quality_companion_endpoint_id"
        ]
        for row in plan[
            "setup_local_tensorrt_quality_companion_identities"
        ]
    } == {
        "hailo8_setup": "tensorrt_at_hailo8_full",
        "hailo10h_setup": "tensorrt_at_hailo10h_full",
        "deepx_setup": "tensorrt_at_deepx_m1_full",
    }
    contract = plan[
        "setup_local_tensorrt_quality_companion_contract"
    ]
    assert contract["status"] == "ready"
    assert contract["identity_authority"] == "effective_execution_plan"
    assert contract["identity_count"] == 3
    assert contract["setup_ids"] == [
        SETUPS["hailo8"], SETUPS["hailo10h"], SETUPS["deepx"],
    ]
    assert {
        row["setup_id"]: row["id"]
        for row in contract["identities"]
    } == {
        SETUPS["hailo8"]: "tensorrt_at_hailo8_full",
        SETUPS["hailo10h"]: "tensorrt_at_hailo10h_full",
        SETUPS["deepx"]: "tensorrt_at_deepx_m1_full",
    }
    assert plan["remote_run_invocations_per_model"] == 3


def test_generated_suite_stops_quality_only_trt_before_performance_branch() -> None:
    source = Path(
        "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")
    compile(source, "benchmark_suite.py.txt", "exec")
    quality_stop = source.index(
        "quality-only dispatch complete; "
    )
    performance_branch = source.index(
        'if rtype in {"onnxruntime", "ort"}:', quality_stop,
    )

    assert "--quality-only-run-ids" in source
    assert quality_stop < performance_branch
    assert "performance execution intentionally skipped" in source
