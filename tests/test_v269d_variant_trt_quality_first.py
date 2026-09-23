from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.trt_quality_chain import TensorRTQualityChainError
from tests.test_v269d_trt_quality_chain import _result, _strict_producer, _summary
from tests.test_v269d_trt_quality_first_runtime import (
    _load_runner,
    _namespace,
    _write_quality_producer_set,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, *, packaged: bool = False):
    directory = (
        ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
        if packaged else ROOT / "scripts"
    )
    path = directory / name
    spec = importlib.util.spec_from_file_location(f"test_quality_first_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evalrun(tmp_path: Path, models: tuple[str, ...] = ("resnet50", "yolo26s")) -> Path:
    run = tmp_path / "eval-20260721"
    # These are compatibility tests for the pre-2.69f TensorRT-Full producer
    # hand-off.  Bind the fixture to that historical workflow explicitly;
    # an unversioned run is intentionally treated as a malformed current run
    # by the 2.69f Quality-FIRST authority gate.
    run.mkdir(parents=True, exist_ok=True)
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "workflow_version": "v2.69d-quality-first-engine-reuse",
        "tool_version": "2.69.4",
    }), encoding="utf-8")
    for model in models:
        benchmark_set = run / "models" / model / "benchmark_set"
        (benchmark_set / "b001").mkdir(parents=True)
        (benchmark_set / "benchmark_set.json").write_text("{}", encoding="utf-8")
        (benchmark_set / "b001" / "split_manifest.json").write_text("{}", encoding="utf-8")
    snapshot_sha = "a" * 64
    selection_sha = "b" * 64
    workflow = "v2.69e-smoke-quality-start-snapshot-repair"
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "workflow_version": workflow,
        "current_workflow_version": workflow,
        "tool_version": "2.69.6",
        "current_tool_version": "2.69.6",
        "execution_sessions": [{
            "workflow_version": workflow,
            "tool_version": "2.69.6",
        }],
        "profile_start_snapshot": {
            "snapshot_sha256": snapshot_sha,
            "requested_selection": {"snapshot_sha256": selection_sha},
            "resolved_selection": {"snapshot_sha256": selection_sha},
        },
    }), encoding="utf-8")
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "native_producer_stage.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/native-producer-stage",
        "schema_version": 3,
        "run_id": run.name,
        "workflow_version": workflow,
        "tool_version": "2.69.6",
        "profile_start_snapshot_sha256": snapshot_sha,
        "profile_selection_snapshot_sha256": selection_sha,
        "native_split_quality_first": {"required": True},
    }), encoding="utf-8")
    return run


def _quality_summary(run: Path, models: tuple[str, ...]) -> Path:
    results = [_result(_strict_producer(model)) for model in models]
    path = run / "quality_management" / "central_quality_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_summary(results)), encoding="utf-8")
    return path


def _variant_cfg(summary: Path) -> dict[str, Any]:
    return {
        "backends": ["hailo8"],
        "remotes": {
            "hailo8": {
                "ssh": "nx@hailo8",
                "setup_id": "orin_nx_hailo8_01",
            },
        },
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {"hailo8": ["hailo8", "tensorrt"]},
        },
        "_workflow_context": {"central_quality_summary": str(summary)},
    }


def test_variant_coordinator_builds_one_multimodel_set_and_one_last_owner(
    tmp_path: Path,
) -> None:
    module = _load_script("run_evalrun_native_producer_variants.py")
    models = ("resnet50", "yolo26s")
    run = _evalrun(tmp_path, models)
    cfg = _variant_cfg(_quality_summary(run, models))
    variants = [
        {"id": "resnet", "case_map": {"resnet50": ["b001"]}},
        {"id": "yolo", "case_map": {"yolo26s": ["b001"]}},
    ]

    paths, plan = module._materialize_trt_quality_producer_sets(
        run, cfg, variants,
    )
    assert set(paths) == {"orin_nx_hailo8_01"}
    payload = json.loads(Path(paths["orin_nx_hailo8_01"]).read_text(encoding="utf-8"))
    assert payload["schema"] == "onnx-splitpoint/tensorrt-quality-producer-set"
    assert set(payload["producers_by_model"]) == set(models)
    assert plan["owner_by_setup"] == {"orin_nx_hailo8_01": 1}

    prepared = module._quality_first_variant_plan(cfg, variants, paths, plan)
    first = prepared[0]["full_baselines"]["backends_by_producer"]["hailo8"]
    second = prepared[1]["full_baselines"]["backends_by_producer"]["hailo8"]
    assert first == ["hailo8"]
    assert second == ["hailo8", "tensorrt"]
    assert prepared[0]["trt_quality_producer_sets_by_setup"] == paths
    assert prepared[1]["trt_quality_producer_sets_by_setup"] == paths
    command = module._build_update_cmd(
        run, cfg, prepared[1], refresh_suites=False, timeout_s=60,
    )
    assert "--trt-quality-producer-sets" in command
    forwarded = json.loads(
        command[command.index("--trt-quality-producer-sets") + 1]
    )
    assert forwarded == paths


def test_parent_mapping_must_match_central_summary_exactly(tmp_path: Path) -> None:
    module = _load_script("run_evalrun_native_producer_variants.py")
    models = ("resnet50", "yolo26s")
    run = _evalrun(tmp_path, models)
    summary = _quality_summary(run, models)
    cfg = _variant_cfg(summary)
    variants = [{"id": "all", "case_map": {"resnet50": ["b001"]}}]
    paths, _ = module._materialize_trt_quality_producer_sets(run, cfg, variants)
    drifted = json.loads(Path(next(iter(paths.values()))).read_text(encoding="utf-8"))
    producer = copy.deepcopy(drifted["producers_by_model"]["resnet50"])
    producer["setup_id"] = "other_setup"
    drifted["producers_by_model"]["resnet50"] = producer
    supplied = tmp_path / "drifted.json"
    supplied.write_text(json.dumps(drifted), encoding="utf-8")
    cfg["trt_quality_producer_sets_by_setup"] = {
        "orin_nx_hailo8_01": str(supplied),
    }

    with pytest.raises(TensorRTQualityChainError):
        module._materialize_trt_quality_producer_sets(run, cfg, variants)


def test_duplicate_backend_alias_and_setup_drift_are_rejected(tmp_path: Path) -> None:
    module = _load_script("run_evalrun_native_producer_variants.py")
    with pytest.raises(TensorRTQualityChainError, match="duplicate"):
        module._variant_backend_bindings(
            {"backends": ["hailo10", "hailo10h_to_trt"]}, {"id": "bad"},
        )

    run = _evalrun(tmp_path, ("resnet50",))
    summary = _quality_summary(run, ("resnet50",))
    cfg = _variant_cfg(summary)
    variants = [
        {"id": "one"},
        {
            "id": "two",
            "remotes": {
                "hailo8": {
                    "ssh": "nx@hailo8",
                    "setup_id": "drifted_setup",
                },
            },
        },
    ]
    with pytest.raises(TensorRTQualityChainError, match="changes setup identity"):
        module._materialize_trt_quality_producer_sets(run, cfg, variants)


def test_update_path_refuses_trt_full_before_remote_execution_without_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("update_evalset_native_producers.py")
    run = _evalrun(tmp_path, ("resnet50",))
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> dict[str, Any]:
        calls.append(list(cmd))
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_summarize_host_telemetry", lambda *a, **k: {"rc": 0})
    stage = module._run_native_producers(run, {
        "backends": ["deepx"],
        "remotes": {"deepx": {"ssh": "nx@deepx", "setup_id": "deepx_setup"}},
        "copy_benchmarksets": False,
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {"deepx": ["tensorrt"]},
        },
    }, timeout=10)

    backend = stage["backend_results"][0]
    assert backend["status"] == "failed"
    assert backend["failure_class"] == "upstream_quality_evidence"
    assert backend["failure_reason"] == "upstream_central_quality_binding_missing"
    assert backend["transfer_attempted"] is False
    assert "without central-quality producer set" in backend["error"]
    assert not calls


def test_update_path_stages_one_set_and_forwards_exact_remote_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("update_evalset_native_producers.py")
    run = _evalrun(tmp_path, ("resnet50",))
    producer_set = tmp_path / "producer_set.json"
    producer_set.write_text("{}", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> dict[str, Any]:
        calls.append((list(cmd), dict(kwargs)))
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr(module, "_sync_remote_package_asset_v263", lambda *a, **k: [])
    monkeypatch.setattr(module, "_verify_remote_module_binding_v263", lambda *a, **k: {"rc": 0})
    monkeypatch.setattr(module, "_validate_trt_quality_producer_set", lambda *a, **k: {})
    monkeypatch.setattr(
        module, "_capture_remote_host_telemetry",
        lambda **kwargs: {"name": f"capture_{kwargs['phase']}", "rc": 0},
    )
    monkeypatch.setattr(module, "_summarize_host_telemetry", lambda *a, **k: {"rc": 0})

    stage = module._run_native_producers(run, {
        "backends": ["deepx"],
        "remotes": {"deepx": {"ssh": "nx@deepx", "setup_id": "deepx_setup"}},
        "copy_benchmarksets": False,
        "build_missing_engines": False,
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {"deepx": ["tensorrt"]},
        },
        "trt_quality_producer_sets_by_setup": {"deepx_setup": str(producer_set)},
    }, timeout=10)

    labelled = {
        str(kwargs.get("label")): cmd
        for cmd, kwargs in calls
        if kwargs.get("label")
    }
    full_shell = labelled["full:deepx"][-1]
    expected_remote = (
        f"/home/nx/native_fifo_evalsets/{run.name}/quality_first/"
        "tensorrt_quality_producer_set.json"
    )
    assert f"--trt-quality-producer-json {expected_remote}" in full_shell
    backend = stage["backend_results"][0]
    assert backend["trt_quality_producer_set"]["remote_path"] == expected_remote
    assert backend["native_full_baseline_available"] is True


@pytest.mark.parametrize("packaged", [False, True], ids=["source", "packaged"])
def test_staged_trt_producer_path_satisfies_runtime_role_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packaged: bool,
) -> None:
    module = _load_script("update_evalset_native_producers.py", packaged=packaged)
    runner = _load_runner()
    suite, producer_file, expected = _write_quality_producer_set(tmp_path)
    ns = _namespace(producer_file)
    transport_calls: list[list[str]] = []

    def fake_transport(cmd: list[str], **_kwargs: Any) -> dict[str, Any]:
        transport_calls.append(list(cmd))
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    def no_runtime(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("producer role validation must not launch a runtime or build")

    monkeypatch.setattr(module, "_run", fake_transport)
    monkeypatch.setattr(runner, "_run", no_runtime)
    staged_path, _steps = module._stage_remote_trt_quality_producer_set(
        local_path=producer_file,
        ssh="fixture-only",
        remote_root=ns.root,
        timeout=10,
    )
    ns.trt_quality_producer_json = staged_path
    loaded, status, verified_path = runner._quality_first_trt_producer_identity(
        suite, "resnet50", ns,
    )
    assert status == "quality_first_identity_verified_exact"
    assert loaded == expected
    assert verified_path == producer_file
    assert staged_path == str(producer_file)
    assert any(cmd[0] == "rsync" for cmd in transport_calls)

    # Identical valid bytes at the former dot-directory path must remain invalid.
    wrong_role = Path(ns.root) / ".quality_first" / producer_file.name
    wrong_role.parent.mkdir()
    wrong_role.write_bytes(producer_file.read_bytes())
    ns.trt_quality_producer_json = str(wrong_role)
    loaded, status, rejected_path = runner._quality_first_trt_producer_identity(
        suite, "resnet50", ns,
    )
    assert loaded is None
    assert status == "quality_first_producer_set_role_path_mismatch"
    assert rejected_path == wrong_role


def test_update_cli_preserves_producer_set_mapping() -> None:
    module = _load_script("update_evalset_native_producers.py")
    value = json.dumps({"setup": "/tmp/set.json"})
    parsed = module._parse_trt_quality_producer_sets(value)
    assert parsed == {"setup": "/tmp/set.json"}
