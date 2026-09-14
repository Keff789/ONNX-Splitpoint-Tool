from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.machinery import SourceFileLoader
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest
import yaml

from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationOrchestrationService,
    BenchmarkGenerationService,
)
from onnx_splitpoint_tool.cache_verify_policy import (
    CACHE_VERIFY_ONLY,
    CacheVerifyPolicyError,
    attest_cache_verify_benchmark_sets,
    bind_artifact_policy,
    cache_verify_contract_view,
    compiler_dispatch_forbidden,
    validate_cache_verify_only_profile,
)
from onnx_splitpoint_tool.deepx import compiler as deepx_compiler
from onnx_splitpoint_tool.deepx import env_status as deepx_env_status
from onnx_splitpoint_tool.gui import benchmark_workflow
from onnx_splitpoint_tool.native_execution_contract import (
    resolve_native_execution_contract,
)
from onnx_splitpoint_tool.native_split_quality import (
    seal_native_split_quality_binding,
    validate_native_split_quality_binding,
)
from onnx_splitpoint_tool.runners import native_split_quality_runtime
from onnx_splitpoint_tool.runners.backends import (
    hailo_backend as native_hailo_backend,
)
from onnx_splitpoint_tool.run_modes import (
    apply_run_mode,
    default_run_modes_config,
)
from onnx_splitpoint_tool.trt_quality_chain import TensorRTQualityChainError
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _cache_verify_candidate_scope,
    _native_full_requested,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from tests.test_v269f_native_split_receipt_validation import _fixture_payload


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/cache_verify_resnet50_b052_hailo8.yaml"
RESOURCE_PROFILE = (
    ROOT
    / "onnx_splitpoint_tool/resources/evaluation_profiles"
    / PROFILE.name
)


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v2757_cache_verify_{path.stem}_{id(path)}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_template(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *, stub_onnx_runtime: bool = False,
):
    path = ROOT / "onnx_splitpoint_tool/resources/templates" / name
    module_name = f"v2757_cache_verify_template_{path.stem}_{id(path)}"
    if stub_onnx_runtime:
        fake_onnx = ModuleType("onnx")
        fake_ort = ModuleType("onnxruntime")

        class FakeSessionOptions:
            pass

        fake_ort.SessionOptions = FakeSessionOptions  # type: ignore[attr-defined]
        fake_ort.InferenceSession = (  # type: ignore[attr-defined]
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("ORT InferenceSession constructor reached")
            )
        )
        fake_ort.get_available_providers = (  # type: ignore[attr-defined]
            lambda: ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        )
        monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    loader = SourceFileLoader(module_name, str(path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    loader.exec_module(module)
    return module


def _source_profile() -> dict[str, Any]:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _resolved_profile(
    *, config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved, _audit = apply_run_mode(
        _source_profile(),
        config=config or default_run_modes_config(),
        follow_tool_config=True,
    )
    return resolved


def test_cache_verify_profile_and_packaged_resource_are_byte_identical() -> None:
    assert PROFILE.is_file()
    assert RESOURCE_PROFILE.is_file()
    assert PROFILE.read_bytes() == RESOURCE_PROFILE.read_bytes()


def test_hostile_tool_config_cannot_expand_canary_or_enable_build_paths() -> None:
    config = default_run_modes_config()
    smoke = config["modes"]["smoke"]
    smoke["defaults"].update({
        "native_enabled": True,
        "energy_enabled": True,
    })
    smoke["runtime"].update({
        "execution_mode": "generate_and_run",
        "skip_runtime_benchmarks": False,
    })
    smoke["runtime"]["native"].update({
        "backends": ["hailo8", "hailo10h", "deepx"],
        "case_policy": "all_accepted",
        "frames": 9999,
        "warmup": 999,
        "repetitions": 9,
        "full_baselines": True,
        "build_missing_engines": True,
    })
    smoke["build"]["hailo"].update({
        "mode": "reuse_and_build_missing",
        "force_build": True,
        "build_full": True,
        "build_part1": True,
        "build_part2": True,
        "preset": "hostile",
        "optimization_level": 2,
        "calibration_items": 500,
        "calibration_batch_size": 64,
        "calibration_storage": "memmap",
        "calibration_memory_cap_mb": 2048,
        "cache_integrity": "strict",
    })
    smoke["build"]["deepx"].update({
        "mode": "reuse_and_build_missing",
        "force_build": True,
    })
    smoke["build"]["scheduler"]["prefetch_deepx_full"] = True

    resolved = _resolved_profile(config=config)
    attestation = validate_cache_verify_only_profile(resolved)
    actual = cache_verify_contract_view(resolved)
    expected = resolved["execution_guard"]["expected_plan"]

    assert attestation["status"] == "verified"
    assert {key: actual[key] for key in expected} == expected
    assert actual["models"] == ["resnet50"]
    assert actual["native_backends"] == ["hailo8"]
    assert actual["native_case_map"] == {"resnet50": ["b052"]}
    assert actual["native_full_backends"] == []
    assert actual["generic_runtime_enabled"] is False
    assert actual["native_energy_enabled"] is False
    assert actual["hailo_build_mode"] == CACHE_VERIFY_ONLY
    assert actual["hailo_force_build"] is False
    assert actual["hailo_build_full"] is False
    assert actual["hailo_build_part1"] is True
    assert actual["hailo_build_part2"] is False
    assert actual["run_mode"] == "smoke"
    assert actual["calibration_items"] == {
        "classification": 8,
        "detection": 8,
    }
    assert actual["hailo_preset"] == "smoke"
    assert actual["hailo_optimization_level"] == 0
    assert actual["hailo_calib_count"] == 8
    assert actual["hailo_calib_batch_size"] == 8
    assert actual["hailo_calibration_storage"] == "memory"
    assert actual["hailo_calibration_memory_cap_mb"] == 256
    assert actual["hailo_cache_integrity"] == "relaxed"
    assert actual["deepx_build_mode"] == CACHE_VERIFY_ONLY
    assert actual["deepx_force_build"] is False
    assert actual["native_build_missing_engines"] is False
    assert actual["native_force_rebuild_variants"] == []
    assert actual["deepx_prefetch_enabled"] is False


def test_loaded_start_snapshot_and_effective_plan_freeze_attestation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "run_modes.yaml"),
    )
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_HARDWARE_SETUPS_FILE",
        str(tmp_path / "hardware_setups.yaml"),
    )
    loaded = load_evaluation_profile(PROFILE)
    assert loaded is not None
    snapshot = loaded.start_snapshot
    attestation = snapshot["cache_verify_attestation"]
    plan = snapshot["effective_execution_plan"]

    assert snapshot["consistency"] == {"status": "ok", "mismatches": []}
    assert attestation["status"] == "verified"
    assert attestation["compiler_dispatch_allowed"] is False
    # The actual attestation also records the mandatory no-build safety axes.
    assert {
        key: attestation["actual_plan"][key]
        for key in attestation["expected_plan"]
    } == attestation["expected_plan"]
    assert str(attestation["attestation_sha256"]).startswith("sha256:")
    assert plan["artifact_policy"] == CACHE_VERIFY_ONLY
    assert plan["compiler_dispatch_allowed"] is False
    assert plan["cache_verify_expected_plan"] == attestation["expected_plan"]
    assert plan["cache_verify_actual_plan"] == attestation["actual_plan"]
    assert plan["generic_runtime_enabled"] is False
    assert plan["generic_rows_total"] == 0
    assert plan["remote_run_invocations_total"] == 0
    assert plan["cold_suite_uploads_total"] == 0
    assert not any(
        warning.get("id") == "ranking_candidate_shortfall"
        for warning in plan["warnings"]
    )


def test_cache_verify_clamps_ranked_pool_to_exact_attested_case() -> None:
    profile = _resolved_profile()
    candidate_plan = {
        "selected_candidates": [{"split_index": 52}],
    }
    guard, cases, ranked, pool, requested = _cache_verify_candidate_scope(
        profile_payload=profile,
        model_id="resnet50",
        candidate_plan=candidate_plan,
        ranked_candidates=[52],
        candidate_search_pool=[52, 39, 38, 37],
        requested=4,
    )

    assert guard["mode"] == CACHE_VERIFY_ONLY
    assert cases == ["b052"]
    assert ranked == [52]
    assert pool == [52]
    assert requested == 1


def test_cache_verify_rejects_candidate_handoff_drift() -> None:
    with pytest.raises(CacheVerifyPolicyError, match="exact-case handoff"):
        _cache_verify_candidate_scope(
            profile_payload=_resolved_profile(),
            model_id="resnet50",
            candidate_plan={
                "selected_candidates": [
                    {"split_index": 52}, {"split_index": 39},
                ],
            },
            ranked_candidates=[52],
            candidate_search_pool=[52, 39],
            requested=2,
        )


def test_disabled_native_full_mapping_is_not_truthy() -> None:
    assert _native_full_requested({
        "enabled": True,
        "full_baselines": {"enabled": False, "backends": []},
    }) is False
    assert _native_full_requested({
        "enabled": True,
        "full_baselines": {"enabled": True, "backends": ["hailo8"]},
    }) is True


def test_real_run_plan_cannot_reactivate_full_under_cache_verify() -> None:
    plan = BenchmarkGenerationService().build_run_plan(
        acc_cpu=False,
        acc_cuda=False,
        acc_trt=True,
        acc_h8=True,
        acc_h10=False,
        acc_deepx=False,
        hailo8_hw="hailo8",
        hailo10_hw="",
        validation_reference_mode="auto",
        hailo_preset="Custom",
        hailo_custom_full=False,
        hailo_custom_composed=True,
        hailo_custom_part1=True,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=True,
        matrix_deepx_to_trt=False,
        matrix_trt_to_deepx=False,
        full_hef_policy="end",
        cache_verify_only=True,
        cache_verify_hailo_variants=["part1", "composed"],
    )

    assert plan.hef_full is False
    assert plan.hef_part1 is True
    assert plan.hef_part2 is False
    assert plan.matrix_variants == ["part1", "composed"]
    hailo_runs = [
        row
        for row in plan.bench_plan_runs
        if row.get("type") == "hailo"
        or any(
            isinstance(row.get(stage), dict)
            and row[stage].get("type") == "hailo"
            for stage in ("stage1", "stage2")
        )
    ]
    assert [row["id"] for row in hailo_runs] == ["hailo8_to_trt"]
    assert all(row.get("variants") == ["part1", "composed"] for row in hailo_runs)


def test_cache_verify_cannot_promote_prepared_or_yolo_full_baseline() -> None:
    service = BenchmarkGenerationOrchestrationService()
    cfg = SimpleNamespace(
        hailo_cache_only=True,
        hef_full=False,
        full_hef_policy="end",
    )
    assert service._materialize_prepared_full_hailo_baseline(
        cfg,
        log=lambda *_args, **_kwargs: None,
        suite_hailo_hefs={},
    ) is False
    assert service._ensure_yolo26_full_hailo_baseline_plan(
        cfg,
        log=lambda *_args, **_kwargs: None,
    ) is cfg
    assert service._force_yolo26_suite_full_baseline_if_needed(
        cfg,
        log=lambda *_args, **_kwargs: None,
    ) is cfg
    assert service._yolo26_should_build_full_first(cfg) is False


def _mutate_model(profile: dict[str, Any]) -> None:
    profile["model_suite"]["primary"].append({
        "id": "yolo26s", "task": "detection", "enabled": True,
    })


def _mutate_backend(profile: dict[str, Any]) -> None:
    profile["native_producers"]["backends"].append("deepx")


def _mutate_case(profile: dict[str, Any]) -> None:
    profile["native_producers"]["variants"][0]["case_map"] = {
        "resnet50": ["b053"],
    }


def _mutate_full(profile: dict[str, Any]) -> None:
    profile["native_producers"]["full_baselines"] = {
        "enabled": True,
        "backends": ["tensorrt"],
    }


def _mutate_energy(profile: dict[str, Any]) -> None:
    profile["native_producers"]["energy"].update({
        "enabled": True,
        "mode": "measure",
    })


def _mutate_frames(profile: dict[str, Any]) -> None:
    profile["native_producers"]["frames"] = 11


def _mutate_setup(profile: dict[str, Any]) -> None:
    profile["hardware"]["selected_setups"] = ["other_hailo8_setup"]


def _mutate_force(profile: dict[str, Any]) -> None:
    profile["hailo_build"]["force_build"] = True


@pytest.mark.parametrize(
    "mutator",
    (
        _mutate_model,
        _mutate_backend,
        _mutate_case,
        _mutate_full,
        _mutate_energy,
        _mutate_frames,
        _mutate_setup,
        _mutate_force,
    ),
    ids=("model", "backend", "case", "full", "energy", "frames", "setup", "force"),
)
def test_actual_matrix_or_safety_mutation_is_rejected(
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    resolved = _resolved_profile()
    mutator(resolved)
    with pytest.raises(
        CacheVerifyPolicyError,
        match="start blocked before run creation",
    ):
        validate_cache_verify_only_profile(resolved)


def test_expected_matrix_mutation_is_rejected() -> None:
    resolved = _resolved_profile()
    resolved["execution_guard"]["expected_plan"]["native_frames"] = 999
    with pytest.raises(
        CacheVerifyPolicyError,
        match=r"native_frames: expected=999, actual=10",
    ):
        validate_cache_verify_only_profile(resolved)


@pytest.fixture
def cache_verify_physical_suite(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], Path, Path]:
    benchmark_set = tmp_path / "models/resnet50/benchmark_set"
    case_dir = benchmark_set / "b052"
    artifact_dir = case_dir / "hailo/hailo8/part1"
    artifact_dir.mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"id": "b052"}]}) + "\n",
        encoding="utf-8",
    )
    (case_dir / "split_manifest.json").write_text(
        json.dumps({"case_id": "b052", "boundary": 52}) + "\n",
        encoding="utf-8",
    )
    hef = artifact_dir / "resnet50_b052_part1.hef"
    hef.write_bytes(b"verified-hailo8-part1-hef")
    receipt = artifact_dir / "hailo_hef_build_receipt.json"
    preprocessing_contract, preprocessing_sha = (
        hailo_backend.resolve_image_preprocessing_contract(
            task="classification",
            target_hw=(224, 224),
        )
    )
    calibration_identity = "manifest:" + "c" * 64
    prepared_calibration_sha = hashlib.sha256(
        json.dumps(
            {
                "calibration_identity": calibration_identity,
                "preprocessing_contract_sha256": preprocessing_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    compiler_sha = "d" * 64
    cache_payload = {
        "schema": "onnx-splitpoint/hailo-hef-cache-key-v3",
        "model_sha256": compiler_sha,
        "activation_part1_sha256": "",
        "hw_arch": "hailo8",
        "hailo_sdk_version": "hailo-dataflow-compiler:3.33.1",
        "optimization_level": 1,
        "calibration_identity": calibration_identity,
        "prepared_calibration_identity_sha256": prepared_calibration_sha,
        "calibration_count": 500,
        "requested_calibration_count": 500,
        "calibration_storage": "memmap",
        "calibration_memory_cap_bytes": 268435456,
        "calibration_batch_size": 8,
        "extra_model_script": "",
        "start_nodes": [],
        "end_nodes": [],
        "integrity": "relaxed",
        "preprocessing_contract": preprocessing_contract,
        "preprocessing_contract_sha256": preprocessing_sha,
        "net_name": "resnet50_part1_b52",
        "net_input_shapes": [1, 3, 224, 224],
        "disable_rt_metadata_extraction": True,
    }
    cache_key = hashlib.sha256(
        json.dumps(
            cache_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    receipt.write_text(
        json.dumps({
            "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
            "source_onnx_sha256": "e" * 64,
            "compiler_onnx_sha256": compiler_sha,
            "compiler_onnx_filename": "resnet50_part1_b52_hailo_fixed.onnx",
            "hef_sha256": hashlib.sha256(hef.read_bytes()).hexdigest(),
            "hef_size_bytes": hef.stat().st_size,
            "hw_arch": "hailo8",
            "net_name": "resnet50_part1_b52",
            "hailo_sdk_version": "hailo-dataflow-compiler:3.33.1",
            "calibration_identity": calibration_identity,
            "prepared_calibration_identity_sha256": prepared_calibration_sha,
            "calibration_count": 500,
            "requested_calibration_count": 500,
            "calibration_storage": "memmap",
            "calibration_memory_cap_bytes": 268435456,
            "preprocessing_contract": preprocessing_contract,
            "preprocessing_contract_sha256": preprocessing_sha,
            "cache_key": cache_key,
            "cache_payload": cache_payload,
        }) + "\n",
        encoding="utf-8",
    )
    return (
        _resolved_profile(),
        {"resnet50": benchmark_set},
        hef,
        receipt,
    )


def test_physical_benchmarkset_attestation_accepts_exact_receipt_bound_hef(
    cache_verify_physical_suite: tuple[
        dict[str, Any], dict[str, Path], Path, Path,
    ],
) -> None:
    profile, benchmark_sets, hef, _receipt = cache_verify_physical_suite
    attestation = attest_cache_verify_benchmark_sets(profile, benchmark_sets)

    assert attestation["status"] == "verified"
    assert attestation["compiler_dispatch_allowed"] is False
    assert str(attestation["attestation_sha256"]).startswith("sha256:")
    observed = attestation["observed"]["resnet50"]
    assert observed["contract_cases"] == ["b052"]
    assert observed["directory_cases"] == ["b052"]
    assert observed["artifacts"] == {
        "b052:hailo8": [
            str(hef.relative_to(benchmark_sets["resnet50"])),
        ],
    }
    receipt_evidence = observed["artifact_receipt_evidence"][
        "b052:hailo8"
    ]
    assert receipt_evidence["rejected"] == []
    assert receipt_evidence["verified"] == [{
        "status": "receipt_bound",
        "hef_path": str(hef.relative_to(benchmark_sets["resnet50"])),
        "hef_sha256": hashlib.sha256(hef.read_bytes()).hexdigest(),
        "hef_size_bytes": hef.stat().st_size,
        "receipt_path": str(
            _receipt.relative_to(benchmark_sets["resnet50"])
        ),
        "receipt_schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "receipt_hw_arch": "hailo8",
        "receipt_cache_key": json.loads(
            _receipt.read_text(encoding="utf-8")
        )["cache_key"],
        "receipt_net_name": "resnet50_part1_b52",
        "expected_net_name": "resnet50_part1_b52",
        "receipt_sdk_version": "hailo-dataflow-compiler:3.33.1",
        "expected_backend": "hailo8",
    }]


def test_physical_benchmarkset_attestation_rejects_extra_case_before_dispatch(
    cache_verify_physical_suite: tuple[
        dict[str, Any], dict[str, Path], Path, Path,
    ],
) -> None:
    profile, benchmark_sets, _hef, _receipt = cache_verify_physical_suite
    benchmark_set = benchmark_sets["resnet50"]
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"id": "b052"}, {"id": "b053"}]}) + "\n",
        encoding="utf-8",
    )
    extra = benchmark_set / "b053"
    extra.mkdir()
    (extra / "split_manifest.json").write_text(
        json.dumps({"case_id": "b053", "boundary": 53}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        CacheVerifyPolicyError,
        match=r"contract_cases expected=\['b052'\].*b053",
    ) as raised:
        attest_cache_verify_benchmark_sets(profile, benchmark_sets)
    assert "Native pre-dispatch attestation failed" in str(raised.value)


@pytest.mark.parametrize("missing", ("receipt", "hef"))
def test_physical_benchmarkset_artifact_miss_is_cache_miss_blocked(
    missing: str,
    cache_verify_physical_suite: tuple[
        dict[str, Any], dict[str, Path], Path, Path,
    ],
) -> None:
    profile, benchmark_sets, hef, receipt = cache_verify_physical_suite
    (receipt if missing == "receipt" else hef).unlink()

    with pytest.raises(
        CacheVerifyPolicyError,
        match=r"cache_miss_blocked\[hailo_dfc\]",
    ) as raised:
        attest_cache_verify_benchmark_sets(profile, benchmark_sets)
    text = str(raised.value)
    assert "Native pre-dispatch attestation failed" in text
    assert "no receipt-bound Part1 HEF for resnet50/b052/hailo8" in text


@pytest.mark.parametrize(
    ("tamper", "expected_reason"),
    (
        ("hef", "receipt_hef_sha256_mismatch"),
        ("receipt", "receipt_json_not_object"),
        ("hash", "receipt_hef_sha256_mismatch"),
        ("size", "receipt_hef_size_mismatch"),
        ("backend", "receipt_hw_arch_mismatch"),
        ("cache_payload", "receipt_semantic_contract_invalid"),
        ("case_identity", "receipt_net_name_mismatch"),
    ),
)
def test_physical_hailo_receipt_tampering_is_cache_miss_blocked(
    tamper: str,
    expected_reason: str,
    cache_verify_physical_suite: tuple[
        dict[str, Any], dict[str, Path], Path, Path,
    ],
) -> None:
    profile, benchmark_sets, hef, receipt = cache_verify_physical_suite
    if tamper == "hef":
        original = hef.read_bytes()
        hef.write_bytes(bytes(value ^ 0xFF for value in original))
    elif tamper == "receipt":
        receipt.write_text("[]\n", encoding="utf-8")
    else:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if tamper == "hash":
            payload["hef_sha256"] = "0" * 64
        elif tamper == "size":
            payload["hef_size_bytes"] = int(payload["hef_size_bytes"]) + 1
        elif tamper == "backend":
            payload["hw_arch"] = "hailo10h"
        elif tamper == "cache_payload":
            payload["cache_payload"]["hailo_sdk_version"] = "unknown"
        elif tamper == "case_identity":
            payload["net_name"] = "resnet50_part1_b39"
            payload["compiler_onnx_filename"] = (
                "resnet50_part1_b39_hailo_fixed.onnx"
            )
            payload["cache_payload"]["net_name"] = "resnet50_part1_b39"
            payload["cache_key"] = hashlib.sha256(
                json.dumps(
                    payload["cache_payload"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        CacheVerifyPolicyError,
        match=r"cache_miss_blocked\[hailo_dfc\]",
    ) as raised:
        attest_cache_verify_benchmark_sets(profile, benchmark_sets)
    text = str(raised.value)
    assert "Native pre-dispatch attestation failed" in text
    assert expected_reason in text


@pytest.mark.parametrize(
    ("backend", "expected_probe"),
    (("auto", True), ("local", True), ("venv", True), ("wsl", False)),
)
def test_hailo_cache_miss_stops_before_every_backend_compiler_dispatch(
    backend: str,
    expected_probe: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"cache-verify-test-onnx")
    miss = hailo_backend.HailoHefBuildResult(
        ok=False,
        elapsed_s=0.0,
        hw_arch="hailo8",
        net_name="model",
        backend="local",
        skipped=True,
        failure_kind="cache_miss_blocked",
        unsupported_reason="cache_only_policy",
        error="cache miss",
        last_stage="cache_lookup",
        calib_info={"cache_hit": False, "cache_only": True},
    )
    probe_calls: list[dict[str, Any]] = []

    def cache_probe(*_args: Any, **kwargs: Any):
        probe_calls.append(dict(kwargs))
        return miss

    def forbidden_dispatch(*_args: Any, **_kwargs: Any):
        raise AssertionError("Hailo local/venv/WSL/DFC dispatch started")

    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    monkeypatch.setattr(
        hailo_backend,
        "_resolve_hailo_image_contract",
        lambda **_kwargs: ({"task": "classification"}, "p" * 64),
    )
    monkeypatch.setattr(hailo_backend, "_hailo_build_hef_legacy", cache_probe)
    monkeypatch.setattr(
        hailo_backend, "hailo_build_hef_via_venv", forbidden_dispatch,
    )
    monkeypatch.setattr(
        hailo_backend, "hailo_build_hef_via_wsl", forbidden_dispatch,
    )
    monkeypatch.setattr(
        hailo_backend, "hailo_sdk_available", forbidden_dispatch,
    )
    monkeypatch.setattr(
        hailo_backend, "auto_prefers_subprocess", forbidden_dispatch,
    )
    monkeypatch.setattr(
        hailo_backend,
        "_hailo_sdk_version_token_from_managed_venv",
        lambda **_kwargs: "hailo-dataflow-compiler:3.33.1",
    )
    monkeypatch.setattr(
        hailo_backend,
        "_hailo_sdk_version_token_from_controller_metadata",
        lambda: "hailo-dataflow-compiler:3.33.1",
    )

    result = hailo_backend._hailo_build_hef_auto_dispatch(
        model,
        backend=backend,
        hw_arch="hailo8",
        task="classification",
        preprocessing_contract={"task": "classification"},
    )

    if expected_probe:
        assert result is miss
        assert len(probe_calls) == 1
        assert probe_calls[0]["cache_only"] is True
        assert probe_calls[0]["force"] is False
        assert (
            probe_calls[0]["sdk_version_token"]
            == "hailo-dataflow-compiler:3.33.1"
        )
    else:
        assert probe_calls == []
        assert result.failure_kind == "cache_miss_blocked"
        assert result.unsupported_reason == "compiler_identity_unavailable"


def test_managed_hailo_identity_is_read_from_metadata_without_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv = tmp_path / "hailo-venv"
    python = venv / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    metadata = (
        venv
        / "lib/python3.10/site-packages"
        / "hailo_dataflow_compiler-3.33.1.dist-info/METADATA"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "Metadata-Version: 2.1\n"
        "Name: hailo-dataflow-compiler\n"
        "Version: 3.33.1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hailo_backend,
        "_resolve_managed_venv_python",
        lambda **_kwargs: ("hailo8", python, str(venv / "bin/activate")),
    )
    monkeypatch.setattr(
        hailo_backend.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("process started while reading DFC identity")
        ),
    )

    assert hailo_backend._hailo_sdk_version_token_from_managed_venv(
        hw_arch="hailo8",
    ) == "hailo-dataflow-compiler:3.33.1"


def test_existing_b052_receipt_key_differs_only_by_compiler_identity() -> None:
    payload = {
        "schema": "onnx-splitpoint/hailo-hef-cache-key-v3",
        "model_sha256": (
            "2d0aea72831d82bbdd01751dc261e246886d2fc7f1d47a8bd8089e9dfd368c39"
        ),
        "activation_part1_sha256": "",
        "hw_arch": "hailo8",
        "hailo_sdk_version": "hailo-dataflow-compiler:3.33.1",
        "optimization_level": 1,
        "calibration_identity": (
            "manifest:6d6978cafd1b828d6fa9438b232120f2a769311c28492bea5a141af586c3d58c"
        ),
        "prepared_calibration_identity_sha256": (
            "fb09c137fe8380101f4d02f4c3998037407c0cce5d1b09aa854c61ba41172b18"
        ),
        "calibration_count": 500,
        "requested_calibration_count": 500,
        "calibration_storage": "memmap",
        "calibration_memory_cap_bytes": 268435456,
        "calibration_batch_size": 8,
        "extra_model_script": "",
        "start_nodes": [],
        "end_nodes": [],
        "integrity": "relaxed",
        "preprocessing_contract": {
            "schema": "onnx-splitpoint/image-preprocessing-contract",
            "schema_version": 2,
            "contract_scope": "prepared_rgb_uint8_semantics",
            "task": "classification",
            "preprocess_mode": "resize",
            "spatial_transform": "direct_resize",
            "target_hw": [224, 224],
            "color_space": "RGB",
            "input_domain": "uint8_0_255",
            "resize_interpolation": "bilinear",
            "resize_rounding": "python_round_ties_to_even",
            "placement": "not_applicable",
            "pad_value": 0,
            "letterbox_pad_value": 0,
            "image_scale": "imagenet",
            "letterbox": False,
        },
        "preprocessing_contract_sha256": (
            "ea28cf5ac35bd4c9a3321ac97fd559f32fd7a661dc4a93c324fe3ffc54188fa9"
        ),
        "net_name": "resnet50_part1_b52",
        "net_input_shapes": [1, 3, 224, 224],
        "disable_rt_metadata_extraction": True,
    }

    def cache_key(version: str) -> str:
        candidate = dict(payload)
        candidate["hailo_sdk_version"] = version
        return hashlib.sha256(
            json.dumps(
                candidate,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    assert cache_key("hailo-dataflow-compiler:3.33.1") == (
        "c4ec8608cc976698de3480726be1e2ae5fe1a9679b60ca47e137967faff8a4ee"
    )
    assert cache_key("unknown") == (
        "e577ae24f0b311b5bb1cc332ff5961763cdd56aedf513169684c63083c3a1c28"
    )


def test_policy_hailo_miss_is_explicit_and_writes_full_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "resnet50_part1_b52.onnx"
    model.write_bytes(b"stable-cache-identity-fixture")
    outdir = tmp_path / "out"
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_HAILO_CACHE_ROOT", str(tmp_path / "cache"),
    )
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ENABLED", "1")
    monkeypatch.setitem(
        sys.modules,
        "hailo_sdk_client",
        SimpleNamespace(
            __getattr__=lambda _name: (_ for _ in ()).throw(
                AssertionError("DFC module accessed on cache miss")
            )
        ),
    )

    result = hailo_backend._hailo_build_hef_legacy(
        model,
        outdir=outdir,
        hw_arch="hailo8",
        net_name="resnet50_part1_b52",
        net_input_shapes=[1, 3, 224, 224],
        fixup=False,
        cache_only=False,
        calib_count=8,
        task="classification",
        sdk_version_token="hailo-dataflow-compiler:3.33.1",
    )

    assert result.ok is False
    assert result.skipped is True
    assert result.failure_kind == "cache_miss_blocked"
    assert result.unsupported_reason == "cache_verify_only_policy"
    assert "cache_miss_blocked[hailo_dfc]" in str(result.error)
    assert "Smoke" not in str(result.error)
    diagnostic = json.loads(
        (outdir / "hailo_cache_miss.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "cache_miss_blocked"
    assert len(diagnostic["cache_key_v3"]) == 64
    assert diagnostic["cache_payload_v3"]["hailo_sdk_version"] == (
        "hailo-dataflow-compiler:3.33.1"
    )
    assert diagnostic["compiler_dispatch_allowed"] is False


def test_native_stage_skips_before_progress_or_dispatch_after_cache_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(tmp_path)),
    )
    runner.run_id = "cache-blocked-native"
    runner.run_dir = tmp_path / runner.run_id
    runner.profile_payload = _resolved_profile()
    runner.stage_results = [{
        "stage": "generate_benchmark_set",
        "status": "failed",
        "details": {"failure_kind": "cache_miss_blocked"},
        "notes": ["exact b052 cache_miss_blocked"],
    }]
    monkeypatch.setattr(
        runner,
        "_native_producer_config",
        lambda: dict(runner.profile_payload["native_producers"]),
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.workflow.runner.run_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native/SSH dispatch started")
        ),
    )

    paths, details, message, status = runner._stage_run_native_producers()

    assert status == "skipped"
    assert details["failure_kind"] == "cache_miss_blocked"
    assert details["native_dispatch_started"] is False
    assert "cache_miss_blocked" in message
    report = json.loads(
        paths["native_producer_stage_json"].read_text(encoding="utf-8")
    )
    assert report["status"] == "cache_miss_blocked"
    assert report["native_dispatch_started"] is False
    progress = runner.run_dir / "reports/native_progress.jsonl"
    assert not progress.exists() or "STAGE_START" not in progress.read_text(
        encoding="utf-8",
    )


def test_hailo_legacy_force_conflict_stops_before_contract_or_compiler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    monkeypatch.setattr(
        hailo_backend,
        "_resolve_hailo_image_contract",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("preprocessing/compiler contract resolution reached")
        ),
    )
    result = hailo_backend._hailo_build_hef_legacy(
        tmp_path / "missing.onnx",
        hw_arch="hailo8",
        force=True,
        cache_only=False,
    )
    assert result.ok is False
    assert result.failure_kind == "cache_miss_blocked"
    assert result.unsupported_reason == "cache_verify_only_force_conflict"
    assert "DFC dispatch was not started" in str(result.error)


def test_direct_native_hailo_compile_core_stops_before_client_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class ForbiddenClientRunner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("Hailo ClientRunner was constructed")

    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    monkeypatch.setitem(
        sys.modules,
        "hailo_sdk_client",
        SimpleNamespace(ClientRunner=ForbiddenClientRunner),
    )
    result = native_hailo_backend._compile_hailo_hef_core(
        model_path=tmp_path / "model.onnx",
        hef_path=tmp_path / "compiled.hef",
        hw_arch="hailo8",
        net_name="model",
        opt_level=1,
        calib_dir=None,
        calib_count=1,
        calib_batch_size=1,
        fixup=False,
        keep_artifacts=False,
    )
    assert result["ok"] is False
    assert result["status"] == "cache_miss_blocked"
    assert "cache_miss_blocked[hailo_dfc]" in result["error"]
    assert not (tmp_path / "compiled.hef").exists()


def test_deepx_low_level_fence_returns_before_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    monkeypatch.setattr(
        deepx_compiler.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DX-COM Popen started")
        ),
    )
    result = deepx_compiler._run_owned_compiler(
        ["dxcom", "-m", "model.onnx"], timeout_s=1,
    )
    assert result.returncode == 78
    assert "cache_miss_blocked[deepx_dx_com]" in result.stdout


def test_deepx_environment_inspection_is_process_free_under_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    root = tmp_path / "dx-all-suite"
    compiler_venv = tmp_path / "compiler-venv"
    runtime_venv = tmp_path / "runtime-venv"
    root.mkdir()
    for path in (
        compiler_venv / "bin/python",
        compiler_venv / "bin/dxcom",
        runtime_venv / "bin/python",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"executable-placeholder")

    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"DeepX environment inspection reached {name}")
        return fail

    monkeypatch.setattr(
        deepx_env_status.subprocess, "Popen", forbidden("Popen"),
    )
    monkeypatch.setattr(
        deepx_env_status, "_python_tag", forbidden("compiler_python_tag"),
    )
    monkeypatch.setattr(
        deepx_env_status, "_probe_python_import", forbidden("module_import"),
    )

    status = deepx_env_status.inspect_deepx_environment(
        root=root,
        probe=True,
        probe_import=True,
        config={
            "compiler_venv": str(compiler_venv),
            "runtime_venv": str(runtime_venv),
            "cache_dir": str(tmp_path / "cache"),
        },
    )

    assert forbidden_calls == []
    assert status["probe_blocked_by_artifact_policy"] is True
    assert status["compiler_python_tag"] == ""
    assert status["compiler_imports"] == []
    assert status["runtime_imports"] == []


def _forbid_deepx_probe_and_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"manual DeepX cache-only path reached {name}")
        return fail

    monkeypatch.setattr(
        deepx_env_status.subprocess, "Popen", forbidden("Popen"),
    )
    monkeypatch.setattr(
        deepx_env_status, "_python_tag", forbidden("compiler_python_tag"),
    )
    monkeypatch.setattr(
        deepx_env_status, "_probe_python_import", forbidden("module_import"),
    )
    monkeypatch.setattr(
        deepx_compiler, "compile_dxnn", forbidden("compile_dxnn"),
    )
    return forbidden_calls


def _manual_deepx_cache_source(path: Path):
    # Mean/Std preparation validates ONNX before the cache-only compiler
    # fence. A valid tiny graph lets these tests reach the intended fence.
    import onnx
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "manual-deepx-cache-source",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return model


def test_manual_deepx_full_cache_miss_blocks_without_probe_or_compile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    monkeypatch.setattr(
        deepx_env_status, "default_cache_dir", lambda: tmp_path / "cache",
    )
    forbidden_calls = _forbid_deepx_probe_and_compile(monkeypatch)
    model_path = tmp_path / "resnet50.onnx"
    model = _manual_deepx_cache_source(model_path)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "sample.jpg").write_bytes(b"image-placeholder")

    result = benchmark_workflow._materialize_manual_deepx_full_artifact(
        out_dir=tmp_path / "suite",
        model_path=str(model_path),
        model=model,
        bench_plan_runs=[{"type": "deepx"}],
        validation_images=str(calibration),
        validation_max_images=1,
        fallback_calib_dir=None,
        calibration_num=1,
        task_hint="classification",
    )

    assert forbidden_calls == []
    assert result["artifact_policy"] == CACHE_VERIFY_ONLY
    assert result["status"] == "cache_miss_blocked"
    assert result["build_status"] == "cache_miss_blocked"
    assert "cache_miss_blocked[deepx_dx_com]" in result["message"]
    assert not (tmp_path / "suite/deepx/deepx_m1/full/build").exists()


def test_manual_deepx_part1_cache_miss_blocks_without_probe_or_compile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    monkeypatch.setattr(
        deepx_env_status, "default_cache_dir", lambda: tmp_path / "cache",
    )
    forbidden_calls = _forbid_deepx_probe_and_compile(monkeypatch)
    suite = tmp_path / "suite"
    case = suite / "b052"
    case.mkdir(parents=True)
    part1 = case / "resnet50_part1.onnx"
    _manual_deepx_cache_source(part1)
    (case / "split_manifest.json").write_text(
        json.dumps({"part1_model": part1.name}) + "\n",
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "sample.jpg").write_bytes(b"image-placeholder")

    result = benchmark_workflow._materialize_manual_deepx_part1_artifacts(
        out_dir=suite,
        bench_plan_runs=[{
            "id": "deepx_m1_to_tensorrt",
            "type": "matrix",
            "stage1": "deepx_m1",
            "stage2": "tensorrt",
        }],
        validation_images=str(calibration),
        fallback_calib_dir=None,
        calibration_num=1,
        task_hint="classification",
    )

    assert forbidden_calls == []
    assert result["artifact_policy"] == CACHE_VERIFY_ONLY
    assert result["status"] == "cache_miss_blocked"
    assert result["ok_count"] == 0
    assert result["failed_count"] == 1
    assert result["cases"][0]["status"] == "cache_miss_blocked"
    assert "cache_miss_blocked[deepx_dx_com]" in result["cases"][0]["error"]
    assert not (case / "deepx/deepx_m1/part1/build").exists()


def test_bound_artifact_policy_enables_low_level_fence_and_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    assert compiler_dispatch_forbidden() is False
    popen_calls: list[list[str]] = []

    def forbidden_popen(cmd: list[str], **_kwargs: Any):
        popen_calls.append(list(cmd))
        raise AssertionError("DX-COM Popen started inside bound cache policy")

    monkeypatch.setattr(deepx_compiler.subprocess, "Popen", forbidden_popen)
    with bind_artifact_policy(_resolved_profile()) as policy:
        assert policy == CACHE_VERIFY_ONLY
        assert compiler_dispatch_forbidden() is True
        blocked = deepx_compiler._run_owned_compiler(
            ["dxcom", "-m", "model.onnx"], timeout_s=1,
        )
        assert blocked.returncode == 78
        assert "cache_miss_blocked[deepx_dx_com]" in blocked.stdout

    assert popen_calls == []
    assert compiler_dispatch_forbidden() is False


def test_bound_artifact_policy_is_visible_to_worker_threads_and_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    assert compiler_dispatch_forbidden() is False
    with bind_artifact_policy(_resolved_profile()):
        with ThreadPoolExecutor(max_workers=1) as pool:
            observed = pool.submit(compiler_dispatch_forbidden).result(timeout=5)
        assert observed is True
    with ThreadPoolExecutor(max_workers=1) as pool:
        reset = pool.submit(compiler_dispatch_forbidden).result(timeout=5)
    assert reset is False


def test_bound_cache_policy_forces_activation_proxy_to_cpu_without_trt_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", raising=False)
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND", "tensorrt_ort",
    )
    session_calls: list[tuple[Any, ...]] = []
    smoke_calls: list[str] = []

    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        @staticmethod
        def InferenceSession(*args: Any, **_kwargs: Any):
            session_calls.append(tuple(args))
            raise AssertionError("activation proxy created an ORT session")

    def forbidden_smoke(_ort: Any, provider: str) -> dict[str, Any]:
        smoke_calls.append(provider)
        raise AssertionError("activation proxy ran provider session smoke")

    monkeypatch.setattr(
        hailo_backend,
        "_activation_proxy_provider_session_smoke",
        forbidden_smoke,
    )
    with bind_artifact_policy(_resolved_profile()):
        selected = hailo_backend._activation_proxy_provider_selection(FakeOrt)

    assert selected["requested_backend"] == "tensorrt_ort"
    assert selected["producer_backend"] == "ort_cpu"
    assert selected["providers_requested"] == ["CPUExecutionProvider"]
    assert selected["provider_session_smoke"] == {}
    assert "cache_miss_blocked[tensorrt_ort_ep]" in selected["fallback_reason"]
    assert smoke_calls == []
    assert session_calls == []
    assert compiler_dispatch_forbidden() is False


def test_native_trt_blocks_build_command_but_allows_load_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native_trt = _load_script("native_trt_from_benchmarkset.py")
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="Throughput: 1.0 qps\n")

    monkeypatch.setattr(native_trt.subprocess, "run", fake_run)
    blocked = native_trt._run(
        ["trtexec", "--onnx=model.onnx", "--saveEngine=model.engine"],
        cwd=tmp_path,
        log_path=tmp_path / "blocked.log",
        dry_run=False,
    )
    assert blocked["returncode"] == 78
    assert blocked["status"] == "cache_miss_blocked"
    assert calls == []

    reused = native_trt._run(
        ["trtexec", "--loadEngine=model.engine", "--iterations=1"],
        cwd=tmp_path,
        log_path=tmp_path / "reused.log",
        dry_run=False,
    )
    assert reused["returncode"] == 0
    assert calls == [[
        "trtexec", "--loadEngine=model.engine", "--iterations=1",
    ]]


def test_generated_runner_rejects_ort_trt_before_session_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_template(
        monkeypatch,
        "run_split_onnxruntime.py.txt",
        stub_onnx_runtime=True,
    )
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    constructor_calls: list[bool] = []

    def forbidden_session(*_args: Any, **_kwargs: Any):
        constructor_calls.append(True)
        raise AssertionError("ORT TensorRT InferenceSession was constructed")

    monkeypatch.setattr(runner.ort, "InferenceSession", forbidden_session)
    with pytest.raises(
        RuntimeError,
        match=(
            "cache_miss_blocked:artifact_policy=cache_verify_only:"
            "compiler=onnxruntime_tensorrt_ep:artifact=session:blocked"
        ),
    ):
        runner._create_session(
            "blocked",
            Path("missing.onnx"),
            ["TensorrtExecutionProvider", "CPUExecutionProvider"],
            None,
            runner.ort.SessionOptions(),
        )
    assert constructor_calls == []


def test_generated_runner_native_trt_cache_miss_never_calls_build_or_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load_template(
        monkeypatch,
        "run_split_onnxruntime.py.txt",
        stub_onnx_runtime=True,
    )
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    model = tmp_path / "part2.onnx"
    model.write_bytes(b"cache-verify-native-trt-source")
    build_calls: list[bool] = []
    load_calls: list[bool] = []

    def forbidden_build(*_args: Any, **_kwargs: Any):
        build_calls.append(True)
        raise AssertionError("Native TensorRT build method reached")

    def forbidden_load(*_args: Any, **_kwargs: Any):
        load_calls.append(True)
        raise AssertionError("Native TensorRT engine load reached after miss")

    monkeypatch.setattr(runner.NativeTRTSession, "_build_engine", forbidden_build)
    monkeypatch.setattr(runner.NativeTRTSession, "_load_engine", forbidden_load)
    with pytest.raises(
        RuntimeError,
        match=(
            "cache_miss_blocked:artifact_policy=cache_verify_only:"
            "compiler=trtexec:artifact=native_tensorrt_part2"
        ),
    ):
        runner.NativeTRTSession(
            "part2",
            model,
            precision="fp16",
            allow_build=True,
            cache_root=tmp_path / "cache",
        )
    assert build_calls == []
    assert load_calls == []


def test_generated_runner_build_engine_fence_precedes_trtexec_and_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_template(
        monkeypatch,
        "run_split_onnxruntime.py.txt",
        stub_onnx_runtime=True,
    )
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    discovery_calls: list[bool] = []
    subprocess_calls: list[bool] = []

    def forbidden_discovery() -> str:
        discovery_calls.append(True)
        raise AssertionError("trtexec discovery reached")

    def forbidden_subprocess(*_args: Any, **_kwargs: Any):
        subprocess_calls.append(True)
        raise AssertionError("trtexec subprocess reached")

    monkeypatch.setattr(runner, "_find_trtexec_local", forbidden_discovery)
    monkeypatch.setattr(runner.subprocess, "run", forbidden_subprocess)
    native = runner.NativeTRTSession.__new__(runner.NativeTRTSession)
    native.kind = "part2"
    # No engine_path/model_path is assigned on purpose: accessing either would
    # prove the fence did not run before filesystem/build preparation.
    with pytest.raises(
        RuntimeError,
        match=(
            "cache_miss_blocked:artifact_policy=cache_verify_only:"
            "compiler=trtexec:artifact=native_tensorrt_part2"
        ),
    ):
        native._build_engine(workspace_mb=1, timeout_s=1)
    assert discovery_calls == []
    assert subprocess_calls == []


def test_benchmark_suite_full_quality_cache_miss_precedes_copy_and_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    suite = _load_template(monkeypatch, "benchmark_suite.py.txt")
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    root = tmp_path / "benchmark_set"
    models = root / "models"
    models.mkdir(parents=True)
    (models / "resnet50.onnx").write_bytes(b"canonical-full-onnx")
    cache_root = tmp_path / "persistent_trt_cache"
    copy_calls: list[bool] = []
    child_calls: list[bool] = []
    subprocess_calls: list[bool] = []

    def forbidden_copy(*_args: Any, **_kwargs: Any):
        copy_calls.append(True)
        raise AssertionError("Full Quality copied a cache-miss source")

    def forbidden_child(*_args: Any, **_kwargs: Any):
        child_calls.append(True)
        raise AssertionError("Full Quality launched the case child")

    def forbidden_subprocess(*_args: Any, **_kwargs: Any):
        subprocess_calls.append(True)
        raise AssertionError("Full Quality launched a build process")

    monkeypatch.setattr(suite.shutil, "copy2", forbidden_copy)
    monkeypatch.setattr(suite, "_run_case", forbidden_child)
    monkeypatch.setattr(suite.subprocess, "run", forbidden_subprocess)
    args = SimpleNamespace(
        quality_evidence_eval_id="eval-cache-verify",
        quality_evidence_setup_id="orin_nx_trt_01",
        quality_evidence_model_id="resnet50",
        trt_cache_root=str(cache_root),
        native_trt_precision="fp16",
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "cache_miss_blocked:artifact_policy=cache_verify_only:"
            "compiler=trtexec:artifact=native_full_quality_source_onnx"
        ),
    ):
        suite._run_native_full_trt_quality_companion(
            root=root,
            bench={"model_id": "resnet50"},
            plan={},
            run={
                "id": "native_full_tensorrt",
                "_native_full_trt_quality_companion": True,
                "quality_canary_endpoint_ids": [
                    "tensorrt_at_deepx_m1_full"
                ],
            },
            run_cases=[{"case_dir": "b052"}],
            args=args,
        )
    assert copy_calls == []
    assert child_calls == []
    assert subprocess_calls == []
    assert not cache_root.exists()


def _patch_split_quality_sources(
    monkeypatch: pytest.MonkeyPatch,
    *, root: Path, part1: Path, part2: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    policy = native_split_quality_runtime.known_native_split_policy(
        model_id="yolo26s",
        case_id="b038",
        setup_id="hailo8_setup",
        backend="hailo8_to_trt",
    )
    assert policy is not None
    boundary_metadata = {
        "name": "cut",
        "runtime_name": "cut/hailort",
        "shape": [2, 2, 2],
        "canonical_part2_shape": [1, 2, 2, 2],
        "dtype": "uint8",
        "quantization": {
            "source": "hailort_hef_output_vstream_info",
            "scale": 0.03125,
            "zero_point": 17.0,
        },
    }
    monkeypatch.setattr(
        native_split_quality_runtime,
        "_find_part1",
        lambda *_args, **_kwargs: part1,
    )
    monkeypatch.setattr(
        native_split_quality_runtime,
        "_find_part2",
        lambda *_args, **_kwargs: part2,
    )
    monkeypatch.setattr(
        native_split_quality_runtime,
        "_onnx_input",
        lambda *_args, **_kwargs: {
            "name": "cut", "shape": [1, 2, 2, 2], "dtype": "uint8",
        },
    )
    monkeypatch.setattr(
        native_split_quality_runtime,
        "_hailo_metadata",
        lambda **_kwargs: copy.deepcopy(boundary_metadata),
    )
    key = native_split_quality_runtime.canonical_json_sha256({
        "policy_sha256": policy["policy_sha256"],
        "part1_sha256": native_split_quality_runtime._sha256_file(part1),
        "source_part2_sha256": native_split_quality_runtime._sha256_file(part2),
        "boundary_tensor": boundary_metadata,
        "resolved_boundary_layout": policy["boundary_layout"],
        "resolved_boundary_transform": policy["boundary_transform"],
    })
    assert (root / "b038").is_dir()
    return policy, boundary_metadata, key


def test_split_quality_cache_miss_never_dispatches_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    root = tmp_path / "benchmark_set"
    (root / "b038").mkdir(parents=True)
    part1 = tmp_path / "source/part1.hef"
    part2 = tmp_path / "source/part2.onnx"
    part1.parent.mkdir(parents=True)
    part1.write_bytes(b"cache-verify-part1")
    part2.write_bytes(b"cache-verify-part2")
    _patch_split_quality_sources(
        monkeypatch, root=root, part1=part1, part2=part2,
    )
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"split Quality reached {name}")
        return fail

    monkeypatch.setattr(
        native_split_quality_runtime, "_persistent_copy", forbidden("copy"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_onnx_input", forbidden("onnx"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_hailo_metadata", forbidden("hailo"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_deepx_metadata", forbidden("deepx"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_builder_script", forbidden("builder"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime.subprocess,
        "run",
        forbidden("subprocess"),
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "cache_miss_blocked:artifact_policy=cache_verify_only:"
            "compiler=trtexec:artifact=native_split_quality_binding:"
            "reason=missing"
        ),
    ):
        native_split_quality_runtime.prepare_native_split_quality_binding(
            benchmark_set=root,
            case_id="b038",
            model_id="yolo26s",
            setup_id="hailo8_setup",
            backend="hailo8_to_trt",
            eval_run_id="eval-cache-verify",
            source_run_id="hailo8_to_trt",
            cache_root=tmp_path / "persistent",
            output_path=tmp_path / "requested/binding.json",
        )
    assert forbidden_calls == []


def test_split_quality_replays_exact_locally_verified_binding_without_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    payload, paths = _fixture_payload(tmp_path / "producer")
    cached_binding = seal_native_split_quality_binding(payload)
    root = tmp_path / "benchmark_set"
    (root / "b038").mkdir(parents=True)
    _policy, _boundary, key = _patch_split_quality_sources(
        monkeypatch,
        root=root,
        part1=paths["part1_runtime"],
        part2=paths["source_part2_onnx"],
    )
    cache_root = tmp_path / "persistent"
    persistent = (
        cache_root / "native_split_quality" / "hailo8_setup"
        / "yolo26s" / "b038" / "hailo8_to_trt" / key
    )
    persistent.mkdir(parents=True)
    persistent_binding = persistent / "native_split_quality_binding.json"
    persistent_binding.write_text(
        json.dumps(cached_binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"split Quality reached {name}")
        return fail

    monkeypatch.setattr(
        native_split_quality_runtime, "_persistent_copy", forbidden("copy"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_onnx_input", forbidden("onnx"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_hailo_metadata", forbidden("hailo"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_deepx_metadata", forbidden("deepx"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_builder_script", forbidden("builder"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime.subprocess,
        "run",
        forbidden("subprocess"),
    )
    requested = tmp_path / "requested/native_split_quality_binding.json"
    replay = native_split_quality_runtime.prepare_native_split_quality_binding(
        benchmark_set=root,
        case_id="b038",
        model_id="yolo26s",
        setup_id="hailo8_setup",
        backend="hailo8_to_trt",
        eval_run_id="eval-cache-verify-new",
        source_run_id="hailo8_to_trt",
        cache_root=cache_root,
        output_path=requested,
    )

    assert forbidden_calls == []
    assert replay["cache_verify_reused"] is True
    assert replay["persistent_binding_path"] == str(persistent_binding)
    assert replay["engine_path"] == cached_binding["artifacts"]["engine"]["path"]
    emitted = json.loads(requested.read_text(encoding="utf-8"))
    assert emitted["eval_run_id"] == "eval-cache-verify-new"
    assert emitted["cache_verify_replay"] == {
        "artifact_policy": CACHE_VERIFY_ONLY,
        "source_binding_sha256": cached_binding["binding_sha256"],
        "local_validation_status": (
            "local_files_rehashed_and_exact_cross_links_verified"
        ),
        "compiler_dispatched": False,
    }
    verified, status = validate_native_split_quality_binding(
        emitted,
        expected_identity={
            "model": "yolo26s", "case": "b038",
            "setup_id": "hailo8_setup", "backend": "hailo8_to_trt",
            "task": "detection", "precision": "uint8_dequant_fp16",
        },
        verification_mode="local",
    )
    assert verified is not None, status


def test_split_quality_replays_identical_binding_copies_deterministically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    payload, paths = _fixture_payload(tmp_path / "producer")
    cached_binding = seal_native_split_quality_binding(payload)
    root = tmp_path / "benchmark_set"
    (root / "b038").mkdir(parents=True)
    _policy, _boundary, _key = _patch_split_quality_sources(
        monkeypatch,
        root=root,
        part1=paths["part1_runtime"],
        part2=paths["source_part2_onnx"],
    )
    binding_root = (
        tmp_path / "persistent/native_split_quality/hailo8_setup"
        / "yolo26s/b038/hailo8_to_trt"
    )
    first = binding_root / "a-copy/native_split_quality_binding.json"
    second = binding_root / "b-copy/native_split_quality_binding.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(
        json.dumps(cached_binding, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(cached_binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"duplicate replay reached {name}")
        return fail

    monkeypatch.setattr(
        native_split_quality_runtime, "_persistent_copy", forbidden("copy"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_onnx_input", forbidden("onnx"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_builder_script", forbidden("builder"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime.subprocess,
        "run",
        forbidden("subprocess"),
    )

    replay = native_split_quality_runtime.prepare_native_split_quality_binding(
        benchmark_set=root,
        case_id="b038",
        model_id="yolo26s",
        setup_id="hailo8_setup",
        backend="hailo8_to_trt",
        eval_run_id="eval-cache-verify-duplicate",
        source_run_id="hailo8_to_trt",
        cache_root=tmp_path / "persistent",
        output_path=tmp_path / "requested/binding.json",
    )

    assert forbidden_calls == []
    assert replay["persistent_binding_path"] == str(first)
    assert replay["cache_verify_reused"] is True


def test_split_quality_selects_one_complete_binding_inside_one_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    payload, paths = _fixture_payload(tmp_path / "producer")
    first_binding = seal_native_split_quality_binding(payload)
    second_payload = copy.deepcopy(payload)
    second_payload["eval_run_id"] = "distinct-sealed-binding"
    second_binding = seal_native_split_quality_binding(second_payload)
    assert first_binding["binding_sha256"] != second_binding["binding_sha256"]

    root = tmp_path / "benchmark_set"
    (root / "b038").mkdir(parents=True)
    _policy, _boundary, _key = _patch_split_quality_sources(
        monkeypatch,
        root=root,
        part1=paths["part1_runtime"],
        part2=paths["source_part2_onnx"],
    )
    binding_root = (
        tmp_path / "persistent/native_split_quality/hailo8_setup"
        / "yolo26s/b038/hailo8_to_trt"
    )
    for name, binding in (
        ("a-first", first_binding), ("b-second", second_binding),
    ):
        path = binding_root / name / "native_split_quality_binding.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    replay = native_split_quality_runtime.prepare_native_split_quality_binding(
        benchmark_set=root,
        case_id="b038",
        model_id="yolo26s",
        setup_id="hailo8_setup",
        backend="hailo8_to_trt",
        eval_run_id="eval-cache-verify-distinct",
        source_run_id="hailo8_to_trt",
        cache_root=tmp_path / "persistent",
        output_path=tmp_path / "requested/binding.json",
    )

    assert replay["persistent_binding_path"].endswith(
        "a-first/native_split_quality_binding.json"
    )
    assert replay["cache_verify_compatible_binding_count"] == 2
    assert replay["cache_verify_distinct_artifact_set_count"] == 2
    assert replay["cache_verify_exact_binding_count"] == 1
    assert (tmp_path / "requested/binding.json").is_file()


def test_cross_suite_cache_replay_deduplicates_full_binding_copies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload, paths = _fixture_payload(tmp_path / "producer")
    first_binding = seal_native_split_quality_binding(payload)
    second_binding = copy.deepcopy(first_binding)

    root = tmp_path / "benchmark_set"
    (root / "b038").mkdir(parents=True)
    _policy, _boundary, _key = _patch_split_quality_sources(
        monkeypatch,
        root=root,
        part1=paths["part1_runtime"],
        part2=paths["source_part2_onnx"],
    )
    cache_parent = tmp_path / "suite-caches"
    for suite, binding in (
        ("a-historical-smoke", first_binding),
        ("b-independent-smoke", second_binding),
    ):
        path = (
            cache_parent / suite / "native_split_quality/hailo8_setup"
            / "yolo26s/b038/hailo8_to_trt/one"
            / "native_split_quality_binding.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(binding, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"cross-suite replay reached {name}")
        return fail

    monkeypatch.setattr(
        native_split_quality_runtime, "_persistent_copy", forbidden("copy"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_onnx_input", forbidden("onnx"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_builder_script", forbidden("builder"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime.subprocess,
        "run",
        forbidden("subprocess"),
    )

    helper = _load_script("materialize_cache_verify_native_split_binding.py")
    output = tmp_path / "run/native_split_quality_binding_set.json"
    result = helper.materialize(
        benchmark_set=root,
        model_id="yolo26s",
        case_id="b038",
        setup_id="hailo8_setup",
        backend="hailo8_to_trt",
        eval_run_id="cross-suite-cache-verify",
        engine_cache_root=cache_parent,
        output=output,
    )

    assert forbidden_calls == []
    attestation = result["cache_verify_attestation"]
    assert Path(attestation["selected_cache_root"]).name == (
        "a-historical-smoke"
    )
    assert attestation["exact_hit_count"] == 2
    assert attestation["distinct_source_binding_sha256_count"] == 1
    assert output.is_file()


def test_split_quality_rejects_invalid_binding_instead_of_reporting_clean_miss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    root = tmp_path / "benchmark_set"
    (root / "b038").mkdir(parents=True)
    part1 = tmp_path / "source/part1.hef"
    part2 = tmp_path / "source/part2.onnx"
    part1.parent.mkdir(parents=True)
    part1.write_bytes(b"cache-verify-part1")
    part2.write_bytes(b"cache-verify-part2")
    _patch_split_quality_sources(
        monkeypatch, root=root, part1=part1, part2=part2,
    )
    persistent = (
        tmp_path / "persistent/native_split_quality/hailo8_setup"
        / "yolo26s/b038/hailo8_to_trt/tampered"
    )
    persistent.mkdir(parents=True)
    (persistent / "native_split_quality_binding.json").write_text(
        '{"schema":"onnx-splitpoint/native-split-quality-binding",'
        '"schema":"forged"}\n',
        encoding="utf-8",
    )
    forbidden_calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"invalid binding reached {name}")
        return fail

    monkeypatch.setattr(
        native_split_quality_runtime, "_persistent_copy", forbidden("copy"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime, "_builder_script", forbidden("builder"),
    )
    monkeypatch.setattr(
        native_split_quality_runtime.subprocess,
        "run",
        forbidden("subprocess"),
    )
    with pytest.raises(
        RuntimeError,
        match="invalid_binding_count_1_first_binding_json_invalid_ValueError",
    ):
        native_split_quality_runtime.prepare_native_split_quality_binding(
            benchmark_set=root,
            case_id="b038",
            model_id="yolo26s",
            setup_id="hailo8_setup",
            backend="hailo8_to_trt",
            eval_run_id="eval-cache-verify",
            source_run_id="hailo8_to_trt",
            cache_root=tmp_path / "persistent",
            output_path=tmp_path / "requested/binding.json",
        )
    assert forbidden_calls == []


def test_split_quality_rejects_symlinked_binding_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    root = tmp_path / "benchmark_set"
    (root / "b038").mkdir(parents=True)
    part1 = tmp_path / "source/part1.hef"
    part2 = tmp_path / "source/part2.onnx"
    part1.parent.mkdir(parents=True)
    part1.write_bytes(b"cache-verify-part1")
    part2.write_bytes(b"cache-verify-part2")
    _patch_split_quality_sources(
        monkeypatch, root=root, part1=part1, part2=part2,
    )
    binding_root = (
        tmp_path / "persistent/native_split_quality/hailo8_setup"
        / "yolo26s/b038/hailo8_to_trt"
    )
    binding_root.mkdir(parents=True)
    outside = tmp_path / "outside/tampered"
    outside.mkdir(parents=True)
    (outside / "native_split_quality_binding.json").write_text(
        "{}\n", encoding="utf-8",
    )
    (binding_root / "symlinked-hash-parent").symlink_to(
        outside, target_is_directory=True,
    )

    with pytest.raises(
        RuntimeError,
        match="invalid_binding_count_1_first_binding_path_not_regular_file",
    ):
        native_split_quality_runtime.prepare_native_split_quality_binding(
            benchmark_set=root,
            case_id="b038",
            model_id="yolo26s",
            setup_id="hailo8_setup",
            backend="hailo8_to_trt",
            eval_run_id="eval-cache-verify",
            source_run_id="hailo8_to_trt",
            cache_root=tmp_path / "persistent",
            output_path=tmp_path / "requested/binding.json",
        )


def test_split_quality_clean_mismatch_reports_exact_hash_axis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    payload, paths = _fixture_payload(tmp_path / "producer")
    cached_binding = seal_native_split_quality_binding(payload)
    root = tmp_path / "benchmark_set"
    (root / "b038").mkdir(parents=True)
    current_part1 = tmp_path / "current/part1.hef"
    current_part1.parent.mkdir(parents=True)
    current_part1.write_bytes(b"different-but-valid-smoke-hef")
    _patch_split_quality_sources(
        monkeypatch,
        root=root,
        part1=current_part1,
        part2=paths["source_part2_onnx"],
    )
    persistent = (
        tmp_path / "persistent/native_split_quality/hailo8_setup"
        / "yolo26s/b038/hailo8_to_trt/old-smoke"
    )
    persistent.mkdir(parents=True)
    (persistent / "native_split_quality_binding.json").write_text(
        json.dumps(cached_binding), encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "mismatch_axes=part1_runtime_sha256:"
            "artifact=native_split_quality_binding:reason=exact_hit_count_0"
        ),
    ):
        native_split_quality_runtime.prepare_native_split_quality_binding(
            benchmark_set=root,
            case_id="b038",
            model_id="yolo26s",
            setup_id="hailo8_setup",
            backend="hailo8_to_trt",
            eval_run_id="eval-cache-verify-new",
            source_run_id="hailo8_to_trt",
            cache_root=tmp_path / "persistent",
            output_path=tmp_path / "requested/binding.json",
        )


def test_native_coordinator_argv_is_cache_only_and_rejects_build_or_force(
    tmp_path: Path,
) -> None:
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    resolved = _resolved_profile()
    benchmark_set = tmp_path / "models/resnet50/benchmark_set"
    (benchmark_set / "b052").mkdir(parents=True)
    base = copy.deepcopy(resolved["native_producers"])
    base["_workflow_context"] = {
        "execution_preset": copy.deepcopy(resolved["execution_preset"]),
    }
    base["_native_execution_contract"] = resolve_native_execution_contract(
        resolved,
    )
    variant = copy.deepcopy(base["variants"][0])

    cmd = coordinator._build_update_cmd(
        tmp_path,
        base,
        variant,
        refresh_suites=False,
        timeout_s=30,
    )
    policy_index = cmd.index("--artifact-policy")
    assert cmd[policy_index + 1] == CACHE_VERIFY_ONLY
    assert "--no-build-missing-engines" in cmd
    assert "--refresh-suites" not in cmd
    assert "--native-force-rebuild-engines" not in cmd

    build = copy.deepcopy(base)
    build["build_missing_engines"] = True
    with pytest.raises(
        TensorRTQualityChainError,
        match="forbids build_missing_engines",
    ):
        coordinator._build_update_cmd(
            tmp_path, build, variant, refresh_suites=False, timeout_s=30,
        )

    forced = copy.deepcopy(variant)
    forced["force_rebuild_engines"] = True
    with pytest.raises(
        TensorRTQualityChainError,
        match="forbids force-rebuild flags",
    ):
        coordinator._build_update_cmd(
            tmp_path, base, forced, refresh_suites=False, timeout_s=30,
        )


def test_runner_rejects_matrix_drift_before_opening_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    valid_options = WorkflowOptions(
        profile=str(PROFILE),
        out=str(tmp_path / "valid-runs"),
        execution_mode="generate_and_run",
        artifact_policy=CACHE_VERIFY_ONLY,
        skip_benchmarks=True,
        require_fresh_run=True,
        required_run_mode="smoke",
    )
    valid_runner = EvaluationWorkflowRunner(valid_options)
    valid_runner.profile_payload = _resolved_profile()
    valid_runner._validate_requested_start_contract()
    assert valid_runner.cache_verify_attestation["mode"] == CACHE_VERIFY_ONLY

    resolved = _resolved_profile()
    resolved["native_producers"]["frames"] = 11
    options = WorkflowOptions(
        profile=str(PROFILE),
        out=str(tmp_path / "runs"),
        execution_mode="generate_and_run",
        artifact_policy=CACHE_VERIFY_ONLY,
    )
    runner = EvaluationWorkflowRunner(options)
    monkeypatch.setattr(
        runner,
        "_load_profile",
        lambda: setattr(runner, "profile_payload", resolved),
    )
    open_calls: list[bool] = []

    def forbidden_open() -> None:
        open_calls.append(True)
        raise AssertionError("_open_run_dir called after an invalid attestation")

    monkeypatch.setattr(runner, "_open_run_dir", forbidden_open)
    with pytest.raises(
        ValueError,
        match="start blocked before run creation",
    ):
        runner.run()
    assert open_calls == []
    assert not (tmp_path / "runs").exists()
