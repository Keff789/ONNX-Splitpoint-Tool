from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy.collector import _diagnostic_claim_payload
from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.runners.native_split_quality_runtime import (
    _hailo_metadata,
)
from onnx_splitpoint_tool.split_export_runners import (
    write_runner_skeleton_onnxruntime,
)
from onnx_splitpoint_tool.trt_quality_chain import (
    TensorRTQualityChainError,
    producer_set_from_central_quality_summary,
    split_binding_set_from_central_quality_summary,
)
from scripts.run_evalrun_native_producer_variants import (
    _validate_supplied_split_binding_set,
)
from scripts.update_evalset_native_producers import (
    _validate_native_split_quality_binding_set,
)
from tests.test_v269d_trt_quality_chain import (
    _build,
    _result,
    _strict_producer,
    _summary,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_SUMMARY = (
    ROOT / "tests" / "fixtures" / "v270b"
    / "central_quality_summary_2.70a_smoke.json"
)
EVAL_RUN_ID = "resnet_yolo26s_yolo7_20260722_152042"


def _real_summary() -> dict:
    value = json.loads(REAL_SUMMARY.read_text(encoding="utf-8"))
    assert len(value["results"]) == 27
    return value


def _selections(backend: str) -> list[dict]:
    return [
        {
            "model_id": "resnet50", "case_id": "b052",
            "backend": backend, "task": "classification",
            "precision": "float32_layout_fp16",
        },
        {
            "model_id": "yolo26s", "case_id": "b038",
            "backend": backend, "task": "detection",
            "precision": "float32_layout_fp16",
        },
        {
            "model_id": "yolov7_paper", "case_id": "b044",
            "backend": backend, "task": "detection",
            "precision": "float32_layout_fp16",
        },
    ]


@pytest.mark.parametrize(
    ("setup_id", "backend", "legacy_source"),
    [
        ("orin_nx_hailo10_01", "hailo10h_to_trt", "hailo10_to_trt"),
        ("orin_nx_deepx_m1_01", "deepx_to_trt", "deepx_m1_to_tensorrt"),
    ],
)
def test_real_270a_summary_rejects_legacy_full_but_preserves_split_bindings(
    tmp_path: Path, setup_id: str, backend: str, legacy_source: str,
) -> None:
    """Regression for the exact 2.70a hardware-summary failure shape."""

    summary = _real_summary()
    models = ["resnet50", "yolo26s", "yolov7_paper"]
    # The historical 2.70a Full producers predate the mandatory, hash-sealed
    # quality-record endpoint.  v2.75.14 must not retrofit or admit them as a
    # fresh Native-Full producer set; only newly exported requests can satisfy
    # the nine-row field gate.
    with pytest.raises(
        TensorRTQualityChainError,
        match="quality-endpoint provenance",
    ):
        producer_set_from_central_quality_summary(
            summary,
            eval_run_id=EVAL_RUN_ID,
            setup_id=setup_id,
            model_ids=models,
        )

    selections = _selections(backend)
    split_set = split_binding_set_from_central_quality_summary(
        summary,
        eval_run_id=EVAL_RUN_ID,
        setup_id=setup_id,
        selections=selections,
    )
    assert set(split_set["bindings_by_model_case_backend"]) == {
        "|".join((row["model_id"], row["case_id"], backend))
        for row in selections
    }
    for binding in split_set["bindings_by_model_case_backend"].values():
        # Old sealed producer artifacts remain unchanged, while the new
        # Central receipt and portable request digest are canonical.
        assert binding["source_run_id"] == legacy_source
        assert binding["central_quality_selection"]["source_run_id"] == backend
        assert len(binding["source_request_sha256"]) == 64
        int(binding["source_request_sha256"], 16)

    validated = _validate_supplied_split_binding_set(
        split_set,
        eval_run_id=EVAL_RUN_ID,
        setup_id=setup_id,
        selections=selections,
    )
    assert validated["binding_set_sha256"] == split_set["binding_set_sha256"]

    set_path = tmp_path / f"{setup_id}.json"
    set_path.write_text(json.dumps(split_set), encoding="utf-8")
    loaded = _validate_native_split_quality_binding_set(
        set_path,
        eval_run_id=EVAL_RUN_ID,
        setup_id=setup_id,
        selections=selections,
    )
    assert loaded["binding_set_sha256"] == split_set["binding_set_sha256"]


def test_real_270a_hailo8_missing_split_remains_a_true_upstream_error() -> None:
    with pytest.raises(TensorRTQualityChainError, match="no completed"):
        split_binding_set_from_central_quality_summary(
            _real_summary(),
            eval_run_id=EVAL_RUN_ID,
            setup_id="orin_nx_hailo8_01",
            selections=_selections("hailo8_to_trt")[:1],
        )


def test_full_request_sha_accepts_prefixed_and_bare_mirrors_but_not_double_prefix() -> None:
    row = _result(_strict_producer())
    row["source_request_sha256"] = "sha256:" + "a" * 64
    row["request_identity"]["source_request_sha256"] = "a" * 64
    assert list(_build(_summary([row]))["producers_by_model"]) == ["resnet50"]

    malformed = copy.deepcopy(row)
    malformed["source_request_sha256"] = "sha256:sha256:" + "a" * 64
    with pytest.raises(TensorRTQualityChainError, match="request SHA"):
        _build(_summary([malformed]))


def test_normal_generated_case_invocation_always_carries_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "remote" / "1" / "suite"
    case = suite / "b038"
    case.mkdir(parents=True)
    write_benchmark_suite_script(suite)
    write_runner_skeleton_onnxruntime(str(case), target="cpu")
    spec = importlib.util.spec_from_file_location(
        f"v270b_generated_suite_{id(tmp_path)}", suite / "benchmark_suite.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "_provider_unavailable_reason", lambda *_: None)
    monkeypatch.setattr(module, "_collect_case_result", lambda *_a, **_k: {"ok": True})
    result = module._run_case(
        case, "hailo8", "cpu", "default", "detection", "auto", 0, 1, 30,
        quality_evidence_model_id="yolo26s",
    )
    assert result == {"ok": True}
    command = captured["command"]
    assert command[command.index("--model-id") + 1] == "yolo26s"
    assert "base_dir.parent.parent.name" not in (
        case / "run_split_onnxruntime.py"
    ).read_text(encoding="utf-8")


def test_broken_hailo_python_fails_closed_without_system_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAILO_PY", str(tmp_path / "missing-hailo-python"))
    with pytest.raises(RuntimeError, match="hailo_python_invalid"):
        _hailo_metadata(
            part1=tmp_path / "boundary.hef",
            part2_input={"name": "x", "shape": [1]},
            policy={"boundary_dtype": "uint8", "quantization_policy": "none"},
        )


def test_diagnostic_energy_clamp_preserves_technical_pass_and_measurements() -> None:
    raw = {
        "final_energy_gate_status": "pass",
        "energy_total_j": 12.5,
        "energy_efficiency_claim_eligible": True,
        "scientific_primary_claim_eligible": True,
        "semantic_claim_ok": True,
        "energy_efficiency_claim_eligible_run_count": 1,
        "runs": [{
            "candidate_eligible_for_scientific_primary": True,
            "eligible_for_scientific_primary": True,
            "scientific_primary_energy_status": "available",
            "energy_per_work_unit_screening_estimate_j": 0.125,
        }],
    }
    clamped = _diagnostic_claim_payload(
        raw, exclusion_reason="smoke_diagnostic_only",
    )
    assert clamped["final_energy_gate_status"] == "pass"
    assert clamped["energy_total_j"] == 12.5
    assert clamped["runs"][0]["scientific_primary_energy_status"] == "available"
    assert clamped["runs"][0]["energy_per_work_unit_screening_estimate_j"] == 0.125
    assert clamped["energy_efficiency_claim_eligible_run_count"] == 0
    assert clamped["diagnostic_only"] is True
    assert clamped["claim_eligible"] is False
    assert clamped["semantic_claim_ok"] is False
    assert clamped["scientific_claim_exclusion_reasons"] == [
        "smoke_diagnostic_only"
    ]
    for field in (
        "energy_efficiency_claim_eligible",
        "scientific_primary_claim_eligible",
    ):
        assert clamped[field] is False
    for field in (
        "candidate_eligible_for_scientific_primary",
        "eligible_for_scientific_primary",
    ):
        assert clamped["runs"][0][field] is False
    # The Standard caller does not invoke the clamp; its input remains intact.
    assert raw["energy_efficiency_claim_eligible"] is True
