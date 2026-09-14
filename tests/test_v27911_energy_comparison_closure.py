from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy.collector import (
    HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
    apply_configured_energy_baselines,
)
from onnx_splitpoint_tool.energy.comparison import (
    resolve_energy_comparison,
    verify_accelerator_idle_calibration_binding,
)
from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup
from onnx_splitpoint_tool.native_energy_reporting import build_native_energy_pairs
from onnx_splitpoint_tool.platform_power import (
    _write_accelerator_idle_calibration_binding,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_row
from onnx_splitpoint_tool.workflow import scientific_reporting as scientific_reporting
from scripts import energy_measurement_cli as energy_cli


def _verified_trt_row() -> dict:
    return {
        "host_normalization_role": "tensorrt_full",
        "host_normalization_source_run_id": "native_full_tensorrt",
        "host_normalization_target_variant": "full",
        "host_normalization_identity_verified": True,
        "accelerator_idle_correction_requested": True,
        "accelerator_idle_correction_applied": True,
        "accelerator_idle_correction_statuses": ["applied"],
        "accelerator_idle_w_applied": 4.0,
        "accelerator_idle_calibration_verified": True,
        "accelerator_idle_calibration_status": "verified",
        "accelerator_idle_calibration_binding_sha256": "a" * 64,
        "energy_efficiency_claim_eligible": True,
        "energy_total_j": 120.0,
        "host_normalized_energy_est_j": 80.0,
        "energy_per_work_j": 0.12,
        "host_normalized_energy_per_work_est_j": 0.08,
        "average_power_w": 12.0,
        "host_normalized_average_power_est_w": 8.0,
        "active_duration_s": 10.0,
        "work_units": 1000,
    }


def test_calibration_binding_is_immutable_and_sha_verified(tmp_path: Path) -> None:
    def identity(name: str, content: bytes = b"{}\n") -> dict[str, str]:
        artifact = tmp_path / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(content)
        return {
            "path": str(artifact.resolve()),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    method = identity("energy_method.json")
    energy_method = {
        **method,
        "verification_status": "inherited_validated_method_verified",
        "runtime_binding_id": "orin_nx_hailo8_01",
    }
    captures = {
        phase: {
            "avg_power_w": power,
            "aggregate": identity(f"{phase}/energy_aggregate.json"),
            "report": identity(f"{phase}/energy_summary.json"),
            "raw": identity(f"{phase}/run_000/raw.parquet", b"parquet" + phase.encode()),
            "run": identity(f"{phase}/run_000/energy_summary.json"),
        }
        for phase, power in (("m2_off", 8.0), ("m2_on", 10.0))
    }
    jetson_host = {
        "address": "192.168.0.104",
        "user": "nx",
        "port": 22,
    }

    def observation(present: bool) -> dict[str, object]:
        return {
            "setup_id": "orin_nx_hailo8_01",
            "urecs_address": "192.168.0.197",
            "jetson_host": dict(jetson_host),
            "jetson_ssh_ready": True,
            "m2_present": present,
        }

    state_observations = {
        "initial": observation(True),
        "m2_off": {
            "pre_measurement": observation(False),
            "post_measurement": observation(False),
        },
        "m2_on": {
            "pre_measurement": observation(True),
            "post_measurement": observation(True),
        },
    }
    transitions = {
        phase: {
            "ok": True,
            "changed": True,
            "desired_accelerator_present": present,
            "after": observation(present),
        }
        for phase, present in (("m2_off", False), ("m2_on", True))
    }
    evidence = identity("accelerator_idle_calibration_evidence.json")
    path, sha256 = _write_accelerator_idle_calibration_binding(
        tmp_path,
        setup_id="orin_nx_hailo8_01",
        accelerator="hailo8",
        started_at="2026-09-02T12:00:00+00:00",
        finished_at="2026-09-02T12:02:00+00:00",
        off_power_w=8.0,
        on_power_w=10.0,
        accelerator_idle_power_w=2.0,
        urecs_address="192.168.0.197",
        data_port=3000,
        jetson_host=jetson_host,
        energy_method=energy_method,
        captures=captures,
        calibration_evidence=evidence,
        state_observations=state_observations,
        transitions=transitions,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert len(payload["binding_payload_sha256"]) == 64
    assert sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    setup = EnergySetup(
        setup_id="orin_nx_hailo8_01",
        urecs_address="192.168.0.197",
        accelerator_idle_w=2.0,
        accelerator_idle_calibration_binding_path=str(path),
        accelerator_idle_calibration_binding_sha256=sha256,
    )
    with pytest.raises(FileExistsError):
        _write_accelerator_idle_calibration_binding(
            tmp_path,
            setup_id="orin_nx_hailo8_01",
            accelerator="hailo8",
            started_at="x",
            finished_at="y",
            off_power_w=8.0,
            on_power_w=10.0,
            accelerator_idle_power_w=2.0,
            urecs_address="192.168.0.197",
            data_port=3000,
            jetson_host=jetson_host,
            energy_method=energy_method,
            captures=captures,
            calibration_evidence=evidence,
            state_observations=state_observations,
            transitions=transitions,
        )
    path.write_bytes(path.read_bytes() + b" ")
    tampered = verify_accelerator_idle_calibration_binding(setup)
    assert tampered["accelerator_idle_calibration_verified"] is False
    assert tampered["accelerator_idle_calibration_status"] == "unavailable_binding_sha256_mismatch"


def test_direct_tensorrt_full_correction_requires_identity_and_binding() -> None:
    summary = {"energy_total_j": 100.0, "active_duration_s": 10.0}
    missing = apply_configured_energy_baselines(
        summary,
        idle_baseline_w=None,
        accelerator_idle_w=2.0,
        host_normalization_role=HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
        host_normalization_source_run_id="native_full_tensorrt",
        host_normalization_target_variant="full",
    )
    assert missing["host_normalized_energy_est_j"] is None
    assert missing["accelerator_idle_correction_applied"] is False
    assert missing["accelerator_idle_correction_status"] == "unavailable_calibration_binding_unverified"

    applied = apply_configured_energy_baselines(
        summary,
        idle_baseline_w=None,
        accelerator_idle_w=2.0,
        host_normalization_role=HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
        accelerator_idle_calibration={
            "accelerator_idle_calibration_verified": True,
            "accelerator_idle_calibration_status": "verified",
            "accelerator_idle_calibration_binding_sha256": "a" * 64,
        },
        host_normalization_source_run_id="native_full_tensorrt",
        host_normalization_target_variant="full",
    )
    assert applied["energy_total_j"] == pytest.approx(100.0)
    assert applied["host_normalized_energy_est_j"] == pytest.approx(80.0)
    assert applied["accelerator_idle_correction_applied"] is True


def test_comparison_is_normalized_only_for_verified_tensorrt_full() -> None:
    trt = resolve_energy_comparison(_verified_trt_row())
    assert trt["raw_energy_per_work_j"] == pytest.approx(0.12)
    assert trt["comparison_energy_per_work_j"] == pytest.approx(0.08)
    assert trt["comparison_average_power_w"] == pytest.approx(8.0)
    assert trt["energy_comparison_basis"] == "host_normalized_accelerator_idle_subtracted"
    assert trt["energy_comparison_claim_ready"] is True

    broken = _verified_trt_row()
    broken["accelerator_idle_calibration_verified"] = False
    unavailable = resolve_energy_comparison(broken)
    assert unavailable["comparison_energy_per_work_j"] is None
    assert unavailable["comparison_average_power_w"] is None
    assert unavailable["energy_comparison_claim_ready"] is False

    raw = resolve_energy_comparison(
        {"energy_per_work_j": 0.2, "average_power_w": 20.0}
    )
    assert raw["comparison_energy_per_work_j"] == pytest.approx(0.2)
    assert raw["energy_comparison_basis"] == "raw_measured"
    assert raw["energy_comparison_claim_ready"] is False


@pytest.mark.parametrize(
    ("field", "value", "expected_status"),
    [
        (
            "host_normalization_role",
            "TENSORRT_FULL",
            "required_tensorrt_full_identity_unverified",
        ),
        (
            "host_normalization_identity_verified",
            "yes",
            "required_tensorrt_full_identity_unverified",
        ),
        (
            "host_normalization_identity_verified",
            1,
            "required_tensorrt_full_identity_unverified",
        ),
        (
            "host_normalization_source_run_id",
            "TRT_FULL",
            "required_tensorrt_full_identity_unverified",
        ),
        (
            "host_normalization_target_variant",
            "FULL",
            "required_tensorrt_full_identity_unverified",
        ),
        (
            "accelerator_idle_calibration_verified",
            "ok",
            "required_tensorrt_full_calibration_unverified",
        ),
        (
            "accelerator_idle_calibration_verified",
            1,
            "required_tensorrt_full_calibration_unverified",
        ),
        (
            "accelerator_idle_calibration_status",
            object(),
            "required_tensorrt_full_calibration_unverified",
        ),
        (
            "accelerator_idle_calibration_binding_sha256",
            object(),
            "required_tensorrt_full_calibration_unverified",
        ),
        (
            "accelerator_idle_calibration_binding_sha256",
            "A" * 64,
            "required_tensorrt_full_calibration_unverified",
        ),
        (
            "accelerator_idle_correction_requested",
            "true",
            "required_tensorrt_full_correction_unavailable",
        ),
        (
            "accelerator_idle_correction_applied",
            1,
            "required_tensorrt_full_correction_unavailable",
        ),
        (
            "accelerator_idle_correction_statuses",
            "[applied]",
            "required_tensorrt_full_correction_unavailable",
        ),
        (
            "accelerator_idle_correction_statuses",
            ("applied",),
            "required_tensorrt_full_correction_unavailable",
        ),
        (
            "energy_total_j",
            "120.0",
            "required_tensorrt_full_normalized_metrics_invalid",
        ),
        (
            "host_normalized_energy_est_j",
            "80.0",
            "required_tensorrt_full_normalized_metrics_invalid",
        ),
        (
            "energy_per_work_j",
            "0.12",
            "required_tensorrt_full_normalized_metrics_invalid",
        ),
        (
            "host_normalized_energy_per_work_est_j",
            "0.08",
            "required_tensorrt_full_normalized_metrics_invalid",
        ),
        (
            "average_power_w",
            "12.0",
            "required_tensorrt_full_normalized_metrics_invalid",
        ),
        (
            "host_normalized_average_power_est_w",
            "8.0",
            "required_tensorrt_full_normalized_metrics_invalid",
        ),
    ],
)
def test_comparison_claim_admission_rejects_nonliteral_or_noncanonical_fields(
    field: str, value: object, expected_status: str
) -> None:
    row = _verified_trt_row()
    row[field] = value
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == expected_status
    assert result["energy_comparison_claim_ready"] is False
    assert result["comparison_energy_per_work_j"] is None


@pytest.mark.parametrize(
    ("claim_value", "metric_value"),
    [
        ("yes", 0.2),
        (1, 0.2),
        (object(), 0.2),
        (True, "0.2"),
        (True, True),
        (True, object()),
    ],
)
def test_raw_comparison_preserves_legacy_diagnostics_but_not_coerced_claims(
    claim_value: object, metric_value: object
) -> None:
    result = resolve_energy_comparison(
        {
            "host_normalization_role": "none",
            "energy_efficiency_claim_eligible": claim_value,
            "energy_per_work_j": metric_value,
        }
    )
    assert result["energy_comparison_basis"] == "raw_measured"
    assert result["energy_comparison_claim_ready"] is False
    if metric_value == "0.2":
        assert result["comparison_energy_per_work_j"] == pytest.approx(0.2)


def test_comparison_rejects_the_fully_coerced_claim_row() -> None:
    result = resolve_energy_comparison(
        {
            "host_normalization_role": "tensorrt_full",
            "energy_efficiency_claim_eligible": "pass",
            "host_normalization_identity_verified": "yes",
            "host_normalization_source_run_id": "TRT_FULL",
            "host_normalization_target_variant": "FULL",
            "accelerator_idle_calibration_verified": "ok",
            "accelerator_idle_calibration_status": "verified",
            "accelerator_idle_calibration_binding_sha256": object(),
            "accelerator_idle_correction_requested": "true",
            "accelerator_idle_correction_applied": 1,
            "accelerator_idle_correction_statuses": "[applied]",
            "energy_per_work_j": "2.0",
            "host_normalized_energy_per_work_est_j": "1.0",
            "avg_power_w": "20",
            "host_normalized_average_power_est_w": "10",
        }
    )
    assert result["energy_comparison_status"] != "host_normalized_verified"
    assert result["energy_comparison_claim_ready"] is False
    assert result["comparison_energy_per_work_j"] is None


def test_exact_scalar_correction_status_remains_supported() -> None:
    row = _verified_trt_row()
    del row["accelerator_idle_correction_statuses"]
    row["accelerator_idle_correction_status"] = "applied"
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == "host_normalized_verified"
    assert result["energy_comparison_claim_ready"] is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("energy_claim_eligible", False),
        ("scientific_primary_claim_eligible", "true"),
        ("claim_eligible", 1),
    ],
)
def test_conflicting_or_coerced_claim_aliases_are_not_claim_ready(
    name: str, value: object,
) -> None:
    row = _verified_trt_row()
    row[name] = value
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == "host_normalized_verified"
    assert result["energy_comparison_claim_ready"] is False


def test_all_populated_claim_aliases_must_be_literal_true() -> None:
    row = _verified_trt_row()
    for name in (
        "energy_claim_eligible",
        "scientific_primary_claim_eligible",
        "claim_eligible",
    ):
        row[name] = True
    assert resolve_energy_comparison(row)[
        "energy_comparison_claim_ready"
    ] is True


def test_correction_status_aliases_must_be_coherent() -> None:
    row = _verified_trt_row()
    row["accelerator_idle_correction_status"] = "failed"
    failed = resolve_energy_comparison(row)
    assert failed["energy_comparison_status"] == (
        "required_tensorrt_full_correction_unavailable"
    )
    assert failed["energy_comparison_claim_ready"] is False

    row["accelerator_idle_correction_status"] = "applied"
    passed = resolve_energy_comparison(row)
    assert passed["energy_comparison_status"] == "host_normalized_verified"
    assert passed["energy_comparison_claim_ready"] is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("avg_energy_total_j", 121.0),
        ("energy_per_work_unit_j", 0.13),
        ("host_normalized_energy_per_work_unit_est_j", 0.07),
        ("avg_power_w", 13.0),
        ("avg_host_normalized_average_power_est_w", 7.0),
    ],
)
def test_active_numeric_alias_conflicts_are_rejected(
    name: str, value: float,
) -> None:
    row = _verified_trt_row()
    row[name] = value
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == (
        "required_tensorrt_full_normalized_metrics_invalid"
    )
    assert result["energy_comparison_claim_ready"] is False


def test_dual_phase_row_selects_streaming_without_equating_latency() -> None:
    row = _verified_trt_row()
    row.pop("energy_per_work_j")
    row.pop("average_power_w")
    row.pop("host_normalized_energy_per_work_est_j")
    row.pop("host_normalized_average_power_est_w")
    row.pop("active_duration_s")
    row.pop("work_units")
    row.update(
        {
            "row_energy_streaming_j_per_frame": 0.12,
            "row_host_normalized_energy_streaming_j_per_frame_est": 0.08,
            "energy_streaming_avg_power_w": 12.0,
            "host_normalized_streaming_avg_power_est_w": 8.0,
            "row_energy_latency_j_per_inference": 0.20,
            "row_host_normalized_energy_latency_j_per_inference_est": 0.15,
            "energy_latency_avg_power_w": 15.0,
            "avg_power_w": 15.0,
            "host_normalized_average_power_est_w": 11.0,
        }
    )
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == "host_normalized_verified"
    assert result["comparison_energy_per_work_j"] == pytest.approx(0.08)
    assert result["comparison_average_power_w"] == pytest.approx(8.0)


def test_cross_phase_raw_and_normalized_substitution_is_rejected() -> None:
    row = _verified_trt_row()
    row.pop("energy_per_work_j")
    row.pop("host_normalized_energy_per_work_est_j")
    row.pop("average_power_w")
    row.pop("host_normalized_average_power_est_w")
    row.pop("active_duration_s")
    row.pop("work_units")
    row.update(
        {
            "row_energy_streaming_j_per_frame": 0.12,
            "energy_streaming_avg_power_w": 12.0,
            "row_host_normalized_energy_latency_j_per_inference_est": 0.08,
            "host_normalized_average_power_est_w": 8.0,
        }
    )
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == (
        "required_tensorrt_full_normalized_metrics_invalid"
    )
    assert result["energy_comparison_claim_ready"] is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("accelerator_idle_w_applied", 2.0),
        ("accelerator_idle_w_applied", "4.0"),
        ("accelerator_idle_w_applied", True),
        ("accelerator_idle_w_applied", float("nan")),
        ("accelerator_idle_w_applied", float("inf")),
        ("host_normalized_energy_est_j", 79.0),
        ("host_normalized_energy_per_work_est_j", 0.079),
        ("host_normalized_average_power_est_w", 7.9),
    ],
)
def test_normalized_arithmetic_tampering_is_rejected(
    name: str, value: object,
) -> None:
    row = _verified_trt_row()
    row[name] = value
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == (
        "required_tensorrt_full_normalized_metrics_invalid"
    )
    assert result["energy_comparison_claim_ready"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"energy_per_work_j_sample_stddev": -0.01},
        {"host_normalized_energy_per_work_est_j_sample_stddev": -0.01},
        {"host_normalized_energy_per_work_est_j_sample_stddev": 0.01},
        {
            "host_normalized_energy_per_work_est_j_sample_stddev": 0.01,
            "host_normalized_energy_per_work_est_j_ci_low": 0.09,
            "host_normalized_energy_per_work_est_j_ci_high": 0.10,
        },
        {
            "host_normalized_energy_per_work_est_j_sample_stddev": 0.01,
            "host_normalized_energy_per_work_est_j_ci_low": 0.09,
            "host_normalized_energy_per_work_est_j_ci_high": 0.07,
        },
        {
            "host_normalized_energy_per_work_est_j_sample_stddev": True,
            "host_normalized_energy_per_work_est_j_ci_low": 0.07,
            "host_normalized_energy_per_work_est_j_ci_high": 0.09,
        },
    ],
)
def test_malformed_normalized_uncertainty_is_rejected(
    mutation: dict[str, object],
) -> None:
    row = _verified_trt_row()
    row.update(mutation)
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == (
        "required_tensorrt_full_normalized_metrics_invalid"
    )
    assert result["energy_comparison_claim_ready"] is False


def test_complete_uncertainty_triplet_is_claim_ready() -> None:
    row = _verified_trt_row()
    row.update(
        {
            "host_normalized_energy_per_work_est_j_sample_stddev": 0.01,
            "host_normalized_energy_per_work_est_j_ci_low": 0.07,
            "host_normalized_energy_per_work_est_j_ci_high": 0.09,
        }
    )
    result = resolve_energy_comparison(row)
    assert result["energy_comparison_status"] == "host_normalized_verified"
    assert result["energy_comparison_claim_ready"] is True


def test_raw_claim_requires_explicit_role_and_complete_uncertainty() -> None:
    exact = {
        "host_normalization_role": "none",
        "energy_efficiency_claim_eligible": True,
        "energy_per_work_j": 0.2,
    }
    assert resolve_energy_comparison(exact)[
        "energy_comparison_claim_ready"
    ] is True
    for role in (None, "", "NONE"):
        row = dict(exact)
        if role is None:
            row.pop("host_normalization_role")
        else:
            row["host_normalization_role"] = role
        assert resolve_energy_comparison(row)[
            "energy_comparison_claim_ready"
        ] is False

    partial = dict(exact, energy_per_work_j_sample_stddev=0.01)
    assert resolve_energy_comparison(partial)[
        "energy_comparison_claim_ready"
    ] is False

    complete = dict(
        exact,
        energy_per_work_j_sample_stddev=0.01,
        energy_per_work_j_ci_low=0.18,
        energy_per_work_j_ci_high=0.22,
    )
    assert resolve_energy_comparison(complete)[
        "energy_comparison_claim_ready"
    ] is True

    for mutation in (
        {"energy_per_work_j_sample_stddev": -0.01},
        {
            "energy_per_work_j_ci_low": 0.21,
            "energy_per_work_j_ci_high": 0.19,
        },
        {
            "energy_per_work_j_ci_low": 0.21,
            "energy_per_work_j_ci_high": 0.23,
        },
        {"energy_per_work_unit_j_sample_stddev": 0.02},
    ):
        malformed = dict(complete)
        malformed.update(mutation)
        assert resolve_energy_comparison(malformed)[
            "energy_comparison_claim_ready"
        ] is False


def test_huge_integer_fails_closed_without_overflow() -> None:
    result = resolve_energy_comparison(
        {
            "host_normalization_role": "none",
            "energy_efficiency_claim_eligible": True,
            "energy_per_work_j": 10 ** 10000,
        }
    )
    assert result["energy_comparison_claim_ready"] is False
    assert result["comparison_energy_per_work_j"] is None


def test_measure_cli_derives_role_from_separate_exact_identity(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    monkeypatch.setattr(energy_cli, "load_energy_defaults", lambda: EnergyDefaults())
    monkeypatch.setattr(
        energy_cli,
        "get_setup_energy",
        lambda _setup_id: EnergySetup(setup_id="s", enabled=True, urecs_address="192.0.2.1"),
    )
    monkeypatch.setattr(
        energy_cli,
        "run_fast_firmware_measurement",
        lambda *args, **kwargs: captured.update(kwargs) or {"ok": True},
    )
    args = SimpleNamespace(
        setup_id="s", out=str(tmp_path / "out"), workdir="", run_id="artifact-composite-id",
        command="true", command_file="", cwd=None, duration=1.0, runs=1,
        exact_run_count=False, timeout=10.0, inference_count=1, pipeline_fps=1.0,
        physical_scope="MB", window_label="command", require_runtime_work_units=False,
        require_command_window_alignment=False, compare_legacy_window=False,
        calibration_manifest="", calibration_sha256="", preflight_command="",
        preflight_command_file="", preflight_timeout_s=10.0,
        preflight_attestation_max_age_s=60.0, preflight_runtime_attestation_path="",
        preflight_expected_command_contract_sha256="", invalid_repeat_max_retries=0,
        diagnostic_only=False, claim_exclusion_reason="", window_method_ab_json="",
        host_normalization_source_run_id="native_full_tensorrt",
        host_normalization_target_variant="full",
    )
    assert energy_cli.cmd_measure(args) == 0
    assert captured["host_normalization_role"] == "tensorrt_full"
    assert captured["host_normalization_source_run_id"] == "native_full_tensorrt"
    assert captured["host_normalization_target_variant"] == "full"


def test_native_trt_pair_has_no_raw_fallback_when_normalization_missing() -> None:
    common = {
        "ok": True,
        "model": "m",
        "setup_id": "s",
        "direction": "hailo8_to_trt",
        "task": "classification",
        "energy_per_work_j": 0.1,
        "average_power_w": 10.0,
    }
    split = {**common, "backend": "hailo8_to_trt", "execution_mode": "native_split"}
    baseline = {
        **common,
        "backend": "native_full_tensorrt",
        "execution_mode": "native_full_baseline",
        "host_normalization_role": "tensorrt_full",
    }
    pair = next(
        row for row in build_native_energy_pairs([split, baseline])
        if row["baseline_kind"] == "tensorrt_full"
    )
    assert pair["baseline_kind"] == "tensorrt_full"
    assert pair["comparable"] is False
    assert "baseline_required_host_normalization_unavailable" in pair["comparison_reasons"]


def test_native_plan_and_vendored_cli_are_mirrored() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts/energy_measurement_cli.py").read_bytes() == (
        root / "onnx_splitpoint_tool/resources/remote_scripts/energy_measurement_cli.py"
    ).read_bytes()
    assert (root / "scripts/native_producer_energy_plan.py").read_bytes() == (
        root / "onnx_splitpoint_tool/resources/remote_scripts/native_producer_energy_plan.py"
    ).read_bytes()
    source = (root / "scripts/native_producer_energy_plan.py").read_text(encoding="utf-8")
    assert 'if str(backend).strip().lower() == "native_full_tensorrt"' in source
    assert '"--host-normalization-source-run-id", "native_full_tensorrt"' in source
    assert '"--host-normalization-target-variant", "full"' in source


def test_scientific_provenance_resolves_model_setup_and_keeps_diagnostic_ineligible() -> None:
    row = _scientific_row(
        {
            "model_id": "model",
            "model": "yolo11l",
            "setup_id": "orin_nx_hailo8_01",
            "run_id": "diagnostic_run",
            "diagnostic_only": True,
            "claim_eligible": False,
            "ranking_eligible": True,
            "performance_eligible": True,
            "energy_eligible": True,
            "pareto_eligible": True,
        }
    )
    assert row["model_id"] == "yolo11l"
    assert row["setup_id"] == "orin_nx_hailo8_01"
    assert row["source_run_id"] == "diagnostic_run"
    assert row["diagnostic_only"] is True
    assert row["declared_claim_eligible"] is False
    for key in (
        "ranking_eligible", "performance_eligible", "energy_eligible",
        "pareto_eligible",
    ):
        assert row[key] is False


def test_benchmark_report_replaces_literal_model_placeholder_from_exact_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_payload(**kwargs):
        captured["rows"] = list(kwargs["rows"])
        return {"rows": [], "ranking_method_macro": []}

    monkeypatch.setattr(scientific_reporting, "_build_report_payload", fake_payload)
    monkeypatch.setattr(scientific_reporting, "_write_reports", lambda *_args, **_kwargs: {})
    scientific_reporting.build_benchmarkset_scientific_report(
        tmp_path,
        [{"model_id": "model", "setup_id": "orin_nx_hailo8_01"}],
        plan={"model_id": "yolo11l"},
    )
    [enriched] = captured["rows"]
    assert enriched["model_id"] == "yolo11l"
    projected = _scientific_row(enriched)
    assert projected["model_id"] == "yolo11l"
    assert projected["setup_id"] == "orin_nx_hailo8_01"
