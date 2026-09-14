from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from onnx_splitpoint_tool import platform_power as pp
from onnx_splitpoint_tool.energy.full_system_gain import (
    verify_full_system_current_scale_calibration,
)


def _status(*, jetson: bool | None, m2: bool | None) -> pp.PlatformStatus:
    return pp.PlatformStatus(
        setup_id="setup",
        checked_at="now",
        urecs_address="192.0.2.10",
        urecs_host="192.0.2.10",
        urecs_port=3000,
        urecs_configured=True,
        urecs_reachable=True,
        urecs_detail="reachable",
        jetson_host="nx@example",
        jetson_configured=True,
        jetson_ssh_ready=jetson,
        jetson_detail="ready" if jetson else "not ready",
        accelerator="hailo8",
        m2_present=m2,
        m2_detail=(
            "present" if m2 is True else "absent" if m2 is False else "unknown"
        ),
    )


def _registry() -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "energy_defaults": {"physical_scope": "FS", "window_label": "command"},
        "hardware_setups": [
            {
                "id": "setup",
                "accelerator": "hailo8",
                "host": {"address": "example", "user": "nx", "port": 22},
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.10",
                    "data_port": 3000,
                },
                "power_control": {},
            }
        ],
    }


def _successful_measurement(power_w: float) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "ok",
        "avg_power_w": power_w,
        "invalid_repeat_max_retries_requested": 1,
        "invalid_repeat_max_retries": 1,
        "invalid_repeat_retry_suppressed_by_exact_run_count": False,
        "repeat_retry_attempt_count": 0,
        "repeat_retry_recovered_count": 0,
        "repeat_retry_history": [],
        "runs": [
            {
                "collector_rc": 0,
                "workload_command_rc": 0,
                "postprocess_status": "ok",
                "final_energy_gate_status": "pass",
                "final_energy_gate_reasons": [],
            }
        ],
    }


def _measurement_powers() -> dict[str, float]:
    factor = 1.02
    return {
        "idle_before": 5.0,
        "load_0.5A": 5.0 + 0.4978 * 19.0 / factor,
        "idle_between": 5.0,
        "load_1A": 5.0 + 0.9963 * 19.0 / factor,
        "idle_after": 5.0,
    }


def _label_from_runner_kwargs(kwargs: Mapping[str, Any]) -> str:
    run_id = str(kwargs.get("run_id") or "")
    for label in (
        "idle_before",
        "load_0.5A",
        "idle_between",
        "load_1A",
        "idle_after",
    ):
        if run_id.endswith("_" + label):
            return label
    raise AssertionError(f"Unexpected calibration run id: {run_id!r}")


def _operator_prompt(steps: list[str]) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        step = str(request.get("step_id") or "")
        steps.append(step)
        if step == "load_0.5A":
            return {"confirmed": True, "current_a": 0.4978, "voltage_v": 19.0}
        if step == "load_1A":
            return {"confirmed": True, "current_a": 0.9963, "voltage_v": 19.0}
        if step == "review":
            return {"save": False}
        return {"confirmed": True}

    return prompt


def _install_platform_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initially_on: bool,
) -> tuple[dict[str, Any], list[bool]]:
    source_registry = _registry()
    setup = source_registry["hardware_setups"][0]
    cfg = {
        "energy": setup["energy"],
        "power_control": dict(pp.DEFAULT_POWER_CONTROL),
    }
    state: dict[str, Any] = {
        "jetson": initially_on,
        # Accelerator presence is observable only while SSH is up.  The flow
        # must nevertheless remember, but never mutate, the initial M.2 state.
        "m2_when_on": True,
        "udp_validation_calls": 0,
    }
    jetson_transitions: list[bool] = []

    monkeypatch.setattr(
        pp,
        "resolve_setup",
        lambda setup_id, registry_path=None, registry=None: (
            source_registry if registry is None else registry,
            setup,
            cfg,
        ),
    )
    monkeypatch.setattr(
        pp, "load_hardware_registry", lambda _path=None: source_registry
    )
    udp = {
        "address": "192.0.2.10",
        "port": 3000,
        "udp_terminator": "lf",
        "jetson_command": "jetson",
        "m2_command": "m.2",
        "resolved_endpoints": (("AF_INET", "192.0.2.10"),),
    }
    def validate_udp(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        state["udp_validation_calls"] += 1
        return udp

    monkeypatch.setattr(pp, "_validate_platform_udp_configuration", validate_udp)
    monkeypatch.setattr(
        pp, "_admit_pinned_udp_preflight", lambda _s, _c, value: value
    )

    @contextmanager
    def no_lock(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", no_lock)

    def probe(*_args: Any, **_kwargs: Any) -> pp.PlatformStatus:
        jetson = bool(state["jetson"])
        return _status(
            jetson=jetson,
            m2=bool(state["m2_when_on"]) if jetson else None,
        )

    monkeypatch.setattr(pp, "probe_platform_status", probe)

    def set_jetson(
        _setup_id: str, desired: bool, **_kwargs: Any
    ) -> dict[str, Any]:
        jetson_transitions.append(bool(desired))
        changed = bool(state["jetson"]) is not bool(desired)
        state["jetson"] = bool(desired)
        return {"ok": True, "changed": changed}

    monkeypatch.setattr(pp, "_set_jetson_state_locked", set_jetson)

    def forbidden_m2_toggle(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("full-system input calibration must not toggle M.2")

    monkeypatch.setattr(pp, "_set_m2_state_locked", forbidden_m2_toggle)
    monkeypatch.setattr(
        pp,
        "_save_full_system_current_scale",
        lambda *_a, **_k: pytest.fail("test declines persistence at review"),
    )
    return state, jetson_transitions


def test_initial_jetson_off_confirmed_has_no_toggles_and_remains_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=False
    )
    steps: list[str] = []
    runner_calls: list[dict[str, Any]] = []
    powers = _measurement_powers()

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        runner_calls.append(dict(kwargs))
        return _successful_measurement(powers[_label_from_runner_kwargs(kwargs)])

    result = pp.calibrate_full_system_input_scale(
        "setup",
        output_dir=tmp_path / "initially-off",
        stabilize_s=0.0,
        measure_s=5.0,
        load_settle_s=0.0,
        operator_prompt=_operator_prompt(steps),
        measurement_runner=runner,
        confirm_jetson_already_off=True,
    )

    assert jetson_transitions == []
    assert state["jetson"] is False
    assert state["udp_validation_calls"] == 0
    assert result.restored_jetson_ready is False
    assert result.initial_state_restored is True
    assert [_label_from_runner_kwargs(call) for call in runner_calls] == [
        "idle_before",
        "load_0.5A",
        "idle_between",
        "load_1A",
        "idle_after",
    ]
    assert steps == [
        "preflight",
        "idle_before",
        "load_0.5A",
        "idle_between",
        "load_1A",
        "idle_after",
        "review",
    ]
    assert result.evidence["restoration"]["initial_state_restored"] is True
    assert result.evidence["restoration"]["m2_untouched"] is True
    assert result.evidence["restoration"]["jetson_ssh_ready"] is False


def test_initial_jetson_off_without_confirmation_aborts_before_any_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=False
    )
    prompt_steps: list[str] = []
    runner_calls: list[dict[str, Any]] = []

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        prompt_steps.append(str(request.get("step_id") or ""))
        return {"confirmed": True}

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        runner_calls.append(dict(kwargs))
        return _successful_measurement(5.0)

    with pytest.raises(pp.PlatformStateError, match="ABORT_NO_MUTATION"):
        pp.calibrate_full_system_input_scale(
            "setup",
            output_dir=tmp_path / "off-not-confirmed",
            stabilize_s=0.0,
            measure_s=5.0,
            load_settle_s=0.0,
            operator_prompt=prompt,
            measurement_runner=runner,
        )

    assert state["jetson"] is False
    assert state["udp_validation_calls"] == 0
    assert jetson_transitions == []
    assert prompt_steps == []
    assert runner_calls == []


def test_unknown_initial_jetson_state_aborts_before_any_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=False
    )
    prompt_steps: list[str] = []
    runner_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(jetson=None, m2=None),
    )

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        prompt_steps.append(str(request.get("step_id") or ""))
        return {"confirmed": True}

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        runner_calls.append(dict(kwargs))
        return _successful_measurement(5.0)

    with pytest.raises(pp.PlatformStateError, match="ABORT_NO_MUTATION"):
        pp.calibrate_full_system_input_scale(
            "setup",
            output_dir=tmp_path / "unknown-initial-state",
            stabilize_s=0.0,
            measure_s=5.0,
            load_settle_s=0.0,
            operator_prompt=prompt,
            measurement_runner=runner,
            # Confirmation only admits an observed False state. It must never
            # turn an unknown SSH result into permission for a blind toggle.
            confirm_jetson_already_off=True,
        )

    assert state["jetson"] is False
    assert state["udp_validation_calls"] == 0
    assert jetson_transitions == []
    assert prompt_steps == []
    assert runner_calls == []


def test_initial_jetson_on_shuts_down_and_restores_once_without_m2_toggle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=True
    )
    powers = _measurement_powers()

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return _successful_measurement(powers[_label_from_runner_kwargs(kwargs)])

    result = pp.calibrate_full_system_input_scale(
        "setup",
        output_dir=tmp_path / "initially-on",
        stabilize_s=0.0,
        measure_s=5.0,
        load_settle_s=0.0,
        operator_prompt=_operator_prompt([]),
        measurement_runner=runner,
    )

    assert jetson_transitions == [False, True]
    assert state["jetson"] is True
    assert state["udp_validation_calls"] == 1
    assert result.restored_jetson_ready is True
    assert result.initial_state_restored is True
    assert result.evidence["restoration"]["m2_untouched"] is True


def test_transient_marker_integrity_failure_is_retried_once_and_recovered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _state, _transitions = _install_platform_harness(
        monkeypatch, initially_on=False
    )
    powers = _measurement_powers()
    runner_calls: list[dict[str, Any]] = []

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        runner_calls.append(dict(kwargs))
        label = _label_from_runner_kwargs(kwargs)
        result = _successful_measurement(powers[label])
        if label == "load_0.5A":
            result.update(
                {
                    "invalid_repeat_max_retries_requested": 1,
                    "invalid_repeat_max_retries": 1,
                    "invalid_repeat_retry_suppressed_by_exact_run_count": False,
                    "repeat_retry_attempt_count": 1,
                    "repeat_retry_recovered_count": 1,
                    "repeat_retry_history": [
                        {
                            "logical_repeat_index": 0,
                            "recovered": True,
                            "attempts": [
                                {
                                    "attempt_index": 0,
                                    "selected": False,
                                    "retry_reasons": [
                                        "marker_dropped_samples_nonzero",
                                        "marker_trace_does_not_cover_window",
                                    ],
                                },
                                {
                                    "attempt_index": 1,
                                    "selected": True,
                                    "retry_reasons": [],
                                },
                            ],
                        }
                    ],
                }
            )
        return result

    result = pp.calibrate_full_system_input_scale(
        "setup",
        output_dir=tmp_path / "retry-recovered",
        stabilize_s=0.0,
        measure_s=5.0,
        load_settle_s=0.0,
        operator_prompt=_operator_prompt([]),
        measurement_runner=runner,
        confirm_jetson_already_off=True,
    )

    assert result.quality_passed is True
    assert all(call["exact_run_count"] is False for call in runner_calls)
    assert all(call["invalid_repeat_max_retries"] == 1 for call in runner_calls)
    assert all(
        call["require_command_window_alignment"] is True
        for call in runner_calls
    )
    measurement = result.evidence["captures"]["load_0.5A"]["measurement"]
    assert measurement["repeat_retry_attempt_count"] == 1
    assert measurement["repeat_retry_recovered_count"] == 1
    assert measurement["repeat_retry_history"][0]["recovered"] is True


def test_permanently_invalid_loaded_capture_fails_and_preserves_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=False
    )
    actual_current_a = 0.4978
    actual_voltage_v = 19.0123
    runner_calls: list[dict[str, Any]] = []

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        step = str(request.get("step_id") or "")
        if step == "load_0.5A":
            return {
                "confirmed": True,
                "current_a": actual_current_a,
                "voltage_v": actual_voltage_v,
            }
        return {"confirmed": True}

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        runner_calls.append(dict(kwargs))
        label = _label_from_runner_kwargs(kwargs)
        if label == "idle_before":
            return _successful_measurement(5.0)
        assert label == "load_0.5A"
        return {
            "ok": False,
            "status": "acquisition_integrity_failed_no_valid_runs",
            "avg_power_w": None,
            "acquisition_integrity_failure_reasons": [
                "marker_dropped_samples_nonzero",
                "marker_trace_does_not_cover_window",
            ],
            "invalid_repeat_max_retries_requested": 1,
            "invalid_repeat_max_retries": 1,
            "invalid_repeat_retry_suppressed_by_exact_run_count": False,
            "repeat_retry_attempt_count": 1,
            "repeat_retry_recovered_count": 0,
            "runs": [
                {
                    "collector_rc": 0,
                    "workload_command_rc": 0,
                    "postprocess_status": "ok",
                    "final_energy_gate_status": "fail",
                    "final_energy_gate_reasons": [
                        "raw_input_energy_provenance_unverified"
                    ],
                }
            ],
        }

    output_dir = tmp_path / "persistent-marker-failure"
    with pytest.raises(pp.PlatformPowerError) as exc_info:
        pp.calibrate_full_system_input_scale(
            "setup",
            output_dir=output_dir,
            stabilize_s=0.0,
            measure_s=5.0,
            load_settle_s=0.0,
            operator_prompt=prompt,
            measurement_runner=runner,
            confirm_jetson_already_off=True,
        )

    assert jetson_transitions == []
    assert state["jetson"] is False
    assert all(call["exact_run_count"] is False for call in runner_calls)
    assert all(call["invalid_repeat_max_retries"] == 1 for call in runner_calls)
    assert all(
        call["require_command_window_alignment"] is True
        for call in runner_calls
    )

    operational_path = Path(exc_info.value.evidence_path)
    assert operational_path == (
        output_dir / "full_system_input_scale_calibration_operational.json"
    )
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    assert operational["status"] == "failed"
    assert operational["initial_state_restored"] is True
    failed = operational["captures"]["load_0.5A"]
    assert failed["status"] == "failed"
    assert failed["target_current_a"] == pytest.approx(0.5)
    assert failed["reference_current_a"] == pytest.approx(actual_current_a)
    assert failed["reference_voltage_v"] == pytest.approx(actual_voltage_v)
    assert failed["reference_power_w"] == pytest.approx(
        actual_current_a * actual_voltage_v
    )
    assert failed["measurement"]["acquisition_integrity_failure_reasons"] == [
        "marker_dropped_samples_nonzero",
        "marker_trace_does_not_cover_window",
    ]
    assert "raw_input_energy_provenance_unverified" in failed["error"]
    assert not (
        output_dir / "full_system_input_scale_calibration.json"
    ).exists()


def test_initially_off_saved_evidence_verifies_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=False
    )
    powers = _measurement_powers()
    save_calls: list[dict[str, Any]] = []

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        step = str(request.get("step_id") or "")
        if step == "load_0.5A":
            return {"confirmed": True, "current_a": 0.4978, "voltage_v": 19.0}
        if step == "load_1A":
            return {"confirmed": True, "current_a": 0.9963, "voltage_v": 19.0}
        if step == "review":
            return {"save": True}
        return {"confirmed": True}

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return _successful_measurement(
            powers[_label_from_runner_kwargs(kwargs)]
        )

    def save_without_touching_real_registry(
        setup_id: str,
        factor: float,
        **kwargs: Any,
    ) -> bool:
        save_calls.append(
            {
                "setup_id": setup_id,
                "factor": factor,
                **kwargs,
            }
        )
        return False

    monkeypatch.setattr(
        pp,
        "_save_full_system_current_scale",
        save_without_touching_real_registry,
    )

    result = pp.calibrate_full_system_input_scale(
        "setup",
        output_dir=tmp_path / "initially-off-saved",
        stabilize_s=0.0,
        measure_s=5.0,
        load_settle_s=0.0,
        operator_prompt=prompt,
        measurement_runner=runner,
        confirm_jetson_already_off=True,
    )

    assert result.saved is True
    assert result.quality_passed is True
    assert result.initial_state_restored is True
    assert state["jetson"] is False
    assert jetson_transitions == []
    assert len(save_calls) == 1
    save_record = save_calls[0]["calibration_record"]
    assert save_record["evidence_path"] == result.evidence_path
    assert save_record["evidence_sha256"] == result.evidence_sha256

    verification = verify_full_system_current_scale_calibration(
        {
            "setup_id": "setup",
            "accelerator": "hailo8",
            "urecs_address": "192.0.2.10",
            "data_port": 3000,
            "full_system_current_scale_factor": result.scale_factor,
            "full_system_current_scale_calibrated_at": result.finished_at,
            "full_system_current_scale_calibration_evidence": (
                result.evidence_path
            ),
            "full_system_current_scale_calibration_sha256": (
                result.evidence_sha256
            ),
        },
        physical_scope="FS",
    )

    assert verification["full_system_current_scale_verified"] is True
    assert verification["full_system_current_scale_verification_status"] == (
        "verified"
    )
    assert verification["full_system_current_scale_verification_errors"] == []


def test_plausibility_warning_does_not_block_save_or_evidence_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=False
    )
    powers = {
        "idle_before": 5.0,
        "load_0.5A": 15.2183,
        "idle_between": 5.0,
        "load_1A": 25.2944,
        "idle_after": 5.0,
    }
    save_calls: list[dict[str, Any]] = []

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        step = str(request.get("step_id") or "")
        if step == "load_0.5A":
            return {"confirmed": True, "current_a": 0.5090, "voltage_v": 18.49}
        if step == "load_1A":
            return {"confirmed": True, "current_a": 1.0090, "voltage_v": 18.15}
        if step == "review":
            assert request["quality_passed"] is True
            assert request["quality_reasons"] == []
            assert request["warnings"]
            return {"save": True}
        return {"confirmed": True}

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return _successful_measurement(powers[_label_from_runner_kwargs(kwargs)])

    def save_without_touching_real_registry(
        setup_id: str,
        factor: float,
        **kwargs: Any,
    ) -> bool:
        save_calls.append({"setup_id": setup_id, "factor": factor, **kwargs})
        return False

    monkeypatch.setattr(
        pp,
        "_save_full_system_current_scale",
        save_without_touching_real_registry,
    )

    result = pp.calibrate_full_system_input_scale(
        "setup",
        output_dir=tmp_path / "warning-still-saved",
        stabilize_s=0.0,
        measure_s=5.0,
        load_settle_s=0.0,
        operator_prompt=prompt,
        measurement_runner=runner,
        confirm_jetson_already_off=True,
    )

    assert result.saved is True
    assert result.quality_passed is True
    assert result.scale_factor == pytest.approx(0.90616, abs=1e-5)
    assert result.evidence["quality_gate"]["reasons"] == []
    assert result.evidence["quality_gate"]["warning_reasons"] == [
        "point_factor_spread_too_high"
    ]
    assert result.evidence["warnings"]
    assert len(save_calls) == 1
    assert state["jetson"] is False
    assert jetson_transitions == []

    verification = verify_full_system_current_scale_calibration(
        {
            "setup_id": "setup",
            "accelerator": "hailo8",
            "urecs_address": "192.0.2.10",
            "data_port": 3000,
            "full_system_current_scale_factor": result.scale_factor,
            "full_system_current_scale_calibrated_at": result.finished_at,
            "full_system_current_scale_calibration_evidence": result.evidence_path,
            "full_system_current_scale_calibration_sha256": result.evidence_sha256,
        },
        physical_scope="FS",
    )
    assert verification["full_system_current_scale_verified"] is True


def test_technical_invalidity_cannot_save_even_if_review_requests_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=False
    )
    powers = {
        "idle_before": 5.0,
        "load_0.5A": 6.0,
        "idle_between": 5.0,
        "load_1A": 25.0,
        "idle_after": 5.0,
    }
    save_calls: list[dict[str, Any]] = []

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        step = str(request.get("step_id") or "")
        if step == "load_0.5A":
            return {"confirmed": True, "current_a": 0.05, "voltage_v": 19.0}
        if step == "load_1A":
            return {"confirmed": True, "current_a": 1.0, "voltage_v": 19.0}
        if step == "review":
            assert request["quality_passed"] is False
            assert "measured_increment_below_minimum" in request[
                "quality_reasons"
            ]
            return {"save": True}
        return {"confirmed": True}

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return _successful_measurement(powers[_label_from_runner_kwargs(kwargs)])

    monkeypatch.setattr(
        pp,
        "_save_full_system_current_scale",
        lambda *args, **kwargs: save_calls.append({
            "args": args, "kwargs": kwargs,
        }),
    )

    with pytest.raises(pp.PlatformStateError, match="quality gate failed"):
        pp.calibrate_full_system_input_scale(
            "setup",
            output_dir=tmp_path / "technical-invalid-save-request",
            stabilize_s=0.0,
            measure_s=5.0,
            load_settle_s=0.0,
            operator_prompt=prompt,
            measurement_runner=runner,
            confirm_jetson_already_off=True,
        )

    assert save_calls == []
    assert state["jetson"] is False
    assert jetson_transitions == []


def test_ambiguous_restore_never_sends_a_second_toggle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state, jetson_transitions = _install_platform_harness(
        monkeypatch, initially_on=True
    )
    powers = _measurement_powers()

    def ambiguous_set_jetson(
        _setup_id: str, desired: bool, **_kwargs: Any
    ) -> dict[str, Any]:
        jetson_transitions.append(bool(desired))
        if desired is False:
            state["jetson"] = False
            return {"ok": True, "changed": True}
        # Model an unacknowledged toggle that may have been consumed while SSH
        # still appears down during boot. A second toggle would be unsafe.
        raise pp.PlatformStateError(
            "Jetson did not become SSH-ready after toggle send"
        )

    monkeypatch.setattr(pp, "_set_jetson_state_locked", ambiguous_set_jetson)

    def runner(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return _successful_measurement(
            powers[_label_from_runner_kwargs(kwargs)]
        )

    output_dir = tmp_path / "ambiguous-restore"
    with pytest.raises(pp.PlatformStateError):
        pp.calibrate_full_system_input_scale(
            "setup",
            output_dir=output_dir,
            stabilize_s=0.0,
            measure_s=5.0,
            load_settle_s=0.0,
            operator_prompt=_operator_prompt([]),
            measurement_runner=runner,
        )

    assert jetson_transitions == [False, True]
    operational = json.loads(
        (
            output_dir
            / "full_system_input_scale_calibration_operational.json"
        ).read_text(encoding="utf-8")
    )
    assert operational["restoration_attempted"] is True
    assert operational["initial_state_restored"] is False
    assert operational["recovery"]["status"] == (
        "restore_attempt_ambiguous_manual_verification_required"
    )
    assert operational["recovery"]["automated_restore_retry_suppressed"] is True
    assert operational["recovery_skipped_reason"] == (
        "restore_command_may_have_been_sent_manual_verification_required"
    )
