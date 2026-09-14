from __future__ import annotations

import pytest

from onnx_splitpoint_tool import platform_power as pp


def _analysis(
    *,
    measured_increments: tuple[float, float],
    currents: tuple[float, float] = (0.5090, 1.0090),
    voltages: tuple[float, float] = (18.49, 18.15),
    idle_powers: tuple[float, float, float] = (5.0, 5.0, 5.0),
    power_control: dict[str, float] | None = None,
) -> dict[str, object]:
    idle_before, idle_between, idle_after = idle_powers
    return pp._full_system_calibration_analysis(
        {"avg_power_w": idle_before, "capture_center_monotonic_s": 0.0},
        [
            {
                "target_current_a": 0.5,
                "reference_current_a": currents[0],
                "reference_voltage_v": voltages[0],
                "avg_power_w": (idle_before + idle_between) / 2.0
                + measured_increments[0],
                "capture_center_monotonic_s": 25.0,
            },
            {
                "target_current_a": 1.0,
                "reference_current_a": currents[1],
                "reference_voltage_v": voltages[1],
                "avg_power_w": (idle_between + idle_after) / 2.0
                + measured_increments[1],
                "capture_center_monotonic_s": 75.0,
            },
        ],
        {"avg_power_w": idle_after, "capture_center_monotonic_s": 100.0},
        idle_between={
            "avg_power_w": idle_between,
            "capture_center_monotonic_s": 50.0,
        },
        power_control=power_control,
    )


def test_real_2045_percent_point_spread_is_warning_and_saveable() -> None:
    analysis = _analysis(measured_increments=(10.2183, 20.2944))

    # The UI excerpt rounds measured increments; reproduce the displayed run
    # within that precision rather than pretending the hidden raw digits exist.
    assert analysis["scale_factor"] == pytest.approx(0.90616, abs=1e-5)
    assert analysis["point_spread_pct"] == pytest.approx(2.0451, abs=1e-3)
    gate = analysis["quality_gate"]
    assert gate["pass"] is True
    assert gate["reasons"] == []
    assert gate["warning_reasons"] == ["point_factor_spread_too_high"]
    assert analysis["warnings"]


def test_expected_factor_range_and_nominal_current_are_warnings() -> None:
    # A consistent factor remains technically usable even when it lies outside
    # the configured expectation and the actual readback is far from the two
    # nominal operator setpoints.  The actual I/V values still define truth.
    actual_currents = (0.40, 0.80)
    voltages = (19.0, 19.0)
    factor = 0.80
    increments = tuple(
        current * voltage / factor
        for current, voltage in zip(actual_currents, voltages)
    )
    analysis = _analysis(
        measured_increments=increments,
        currents=actual_currents,
        voltages=voltages,
    )

    gate = analysis["quality_gate"]
    assert gate["pass"] is True
    assert gate["reasons"] == []
    assert gate["warning_reasons"] == [
        "scale_factor_outside_configured_range",
        "reference_current_outside_target_tolerance",
    ]


def test_insufficient_signal_remains_a_hard_blocker() -> None:
    analysis = _analysis(
        measured_increments=(1.0, 20.0),
        currents=(0.05, 1.0),
        voltages=(19.0, 19.0),
    )

    gate = analysis["quality_gate"]
    assert gate["pass"] is False
    assert "measured_increment_below_minimum" in gate["reasons"]


def test_unstable_idle_baseline_remains_a_hard_blocker() -> None:
    analysis = _analysis(
        measured_increments=(10.0, 20.0),
        currents=(0.5, 1.0),
        voltages=(19.0, 19.0),
        idle_powers=(5.0, 5.7, 5.0),
    )

    gate = analysis["quality_gate"]
    assert gate["pass"] is False
    assert gate["reasons"] == ["idle_drift_too_high"]


def test_physically_implausible_factor_still_raises() -> None:
    with pytest.raises(pp.PlatformStateError, match="hard plausibility range"):
        _analysis(
            measured_increments=(30.0, 60.0),
            currents=(0.5, 1.0),
            voltages=(19.0, 19.0),
        )
