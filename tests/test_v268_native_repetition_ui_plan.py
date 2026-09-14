from __future__ import annotations

from onnx_splitpoint_tool.execution_plan import (
    build_effective_execution_plan,
    execution_plan_text,
)
from onnx_splitpoint_tool.gui.run_mode_editor import RunModeEditDialog, _SECTIONS


def _profile(mode: str, repetitions: int, *, energy_repeats: int = 99) -> dict:
    return {
        "model_suite": {
            "primary": [
                {"id": "resnet50", "enabled": True, "task": "classification"}
            ]
        },
        "selection_policy": {"max_accepted_cases_per_model": 1},
        "run_profiles": [{"id": "hailo8", "enabled": True}],
        "execution_preset": {
            "id": mode,
            "label": mode.title(),
            "snapshot": {
                "defaults": {"native_enabled": True, "energy_enabled": True},
                "runtime": {
                    "native": {
                        "frames": 100,
                        "warmup": 10,
                        "repetitions": repetitions,
                    }
                },
                "energy": {"repeats": energy_repeats},
            },
            "overrides": {},
        },
    }


def test_editor_exposes_separate_native_performance_repetitions() -> None:
    fields = {
        path: (label, kind, default, tip)
        for _section, specs in _SECTIONS
        for label, path, kind, default, _choices, tip in specs
    }
    label, kind, default, tip = fields["runtime.native.repetitions"]
    assert label == "Native performance repetitions"
    assert kind == "int"
    assert default == 1
    assert "separate from Native energy repeats" in tip
    assert "energy.repeats" in fields


def test_editor_rejects_native_performance_repetitions_below_one() -> None:
    spec = next(
        spec
        for _section, specs in _SECTIONS
        for spec in specs
        if spec[1] == "runtime.native.repetitions"
    )

    class _Value:
        def __init__(self, value: str) -> None:
            self.value = value

        def get(self) -> str:
            return self.value

    editor = object.__new__(RunModeEditDialog)
    editor.vars = {"runtime.native.repetitions": _Value("0")}
    try:
        editor._coerce(spec)
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("zero Native performance repetitions must be rejected")


def test_execution_plan_shows_resolved_performance_repetitions_not_energy_repeats() -> None:
    plan = build_effective_execution_plan(_profile("standard", 3, energy_repeats=17))
    assert plan["native_performance_repetitions"] == 3
    rendered = execution_plan_text(plan)
    assert "performance repetitions=3" in rendered
    assert "performance repetitions=17" not in rendered


def test_execution_plan_uses_sealed_native_repetition_contract() -> None:
    profile = _profile("final", 5)
    # Historical snapshots may still carry this derived display field.  The
    # sealed Native execution contract is the sole runtime authority.
    profile["execution_preset"]["effective"] = {
        "native_performance_repetitions": 7
    }
    plan = build_effective_execution_plan(profile)
    assert plan["native_performance_repetitions"] == 5
    assert plan["native_execution_contract"]["repetitions"] == 5
    assert plan["native_execution_contract"]["field_sources"][
        "repetitions"
    ] == "mode_default"


def test_missing_values_use_smoke_and_standard_path_contract_defaults() -> None:
    expected = {"smoke": 1, "standard": 3, "final": 3}
    for mode, repetitions in expected.items():
        profile = _profile(mode, repetitions)
        del profile["execution_preset"]["snapshot"]["runtime"]["native"]["repetitions"]
        plan = build_effective_execution_plan(profile)
        assert plan["native_performance_repetitions"] == repetitions
