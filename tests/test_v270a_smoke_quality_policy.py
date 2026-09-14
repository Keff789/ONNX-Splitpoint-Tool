from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from tests.test_v269f_variant_native_split_quality_first import (
    _evalrun,
    _variant_cfg,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v270a_{path.stem}_{id(path)}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_standard_preset_overrides_conflicting_smoke_fallback_policy() -> None:
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    cfg = {
        "_workflow_context": {"execution_preset": {"id": "standard"}},
        "quality_gate_policy": {
            "diagnostic_only": True,
            "enforcement": {
                "technical_quality_error": "partial_continue_downstream",
            },
        },
    }

    assert coordinator._smoke_diagnostic_policy(cfg) is False
    assert coordinator._standard_quality_enforced_policy(cfg) is True


def test_smoke_clamp_separates_metric_warning_from_technical_error() -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    rows = [
        {
            "backend": "hailo8_to_trt",
            "native_split_quality_required": True,
            "tensor_ok": True,
            "buildable": True,
            "runtime_executable": True,
            "contract_consistent": True,
            "central_quality_evidence_verified": True,
            "accuracy_gate_decision": "fail",
            "claim_ok": True,
            "eligible_for_ranking": True,
        },
        {
            "backend": "hailo8_to_trt",
            "native_split_quality_required": True,
            "tensor_ok": True,
            "buildable": True,
            "runtime_executable": True,
            "contract_consistent": True,
            "central_quality_evidence_verified": False,
            "accuracy_gate_decision": "unavailable",
        },
        {
            "backend": "native_full_hailo10h",
            "tensor_ok": False,
            "accuracy_gate_decision": "unavailable",
            "error": "missing output dump",
        },
    ]

    validator._apply_smoke_diagnostic_policy(rows)

    assert rows[0]["ok"] is True
    assert rows[0]["status"] == "diagnostic_metric_threshold_warning"
    assert rows[0]["metric_threshold_miss_warning"] is True
    assert rows[1]["ok"] is False
    assert rows[1]["status"] == "diagnostic_technical_error"
    assert rows[1]["metric_threshold_miss_warning"] is False
    assert rows[2]["status"] == "diagnostic_technical_error"
    for row in rows:
        assert row["diagnostic_only"] is True
        assert row["claim_eligible"] is False
        assert row["claim_ok"] is False
        assert row["eligible_for_ranking"] is False
        assert row["energy_claim_eligible"] is False


def _diagnostic_join_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    request_sha = "1" * 64
    result_sha = "2" * 64
    selection_sha = "3" * 64
    identity = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "precision": "float32_layout_fp16",
        "setup_id": "hailo8_setup",
        "comparison_backend": "hailo8",
    }
    row = {
        **identity,
        "ok": True,
        "quality_evidence_verified": False,
        "performance_claim_eligible": False,
        "source_request_sha256": request_sha,
        "native_split_quality_source_request_sha256": request_sha,
        "native_split_quality_central_result_sha256": result_sha,
        "native_split_quality_selection_sha256": selection_sha,
    }
    validation = {
        **identity,
        "ok": True,
        "diagnostic_only": True,
        "claim_ok": False,
        "accuracy_gate_decision": "fail",
        "central_quality_evidence_verified": True,
        "source_request_sha256": request_sha,
        "native_split_quality_source_request_sha256": request_sha,
        "native_split_quality_central_result_sha256": result_sha,
        "native_split_quality_selection_sha256": selection_sha,
    }
    return row, validation


def test_smoke_energy_join_accepts_bound_metric_miss_but_not_technical_error() -> None:
    energy = _load_script("native_producer_energy_plan.py")
    row, validation = _diagnostic_join_fixture()

    accepted, status = energy._current_quality_first_validation_join(
        row, validation, require_performance_claim=False,
    )
    assert accepted is True
    assert status == "exact_final_validation_and_central_selection_join_verified"
    assert energy._current_quality_first_validation_join(row, validation)[0] is False

    invalid = dict(validation)
    invalid["central_quality_evidence_verified"] = False
    assert energy._current_quality_first_validation_join(
        row, invalid, require_performance_claim=False,
    ) == (
        False,
        "native_split_quality_diagnostic_central_evidence_not_verified",
    )


def test_smoke_variant_plan_removes_unbound_paths_and_forwards_policy(
    tmp_path: Path,
) -> None:
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    run = _evalrun(tmp_path)
    missing_summary = run / "quality_management" / "missing.json"
    cfg = _variant_cfg(missing_summary)
    cfg["full_baselines"] = {
        "enabled": True,
        "backends_by_producer": {"hailo8": ["hailo8", "tensorrt"]},
    }
    variants = [{"id": "smoke", "case_map": {"yolo26s": ["b038"]}}]

    split_paths, split_plan = (
        coordinator._materialize_native_split_quality_binding_sets(
            run, cfg, variants, allow_missing=True,
        )
    )
    trt_paths, trt_plan = coordinator._materialize_trt_quality_producer_sets(
        run, cfg, variants, allow_missing=True,
    )
    prepared = coordinator._quality_first_variant_plan(
        cfg, variants, trt_paths, trt_plan, split_paths, split_plan,
        allow_missing=True,
    )[0]

    assert split_plan["errors_by_variant_setup"]
    assert trt_plan["errors_by_setup"]
    assert prepared["smoke_diagnostic_quality_continue"] is True
    assert prepared["claim_eligible"] is False
    assert prepared["native_split_quality_required"] is False
    assert prepared["full_baselines"]["backends_by_producer"]["hailo8"] == [
        "hailo8",
    ]
    command = coordinator._build_update_cmd(
        run, cfg, prepared, refresh_suites=False, timeout_s=30,
    )
    assert "--smoke-diagnostic-quality-continue" in command
    assert "--native-split-quality-required" not in command
    assert "--trt-quality-producer-sets" not in command


def _native_cfg(*, smoke: bool) -> dict[str, Any]:
    return {
        "backends": ["deepx"],
        "case_policy": "case_map_only",
        "case_map": {"yolo26s": ["b038"]},
        "precision": "uint8_dequant_fp16",
        "remotes": {
            "deepx": {"ssh": "nx@deepx", "setup_id": "deepx_setup"},
        },
        "copy_benchmarksets": False,
        "build_missing_engines": False,
        "smoke_diagnostic_quality_continue": smoke,
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {"deepx": ["deepx"]},
        },
    }


def test_standard_missing_binding_is_upstream_failure_before_any_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    updater = _load_script("update_evalset_native_producers.py")
    run = _evalrun(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        updater, "_run",
        lambda command, **_kwargs: calls.append(list(command)) or {"rc": 0},
    )

    stage = updater._run_native_producers(
        run, _native_cfg(smoke=False), timeout=10,
    )

    assert stage["status"] == "failed"
    assert stage["failure_class"] == "upstream_quality_evidence"
    assert stage["failure_reason"] == "upstream_central_quality_binding_missing"
    assert stage["upstream_stage"] == "central_quality"
    assert stage["transfer_attempted"] is False
    assert calls == []
    for row in stage["backend_results"]:
        assert row["failure_class"] == "upstream_quality_evidence"
        assert row["transfer_attempted"] is False


def test_smoke_missing_split_binding_skips_split_but_runs_vendor_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    updater = _load_script("update_evalset_native_producers.py")
    run = _evalrun(tmp_path)
    labels: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        label = str(kwargs.get("label") or "")
        labels.append(label)
        assert not label.startswith("split:")
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(updater, "_run", fake_run)
    monkeypatch.setattr(updater, "_sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr(updater, "_sync_remote_package_asset_v263", lambda *a, **k: [])
    monkeypatch.setattr(
        updater, "_verify_remote_module_binding_v263",
        lambda *a, **k: {"rc": 0},
    )
    monkeypatch.setattr(
        updater, "_capture_remote_host_telemetry",
        lambda **kwargs: {"name": kwargs["phase"], "rc": 0},
    )
    monkeypatch.setattr(
        updater, "_summarize_host_telemetry", lambda *a, **k: {"rc": 0},
    )

    stage = updater._run_native_producers(
        run, _native_cfg(smoke=True), timeout=10,
    )

    assert "full:deepx" in labels
    backend = stage["backend_results"][0]
    assert backend["status"] == "partial"
    assert backend["failure_reason"] == "upstream_central_quality_binding_missing"
    assert backend["claim_eligible"] is False
    assert backend["native_full_baseline_available"] is True
    split_step = next(
        step for step in backend["steps"]
        if step.get("name") == "run_native_producer"
    )
    assert split_step["status"] == "blocked_upstream_quality"
    assert split_step["transfer_attempted"] is False


@pytest.mark.parametrize("energy_mode", ["plan", "measure"])
def test_standalone_smoke_energy_is_automatically_unpaired_and_non_claimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, energy_mode: str,
) -> None:
    updater = _load_script("update_evalset_native_producers.py")
    run = _evalrun(tmp_path)
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "native_producer_summary.json").write_text(
        json.dumps({
            "row_count": 1,
            "rows": [{
                "backend": "native_full_deepx",
                "model": "yolo26s",
                "case": "full",
                "setup_id": "deepx_setup",
                "comparison_backend": "deepx",
                "ok": True,
            }],
        }),
        encoding="utf-8",
    )
    energy_commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> dict[str, Any]:
        command_text = [str(item) for item in command]
        if any(
            Path(item).name in {
                "native_producer_energy_plan.py",
                "run_native_producer_energy_from_summary.py",
            }
            for item in command_text
        ):
            energy_commands.append(command_text)
        assert not str(kwargs.get("label") or "").startswith("split:")
        return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(updater, "_run", fake_run)
    monkeypatch.setattr(
        updater, "_sync_remote_script_v60i", lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        updater, "_sync_remote_package_asset_v263",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        updater, "_verify_remote_module_binding_v263",
        lambda *args, **kwargs: {"rc": 0},
    )
    monkeypatch.setattr(
        updater, "_capture_remote_host_telemetry",
        lambda **kwargs: {"name": kwargs["phase"], "rc": 0},
    )
    monkeypatch.setattr(
        updater, "_summarize_host_telemetry",
        lambda *args, **kwargs: {"rc": 0},
    )

    cfg = _native_cfg(smoke=True)
    cfg["energy"] = {
        "enabled": True,
        "mode": energy_mode,
        "duration_s": 1.0,
        "timeout": 1,
        # Deliberately omit allow_unpaired: Smoke must force it.
    }
    stage = updater._run_native_producers(run, cfg, timeout=10)

    assert len(energy_commands) == 1
    command = energy_commands[0]
    expected_script = (
        "native_producer_energy_plan.py"
        if energy_mode == "plan"
        else "run_native_producer_energy_from_summary.py"
    )
    assert expected_script in {Path(item).name for item in command}
    assert "--allow-unpaired" in command
    assert "--smoke-diagnostic" in command
    assert stage["diagnostic_only"] is True
    assert stage["claim_eligible"] is False
    assert stage["native_energy"]["diagnostic_only"] is True
    assert stage["native_energy"]["claim_eligible"] is False
    assert stage["native_energy"]["energy_claim_eligible"] is False
