from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from onnx_splitpoint_tool.native_energy_quality_admission import (
    BOOLEAN_FIELDS,
    DIAGNOSTIC_FALSE_ROW_FIELDS,
    canonical_json_sha256,
    verify_sealed_energy_quality_admission,
)
from tests.test_v2727_native_energy_runtime_admission import (
    _load_energy_runner,
    _load_planner,
    _runtime_matrix_rows,
)
from tests.test_v2727_native_energy_runtime_admission_edges import (
    _runtime_row,
    _write_inputs,
)
from tests.test_v269f_variant_native_split_quality_first import (
    _binding_and_summary,
    _evalrun,
    _load_script as _load_variant_script,
    _variant_cfg,
    _write_summary,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT / "scripts"
    / "run_native_producer_energy_from_summary.py"
)


def _runtime_observation_row() -> dict[str, Any]:
    identity = {
        "backend": "deepx_to_trt",
        "model": "yolo26s",
        "case": "b026",
        "setup_id": "orin_nx_deepx_m1_01",
        "precision": "float32_layout_fp16",
        "comparison_backend": "deepx",
        "successful_command_contract_sha256": "c" * 64,
    }
    axes = {
        field: field == "diagnostic_only"
        for field in BOOLEAN_FIELDS
    }
    admission = {
        "schema": "onnx-splitpoint/native-energy-quality-admission",
        "schema_version": 1,
        "admission_scope": "native_runtime_observation",
        "runtime_observation_reason": (
            "quality_binding_unavailable"
        ),
        **identity,
        **axes,
    }
    admission["admission_sha256"] = canonical_json_sha256(admission)
    return {
        **identity,
        **axes,
        "energy_quality_admission": admission,
        "energy_quality_admission_sha256": admission[
            "admission_sha256"
        ],
        "claim_ok": False,
        "semantic_claim_ok": False,
        "claim_eligible": False,
        "eligible_for_energy_results_import": False,
        "eligible_for_scientific_claim": False,
    }


@pytest.mark.parametrize(
    "part2_input_count",
    [2, True, 1.5, "1.5"],
)
def test_unsealed_summary_count_cannot_veto_verified_part2_proof(
    monkeypatch: pytest.MonkeyPatch,
    part2_input_count: Any,
) -> None:
    planner = _load_planner()
    row = _runtime_row("b003")
    row["part2_input_count"] = part2_input_count
    monkeypatch.setattr(
        planner,
        "verify_native_split_part2_input_contract",
        lambda _contract: (
            {"inputs": [{"name": "cut"}]},
            "verified_test_single_static_part2_input",
        ),
    )
    assert planner._split_part2_input_admission(
        row, {"contract_sha256": "a" * 64},
    ) == (
        True,
        "verified_test_single_static_part2_input",
        1,
    )


def test_empty_runtime_cohort_preserves_runtime_blocked_status(
    tmp_path: Path,
) -> None:
    row = _runtime_row("b004")
    # Even a claimed count of one cannot replace the missing sealed technical
    # Part-2 proof in this deliberately incomplete subprocess fixture.
    row["part2_input_count"] = 1
    summary, validation = _write_inputs(tmp_path, row)
    out = tmp_path / "energy"

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--summary", str(summary),
            "--validation-summary", str(validation),
            "--out-dir", str(out),
            "--hailo10-ssh", "h10-test-host",
            "--duration-s", "1",
            "--measure-all-runtime-successful",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 3, (
        completed.stdout,
        completed.stderr,
    )
    report = json.loads(
        (out / "native_producer_energy_results.json").read_text(
            encoding="utf-8",
        )
    )
    assert report["status"] == (
        "blocked_no_runtime_constructible_rows"
    )
    assert report["preflight_status"] == report["status"]
    assert report["blocked_reason"] == (
        "no_runtime_constructible_native_rows"
    )
    assert report["started_measurement_count"] == 0


def test_resealed_runtime_observation_claim_tamper_is_rejected() -> None:
    row = _runtime_observation_row()

    missing_reason = dict(row)
    reasonless_admission = dict(
        missing_reason["energy_quality_admission"]
    )
    reasonless_admission.pop("admission_sha256")
    reasonless_admission.pop("runtime_observation_reason")
    reasonless_admission["admission_sha256"] = (
        canonical_json_sha256(reasonless_admission)
    )
    missing_reason.update({
        "energy_quality_admission": reasonless_admission,
        "energy_quality_admission_sha256": (
            reasonless_admission["admission_sha256"]
        ),
    })
    with pytest.raises(
        ValueError,
        match="native_runtime_observation_not_diagnostic_only",
    ):
        verify_sealed_energy_quality_admission(
            missing_reason,
            required=True,
        )

    promoted = dict(row)
    admission = dict(promoted["energy_quality_admission"])
    admission.pop("admission_sha256")
    admission["claim_comparable"] = True
    admission["energy_claim_eligible"] = True
    admission["admission_sha256"] = canonical_json_sha256(admission)
    promoted.update({
        "claim_comparable": True,
        "energy_claim_eligible": True,
        "energy_quality_admission": admission,
        "energy_quality_admission_sha256": admission[
            "admission_sha256"
        ],
    })
    with pytest.raises(
        ValueError,
        match="native_runtime_observation_not_diagnostic_only",
    ):
        verify_sealed_energy_quality_admission(
            promoted,
            required=True,
        )

    row_claim_errors = {
        **{
            field: (
                rf"diagnostic_energy_row_{field}_must_be_false"
            )
            for field in DIAGNOSTIC_FALSE_ROW_FIELDS
            if field != "energy_claim_eligible"
        },
        "energy_claim_eligible": (
            "energy_quality_admission_energy_claim_eligible_drift"
        ),
        "semantic_claim_ok": (
            "native_runtime_observation_"
            "semantic_claim_must_be_false"
        ),
    }
    for field, expected_error in row_claim_errors.items():
        with pytest.raises(
            ValueError,
            match=expected_error,
        ):
            verify_sealed_energy_quality_admission(
                {**row, field: True},
                required=True,
            )


@pytest.mark.parametrize(
    ("split_binding_available", "expected_scope"),
    [
        (False, "native_runtime_observation"),
        (True, "native_energy"),
    ],
)
def test_split_binding_controls_claim_scope_with_good_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split_binding_available: bool,
    expected_scope: str,
) -> None:
    planner = _load_planner()
    matrix = _runtime_matrix_rows()
    rows = [matrix[index] for index in (0, 3, 4)]
    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    out = tmp_path / "energy-plan"
    summary.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    digest = "e" * 64
    validation.write_text(json.dumps({"rows": [{
        **row,
        "semantic_ok": True,
        "contract_consistent": True,
        "top1_match": True,
        "claim_ok": True,
        "central_quality_evidence_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_provenance_complete": True,
        "quality_claim_result_verified": True,
        "source_request_sha256": digest,
        "model_sha256": digest,
        "validation_dataset_sha256": digest,
        "validation_dataset_image_ids_sha256": digest,
        "validation_dataset_ground_truth_sha256": digest,
        "accuracy_gate_policy_sha256": digest,
        "task_quality_policy_sha256": digest,
        "runtime_quality_gate_policy_sha256": digest,
    } for row in rows]}), encoding="utf-8")

    def verified_contract(
        raw: Any, *, expected_identity: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        return {
            "contract_sha256": str(
                (raw or {}).get("contract_sha256") or "f" * 64
            ),
            **{
                field: expected_identity[field]
                for field in (
                    "backend", "model", "case", "setup_id",
                    "comparison_backend",
                )
            },
            "runtime_options": {},
            "energy_workload": {},
            "artifacts": {},
        }, "verified_test_contract"

    monkeypatch.setattr(
        planner,
        "_energy_split_quality_authority",
        lambda _path: {
            "valid": True,
            "mode": "required",
            "native_split_quality_required": True,
            "workflow_version": "test",
            "run_id": "test",
        },
    )
    monkeypatch.setattr(
        planner, "_verify_full_command_contract", verified_contract,
    )
    monkeypatch.setattr(
        planner, "verify_native_energy_command_contract", verified_contract,
    )
    monkeypatch.setattr(
        planner, "verify_native_split_part2_input_contract",
        lambda _contract: ({"inputs": [{}]}, "verified_test_part2"),
    )
    monkeypatch.setattr(
        planner,
        "_split_quality_energy_evidence",
        lambda *_args, **_kwargs: ({
            "native_split_quality_required": (
                split_binding_available
            ),
            "native_split_energy_binding_valid": (
                split_binding_available
            ),
            "native_split_energy_binding_status": (
                "portable_join_and_semantic_payload_bytes_rehashed"
                if split_binding_available
                else "quality_binding_unavailable"
            ),
        }, (
            "portable_join_and_semantic_payload_bytes_rehashed"
            if split_binding_available
            else "quality_binding_unavailable"
        )),
    )
    monkeypatch.setattr(
        planner,
        "split_energy_runtime_argv",
        lambda *_args, **_kwargs: ["python", "split-energy.py"],
    )
    monkeypatch.setattr(
        planner,
        "_full_runtime_argv",
        lambda *_args, **_kwargs: ["python", "full-energy.py"],
    )
    monkeypatch.setattr(
        planner,
        "_split_preflight_argv",
        lambda *_args, **_kwargs: ["python", "split-preflight.py"],
    )
    monkeypatch.setattr(
        planner,
        "_full_preflight_argv",
        lambda *_args, **_kwargs: ["python", "full-preflight.py"],
    )
    monkeypatch.setattr(
        planner,
        "_process_local_runtime_environment",
        lambda _contract: {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(planner.__file__),
            "--summary", str(summary),
            "--validation-summary", str(validation),
            "--out-dir", str(out),
            "--hailo8-ssh", "hailo8-host",
            "--duration-s", "1",
            "--measure-all-runtime-successful",
        ],
    )

    assert planner.main() == 0
    payload = json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )
    split = next(
        row for row in payload["rows"]
        if row["backend"] == "hailo8_to_trt"
    )
    assert split["energy_quality_admission"][
        "admission_scope"
    ] == expected_scope
    assert split["native_split_quality_required"] is True
    assert split["native_split_energy_binding_valid"] is (
        split_binding_available
    )
    assert split["diagnostic_only"] is (
        not split_binding_available
    )
    assert split["claim_eligible"] is split_binding_available
    assert split["energy_claim_eligible"] is (
        split_binding_available
    )


def test_measure_all_runtime_successful_flag_reaches_planner_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_energy_runner()
    summary = tmp_path / "native_producer_summary.json"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    out = tmp_path / "energy"
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        commands.append([str(value) for value in command])
        return {
            "rc": 2,
            "stdout_tail": "",
            "stderr_tail": "intentional planner stop",
        }

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner.__file__),
            "--summary", str(summary),
            "--out-dir", str(out),
            "--measure-all-runtime-successful",
        ],
    )

    assert runner.main() == 2
    assert len(commands) == 1
    assert Path(commands[0][2]).name == (
        "native_producer_energy_plan.py"
    )
    assert commands[0].count(
        "--measure-all-runtime-successful"
    ) == 1


@pytest.mark.parametrize(
    ("mode", "expected_script"),
    [
        ("plan", "native_producer_energy_plan.py"),
        (
            "measure",
            "run_native_producer_energy_from_summary.py",
        ),
    ],
)
def test_variant_wrapper_forwards_and_archives_final_all_split_energy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_script: str,
) -> None:
    coordinator = _load_variant_script(
        "run_evalrun_native_producer_variants.py"
    )
    run = _evalrun(tmp_path)
    _, central_summary = _binding_and_summary(tmp_path)
    summary_path = _write_summary(run, central_summary)
    cfg = {
        **_variant_cfg(summary_path),
        "variants": [{
            "id": "final",
            "case_map": {"yolo26s": ["b038"]},
        }],
        "validation": {"enabled": True},
        "full_baselines": {
            "enabled": True,
            "backends_by_producer": {
                "hailo8": ["hailo8", "tensorrt"],
            },
        },
        "energy": {
            "enabled": True,
            "mode": mode,
        },
    }
    cfg["_workflow_context"].update({
        "campaign": {"mode": "final"},
        "execution_preset": {"id": "final"},
        "native_energy_final_contract_requested": True,
        "energy": {"final_all_split_energy": True},
    })
    config_path = tmp_path / "variants.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    reports = run / "reports"
    energy_commands: list[list[str]] = []
    producer_set = tmp_path / "trt_quality_producer_set.json"
    producer_set.write_text("{}", encoding="utf-8")

    def fake_run(
        command: list[str],
        *,
        label: str = "native-child",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        command = [str(value) for value in command]
        if label.startswith("variant:"):
            namespace = command[
                command.index("--artifact-namespace") + 1
            ]
            root = (
                run / "native_producers" / "variants" / namespace
                / "hailo8"
            )
            root.mkdir(parents=True, exist_ok=True)
        elif label == "final_report":
            (reports / "native_producer_combined_summary.json").write_text(
                json.dumps({"rows": [{
                    "backend": "hailo8_to_trt",
                    "model": "yolo26s",
                    "case": "b038",
                    "precision": "uint8_dequant_fp16",
                    "ok": True,
                }]}),
                encoding="utf-8",
            )
        elif label == "native_validation":
            validation = reports / "native_validation"
            validation.mkdir(parents=True, exist_ok=True)
            (
                validation
                / "native_producer_validation_summary.json"
            ).write_text(
                json.dumps({
                    "technical_error_count": 0,
                    "row_count": 1,
                    "rows": [{"ok": True}],
                }),
                encoding="utf-8",
            )
        elif label in {
            "native-energy:plan",
            "native-energy:measure",
        }:
            energy_commands.append(command)
            energy_out = reports / (
                "native_energy_measurements"
                if mode == "measure"
                else "native_energy_plan"
            )
            plan = (
                energy_out / "plan"
                if mode == "measure" else energy_out
            )
            plan.mkdir(parents=True, exist_ok=True)
            (
                plan / "native_producer_energy_plan.json"
            ).write_text(
                json.dumps({
                    "rows": [],
                    "preflight_status": "passed",
                    "preflight": {"status": "passed"},
                }),
                encoding="utf-8",
            )
            if mode == "measure":
                (
                    energy_out
                    / "native_producer_energy_results.json"
                ).write_text(
                    json.dumps({
                        "ok": True,
                        "complete": True,
                        "started_measurement_count": 1,
                        "rows": [],
                    }),
                    encoding="utf-8",
                )
        return {
            "cmd": command,
            "rc": 0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    monkeypatch.setattr(coordinator, "_run", fake_run)
    monkeypatch.setattr(
        coordinator,
        "_materialize_trt_quality_producer_sets",
        lambda *args, **kwargs: (
            {"hailo8_setup": str(producer_set)},
            {
                "required_setups": ["hailo8_setup"],
                "owner_by_setup": {"hailo8_setup": 0},
                "owners": [{
                    "setup_id": "hailo8_setup",
                    "variant_index": 0,
                }],
                "models": ["yolo26s"],
                "errors_by_setup": {},
            },
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_select_report_python",
        lambda *args, **kwargs: (
            sys.executable,
            {"onnxruntime_ok": True},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(coordinator.__file__),
            "--eval-run-dir", str(run),
            "--config", str(config_path),
            "--timeout", "30",
        ],
    )

    assert coordinator.main() == 0
    [energy_command] = energy_commands
    assert Path(energy_command[2]).name == expected_script
    assert "--measure-all-runtime-successful" in energy_command
    assert "--final-all-split-energy" in energy_command
    assert "--screening-energy" not in energy_command
    stage = json.loads(
        (reports / "native_producer_stage.json").read_text(
            encoding="utf-8",
        )
    )
    assert stage["native_energy"][
        "final_all_split_energy_required"
    ] is True
    assert coordinator._final_all_split_energy_requested(
        cfg,
        {"final_all_split_energy": False},
    ) is False


@pytest.mark.parametrize(
    "name",
    [
        "native_producer_energy_plan.py",
        "run_native_producer_energy_from_summary.py",
        "run_evalrun_native_producer_variants.py",
        "update_evalset_native_producers.py",
    ],
)
def test_runtime_admission_primary_and_remote_mirrors_match(
    name: str,
) -> None:
    primary = (ROOT / "scripts" / name).read_bytes()
    remote = (
        ROOT
        / "onnx_splitpoint_tool"
        / "resources"
        / "remote_scripts"
        / name
    ).read_bytes()

    assert primary == remote
    assert b"--measure-all-runtime-successful" in primary
