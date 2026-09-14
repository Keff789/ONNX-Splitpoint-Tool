from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import sys
from typing import Any, Callable

import pytest

import onnx_splitpoint_tool.resume_cohort_preflight as cohort
import onnx_splitpoint_tool.resume_preparation as preparation
from onnx_splitpoint_tool.native_energy_reporting import (
    _energy_payload,
    collect_native_energy,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_native_producer_energy_from_summary.py"
HAILO8_RESUME_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "v2725_hailo8_resume"
    / (
        "hailo8_to_trt__resnet50__b052__orin_nx_hailo8_01__"
        "8eb40a232207.command_contract.json"
    )
)
HAILO8_RESUME_FIXTURE_FILE_SHA256 = (
    "4e5bac737c9a071eee2bf218c3f76e996f69dd0183c7796bdca088c8c91aaba6"
)
HAILO8_RESUME_FIXTURE_LOGICAL_SHA256 = (
    "8eb40a2322078272add157c50b41309f2a5b43be5dff16eae773eb339f106a8b"
)
HAILO8_RESUME_REMOTE_ROOT = (
    "/home/nx/native_fifo_evalsets/"
    "resnet_yolo26s_yolo7_20260729_214045"
)

_EXECUTION_CONTEXT = {
    "remote_root": "/remote/run",
    "remote_tool_dir": "/remote/tool",
    "hailo8_ssh": "nx@hailo8",
    "hailo10_ssh": "nx@hailo10",
    "deepx_ssh": "nx@deepx",
    "hailo8_env": "source /env/hailo8/bin/activate",
    "hailo10_env": "source /env/hailo10/bin/activate",
    "deepx_env": "source /env/deepx/bin/activate",
    "engine_build_python": "auto",
}


def _namespace() -> dict[str, Any]:
    return runpy.run_path(str(SCRIPT))


def _command(
    out: Path, *, setup_id: str, run_id: str, requested_runs: int = 3,
    command_file: Path, preflight_file: Path, attestation_path: str,
) -> str:
    return (
        "python energy_measurement_cli.py measure "
        f"--setup-id {shlex.quote(setup_id)} "
        f"--run-id {shlex.quote(run_id)} "
        f"--command-file {shlex.quote(str(command_file))} "
        f"--runs {requested_runs} "
        f"--out {shlex.quote(str(out))} "
        f"--preflight-command-file {shlex.quote(str(preflight_file))} "
        "--preflight-runtime-attestation-path "
        f"{shlex.quote(attestation_path)} "
        "--command 'runner --out remote.json --runs 99'"
    )


def _aggregate(
    out: Path, *, setup_id: str, run_id: str, valid: int = 3,
    requested_runs: int = 3,
) -> dict[str, Any]:
    complete = valid == 3
    return {
        "ok": complete,
        "status": "ok" if complete else "incomplete_valid_repeats",
        "setup_id": setup_id,
        "run_id": run_id,
        "out_dir": str(out.resolve()),
        "run_count": 3,
        "requested_valid_repeat_count": 3,
        "materialized_logical_repeat_count": 3,
        "valid_postprocessed_runs": valid,
        "scientific_primary_valid_run_count": valid,
        "scientific_primary_method_frozen": True,
        "scientific_primary_method": "command_marker_window",
        "scientific_primary_energy_status": (
            "available" if complete else "unavailable_incomplete_valid_repeats"
        ),
        "repeat_contract_complete": complete,
        "energy_window_method_ab": {
            "requested_run_count": requested_runs,
            "effective_run_count": 3,
            "scientific_primary_method": "command_marker_window",
        },
        "runs": [
            {
                "run_index": index,
                # A successful last repeat must never overwrite aggregate 2/3.
                "valid_postprocessed_runs": 3,
                "postprocess_status": "ok",
                "final_energy_gate_status": "pass",
            }
            for index in range(3)
        ],
        "scientific_primary_energy_statistics": {
            "energy_j": {"n": valid, "mean": 10.0},
        },
        "legacy_window_comparison_requested": True,
        "legacy_window_comparison_attempted_runs": 3,
        "legacy_window_comparison_successful_runs": valid,
        "window_method_comparison_statistics": {
            "energy_j": {
                "relative_percent": {"n": valid, "mean": 1.0},
            },
        },
        "postprocess_status": (
            "ok" if complete else "incomplete_valid_repeats"
        ),
        "final_energy_gate_status": "pass" if complete else "fail",
    }


def _plan_row(
    base: Path, *, backend: str, model: str, case: str, setup_id: str,
    attempt: str,
) -> dict[str, Any]:
    run_id = f"native_{backend}_{model}_{case}_{setup_id}"
    command_dir = base.parent / "plan_commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{backend}__{model}__{case}__{setup_id}"
    contract_file = command_dir / f"{stem}.command_contract.json"
    command_file = command_dir / f"{stem}.sh"
    preflight_file = command_dir / f"{stem}.preflight.sh"
    contract_file.write_text('{"contract":"frozen"}\n', encoding="utf-8")
    if "hailo8" in backend:
        ssh = _EXECUTION_CONTEXT["hailo8_ssh"]
        remote_env = _EXECUTION_CONTEXT["hailo8_env"]
    elif "deepx" in backend:
        ssh = _EXECUTION_CONTEXT["deepx_ssh"]
        remote_env = _EXECUTION_CONTEXT["deepx_env"]
    else:
        ssh = _EXECUTION_CONTEXT["hailo10_ssh"]
        remote_env = _EXECUTION_CONTEXT["hailo10_env"]
    remote_base = (
        f"{_EXECUTION_CONTEXT['remote_root']}/.energy_replays/{stem}"
    )
    remote_contract = f"{remote_base}/command_contract.json"
    remote_command = (
        f"{remote_env} && mkdir -p {remote_base} && "
        f"cd {_EXECUTION_CONTEXT['remote_tool_dir']} && "
        f"runner --contract {remote_contract} --out __FRESH_OUTPUT__"
    )
    command_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"ssh {ssh} {shlex.quote(remote_command)}\n",
        encoding="utf-8",
    )
    remote_preflight = (
        f"mkdir -p {remote_base} && {remote_env} && "
        f"cd {_EXECUTION_CONTEXT['remote_tool_dir']} && "
        f"preflight --contract {remote_contract}"
    )
    preflight_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"ssh {ssh} {shlex.quote(remote_preflight)}"
        f" < {shlex.quote(str(contract_file))}\n",
        encoding="utf-8",
    )
    return {
        "backend": backend,
        "model": model,
        "case": case,
        "setup_id": setup_id,
        "comparison_backend": {
            "hailo8_to_trt": "hailo8",
            "hailo10h_to_trt": "hailo10h",
            "deepx_to_trt": "deepx",
        }.get(backend, backend),
        "precision": "fp16",
        "successful_command_contract_sha256": "a" * 64,
        "measurement_output_dir": str(base),
        "measurement_output_base_dir": str(base),
        "measurement_plan_attempt_id": attempt,
        "measurement_setup_id": setup_id,
        "measurement_run_id": run_id,
        "measurement_requested_repeats": 3,
        "measure_command": _command(
            base, setup_id=setup_id, run_id=run_id,
            command_file=command_file,
            preflight_file=preflight_file,
            attestation_path=(
                f"{remote_base}/"
                "preflight___ONNX_SPLITPOINT_PREFLIGHT_NONCE__.json"
            ),
        ),
        "command_file": str(command_file),
        "command_contract_file": str(contract_file),
        "command_contract_file_sha256": hashlib.sha256(
            contract_file.read_bytes()
        ).hexdigest(),
        "remote_command_contract_file": remote_contract,
        "preflight_command_file": str(preflight_file),
        "preflight_runtime_attestation_path_template": (
            f"{remote_base}/preflight___ONNX_SPLITPOINT_PREFLIGHT_NONCE__.json"
        ),
        "window_method_ab": {
            "requested_run_count": 3,
            "effective_run_count": 3,
        },
        "energy_scope": "MB",
        "energy_window": "command",
        "energy_evidence_tier": "smoke_diagnostic",
        "energy_tier": "smoke_diagnostic",
        "smoke_diagnostic": True,
    }


def _plan(rows: list[dict[str, Any]], *, attempt: str) -> dict[str, Any]:
    return {
        "schema": "onnx_splitpoint_native_energy_plan_v60i",
        "schema_version": 1,
        "execution_context": {
            "schema": "onnx-splitpoint/native-energy-execution-context",
            "schema_version": 1,
            **_EXECUTION_CONTEXT,
        },
        "energy_runs_per_row": 3,
        "measurement_plan_attempt_id": attempt,
        "window_method_ab": {
            "requested_run_count": 3,
            "effective_run_count": 3,
        },
        "physical_scope": "MB",
        "window_label": "command",
        "energy_evidence_tier": "smoke_diagnostic",
        "energy_tier": "smoke_diagnostic",
        "smoke_diagnostic": True,
        "diagnostic_only": True,
        "preflight_status": "passed",
        "technical_measurement_contract_valid": True,
        "preflight": {
            "status": "passed",
            "ok": True,
            "measurement_start_allowed": True,
            "technical_measurement_contract_valid": True,
            "energy_plan_coverage_contract_valid": True,
        },
        "rows": rows,
    }


def _complete_result(row: dict[str, Any], *, marker: str) -> dict[str, Any]:
    aggregate = _aggregate(
        Path(row["measurement_output_dir"]),
        setup_id=row["setup_id"],
        run_id=row["measurement_run_id"],
    )
    return {
        "row": copy.deepcopy(row),
        "ok": True,
        "preserved_marker": marker,
        "run": {
            "rc": 0,
            "energy_aggregate": aggregate,
            "energy_aggregate_embedded": True,
            "energy_aggregate_bound": True,
            "energy_aggregate_complete": True,
            "energy_aggregate_verified": True,
        },
    }


def _failed_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row": copy.deepcopy(row),
        "ok": False,
        "run": {"rc": 1, "energy_aggregate_verified": False},
    }


def _resume_fixture(tmp_path: Path) -> tuple[
    Path, Path, dict[str, Any], dict[str, Any], list[str],
]:
    out = tmp_path / "native_energy_measurements"
    old_attempt = "old-plan"
    identities = [
        ("hailo8_to_trt", "resnet50", "b039", "setup-a"),
        ("hailo10h_to_trt", "yolo26s", "b038", "setup-b"),
        ("native_full_hailo10h", "yolov7_paper", "full", "setup-b"),
    ]
    old_rows = [
        _plan_row(
            out / "measurements" / f"old-{index}",
            backend=backend, model=model, case=case, setup_id=setup,
            attempt=old_attempt,
        )
        for index, (backend, model, case, setup) in enumerate(identities)
    ]
    old_plan = _plan(old_rows, attempt=old_attempt)
    old_results = [
        _complete_result(old_rows[0], marker="must-stay-byte-equivalent"),
        _failed_result(old_rows[1]),
        _failed_result(old_rows[2]),
    ]
    existing = {
        "ok": False,
        "complete": True,
        "status": "failed_measurement_or_aggregate_contract",
        "smoke_diagnostic": True,
        "diagnostic_only": True,
        "plan": {
            "cmd": [
                "/venv/bin/python", "-u", "/tool/energy_plan.py",
                "--remote-root", _EXECUTION_CONTEXT["remote_root"],
                "--remote-tool-dir", _EXECUTION_CONTEXT["remote_tool_dir"],
                "--hailo8-ssh", _EXECUTION_CONTEXT["hailo8_ssh"],
                "--hailo10-ssh", _EXECUTION_CONTEXT["hailo10_ssh"],
                "--deepx-ssh", _EXECUTION_CONTEXT["deepx_ssh"],
                "--hailo8-env", _EXECUTION_CONTEXT["hailo8_env"],
                "--hailo10-env", _EXECUTION_CONTEXT["hailo10_env"],
                "--deepx-env", _EXECUTION_CONTEXT["deepx_env"],
                "--engine-build-python",
                _EXECUTION_CONTEXT["engine_build_python"],
            ],
        },
        "plan_payload": old_plan,
        "rows": old_results,
    }
    out.mkdir(parents=True, exist_ok=True)
    canonical = out / "native_producer_energy_results.json"
    canonical.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    (out / "native_producer_energy_results.md").write_text(
        "old markdown\n", encoding="utf-8",
    )
    (out / "plan").mkdir()
    (out / "plan" / "native_producer_energy_plan.json").write_text(
        json.dumps(old_plan, indent=2), encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    selectors = [
        "hailo10h_to_trt|yolo26s|b038|setup-b",
        "native_full_hailo10h|yolov7_paper|full|setup-b",
    ]
    return out, summary, existing, old_plan, selectors


def _bind_archived_hailo8_contract(
    row: dict[str, Any],
    *,
    plan_root: Path,
) -> dict[str, Any]:
    fixture = HAILO8_RESUME_FIXTURE.read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == (
        HAILO8_RESUME_FIXTURE_FILE_SHA256
    )
    payload = json.loads(fixture)
    assert payload["contract_sha256"] == (
        HAILO8_RESUME_FIXTURE_LOGICAL_SHA256
    )
    for field in ("backend", "model", "case", "setup_id"):
        assert row[field] == payload[field]

    plan_root.mkdir(parents=True, exist_ok=True)
    stem = "archived_real_hailo8"
    contract_file = plan_root / f"{stem}.command_contract.json"
    command_file = plan_root / f"{stem}.sh"
    preflight_file = plan_root / f"{stem}.preflight.sh"
    contract_file.write_bytes(fixture)
    command_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "run-workload "
        "--source-contract-sha256 "
        f"{HAILO8_RESUME_FIXTURE_LOGICAL_SHA256} "
        "--nonce __ONNX_SPLITPOINT_PREFLIGHT_NONCE__ "
        "--attestation __ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__\n",
        encoding="utf-8",
    )
    preflight_file.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "run-preflight "
        f"--contract-json {shlex.quote(str(contract_file))} "
        "--contract-file-sha256 "
        f"{HAILO8_RESUME_FIXTURE_FILE_SHA256} "
        "--nonce __ONNX_SPLITPOINT_PREFLIGHT_NONCE__ "
        "--attestation __ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__\n",
        encoding="utf-8",
    )

    updated = dict(row)
    updated.update({
        "comparison_backend": "hailo8",
        "successful_command_contract_sha256": (
            HAILO8_RESUME_FIXTURE_LOGICAL_SHA256
        ),
        "command_file": str(command_file),
        "command_contract_file": str(contract_file),
        "command_contract_file_sha256": (
            HAILO8_RESUME_FIXTURE_FILE_SHA256
        ),
        "preflight_command_file": str(preflight_file),
    })
    updated["measure_command"] = _command(
        Path(updated["measurement_output_dir"]),
        setup_id=updated["setup_id"],
        run_id=updated["measurement_run_id"],
        requested_runs=updated["measurement_requested_repeats"],
        command_file=command_file,
        preflight_file=preflight_file,
        attestation_path=updated[
            "preflight_runtime_attestation_path_template"
        ],
    )
    return updated


def _single_hailo8_resume_fixture(tmp_path: Path) -> tuple[
    Path, Path, dict[str, Any], dict[str, Any], list[str],
]:
    out = tmp_path / "EvaluationRun" / "reports" / (
        "native_energy_measurements"
    )
    plan_root = out / "plan"
    plan_root.mkdir(parents=True)
    identities = [
        (
            "hailo8_to_trt",
            "resnet50",
            "b052",
            "orin_nx_hailo8_01",
        ),
        (
            "native_full_hailo8",
            "resnet50",
            "full",
            "orin_nx_hailo8_01",
        ),
        (
            "native_full_tensorrt",
            "resnet50",
            "full",
            "orin_nx_hailo8_01",
        ),
    ]
    old_rows = [
        _plan_row(
            out / "measurements" / f"old-{index}",
            backend=backend,
            model=model,
            case=case,
            setup_id=setup,
            attempt="old-plan",
        )
        for index, (backend, model, case, setup) in enumerate(identities)
    ]
    for row in old_rows:
        row["comparison_backend"] = "hailo8"
    old_rows[0] = _bind_archived_hailo8_contract(
        old_rows[0],
        plan_root=plan_root,
    )
    old_plan = _plan(old_rows, attempt="old-plan")
    existing = {
        "ok": False,
        "complete": True,
        "status": "failed_measurement_or_aggregate_contract",
        "smoke_diagnostic": True,
        "diagnostic_only": True,
        "plan": {
            "cmd": [
                "/venv/bin/python",
                "-u",
                "/tool/energy_plan.py",
                "--remote-root",
                _EXECUTION_CONTEXT["remote_root"],
                "--remote-tool-dir",
                _EXECUTION_CONTEXT["remote_tool_dir"],
                "--hailo8-ssh",
                _EXECUTION_CONTEXT["hailo8_ssh"],
                "--hailo10-ssh",
                _EXECUTION_CONTEXT["hailo10_ssh"],
                "--deepx-ssh",
                _EXECUTION_CONTEXT["deepx_ssh"],
                "--hailo8-env",
                _EXECUTION_CONTEXT["hailo8_env"],
                "--hailo10-env",
                _EXECUTION_CONTEXT["hailo10_env"],
                "--deepx-env",
                _EXECUTION_CONTEXT["deepx_env"],
                "--engine-build-python",
                _EXECUTION_CONTEXT["engine_build_python"],
            ],
        },
        "plan_payload": old_plan,
        "rows": [
            _failed_result(old_rows[0]),
            _complete_result(old_rows[1], marker="reuse-full-hailo8"),
            _complete_result(
                old_rows[2],
                marker="reuse-full-tensorrt",
            ),
        ],
    }
    canonical = out / "native_producer_energy_results.json"
    canonical.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    (out / "native_producer_energy_results.md").write_text(
        "old markdown\n",
        encoding="utf-8",
    )
    (plan_root / "native_producer_energy_plan.json").write_text(
        json.dumps(old_plan, indent=2),
        encoding="utf-8",
    )
    summary = out.parent / "native_producer_summary.json"
    summary.write_text('{"rows":[]}\n', encoding="utf-8")
    selector = (
        "hailo8_to_trt|resnet50|b052|orin_nx_hailo8_01"
    )
    return out, summary, existing, old_plan, [selector]


def _install_fake_runner(
    namespace: dict[str, Any],
    *, out: Path,
    old_plan: dict[str, Any],
    fail_selector: str = "",
    row_transform: (
        Callable[[dict[str, Any], int, Path], dict[str, Any]] | None
    ) = None,
    stub_resume_preparation: bool = True,
) -> list[str]:
    measurement_calls: list[str] = []

    def fake_run(
        cmd: list[str], timeout: int | None = None, label: str = "",
    ) -> dict[str, Any]:
        del timeout
        if label == "energy_plan":
            plan_dir = Path(cmd[cmd.index("--out-dir") + 1])
            attempt = "fresh-plan"
            fresh_rows = []
            for index, old_row in enumerate(old_plan["rows"]):
                identity = (
                    old_row["backend"], old_row["model"],
                    old_row["case"], old_row["setup_id"],
                )
                fresh_row = _plan_row(
                    plan_dir.parent / "measurements" / f"row-{index}",
                    backend=identity[0], model=identity[1],
                    case=identity[2], setup_id=identity[3],
                    attempt=attempt,
                )
                if row_transform is not None:
                    fresh_row = row_transform(
                        fresh_row, index, plan_dir,
                    )
                fresh_rows.append(fresh_row)
            payload = _plan(fresh_rows, attempt=attempt)
            payload.pop("execution_context")
            plan_dir.mkdir(parents=True, exist_ok=True)
            (plan_dir / "native_producer_energy_plan.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8",
            )
            return {"rc": 0, "elapsed_s": 0.01}

        command = shlex.join(cmd)
        actual_out = Path(namespace["_command_option"](command, "--out"))
        setup = namespace["_command_option"](command, "--setup-id")
        run_id = namespace["_command_option"](command, "--run-id")
        selector = next(
            (
                "|".join((
                    row["backend"], row["model"], row["case"], row["setup_id"],
                ))
                for row in old_plan["rows"]
                if row["measurement_run_id"] == run_id
            ),
            "",
        )
        measurement_calls.append(selector)
        valid = 2 if selector == fail_selector else 3
        aggregate = _aggregate(
            actual_out, setup_id=setup, run_id=run_id, valid=valid,
        )
        (actual_out / "energy_aggregate.json").write_text(
            json.dumps(aggregate), encoding="utf-8",
        )
        return {"rc": 1 if valid < 3 else 0, "elapsed_s": 0.01}

    namespace["main"].__globals__["_run"] = fake_run
    if stub_resume_preparation:
        namespace["main"].__globals__[
            "_prepare_resume_measurement_cohort"
        ] = lambda *args, **kwargs: {
            "ok": True,
            "status": "ready_for_measurement",
            "measurement_wrapper_allowed": True,
            "report_path": str(out / "fake_resume_preparation.json"),
            "cohort_preflight": {
                "ok": True,
                "measurement_wrapper_allowed": True,
            },
        }
    return measurement_calls


def _resume_argv(
    out: Path, summary: Path, selectors: list[str],
) -> list[str]:
    argv = [
        str(SCRIPT),
        "--summary", str(summary),
        "--out-dir", str(out),
        "--smoke-diagnostic",
        "--runs", "3",
        "--physical-scope", "MB",
        "--window-label", "command",
        "--resume-existing",
        "--expected-selected-rows", str(len(selectors)),
    ]
    for selector in selectors:
        argv.extend(["--only-row", selector])
    return argv


def test_authoritative_aggregate_projection_never_uses_nested_repeat_status() -> None:
    aggregate = _aggregate(
        Path("/tmp/aggregate"), setup_id="setup-a", run_id="run-a", valid=2,
    )
    payload = _energy_payload({
        "run": {
            "rc": 1,
            "stdout_tail": json.dumps(aggregate["runs"][-1]),
            "energy_aggregate": aggregate,
            "energy_aggregate_embedded": True,
            "energy_aggregate_bound": True,
            "energy_aggregate_complete": False,
            "energy_aggregate_verified": False,
        },
    })

    assert payload["valid_postprocessed_runs"] == 2
    assert payload["postprocess_status"] == "incomplete_valid_repeats"
    assert payload["final_energy_gate_status"] == "fail"


def test_reporting_keeps_two_of_three_incomplete_without_contradictory_pass(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    reports = run_dir / "reports" / "native_energy_measurements"
    reports.mkdir(parents=True)
    aggregate = _aggregate(
        tmp_path / "measurement",
        setup_id="setup-a", run_id="run-a", valid=2,
    )
    payload = {
        "rows": [{
            "row": {
                "backend": "hailo10h_to_trt",
                "model": "yolo26s",
                "case": "b038",
                "setup_id": "setup-a",
                "task": "detection",
                "comparison_backend": "hailo10h_to_trt",
            },
            "ok": False,
            "run": {
                "rc": 1,
                "energy_aggregate": aggregate,
                "energy_aggregate_embedded": True,
                "energy_aggregate_bound": True,
                "energy_aggregate_complete": False,
                "energy_aggregate_verified": False,
                "energy_aggregate_import_status": "bound_incomplete",
            },
        }],
    }
    (reports / "native_producer_energy_results.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )

    [row] = collect_native_energy(run_dir)
    assert row["energy_repeat_requested_n"] == 3
    assert row["energy_repeat_valid_n"] == 2
    assert row["energy_repeat_status"] == "incomplete_valid_repeats"
    assert row["energy_ab_valid_n"] == 2
    assert row["energy_ab_status"] == "incomplete_valid_repeats"
    assert row["postprocess_status"] == "incomplete_valid_repeats"
    assert row["final_energy_gate_status"] == "fail"
    assert row["energy_aggregate_bound"] is True
    assert row["energy_aggregate_complete"] is False
    assert row["energy_aggregate_verified"] is False
    assert row["ok"] is False


def test_resume_reruns_exact_rows_as_fresh_series_and_atomically_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    out, summary, existing, old_plan, selectors = _resume_fixture(tmp_path)
    canonical = out / "native_producer_energy_results.json"
    prior_raw = canonical.read_bytes()
    measurement_calls = _install_fake_runner(
        namespace, out=out, old_plan=old_plan,
    )
    monkeypatch.setattr(sys, "argv", _resume_argv(out, summary, selectors))

    assert namespace["main"]() == 0
    assert measurement_calls == selectors
    merged = json.loads(canonical.read_text(encoding="utf-8"))
    assert merged["ok"] is True
    assert merged["resume_selected_row_count"] == 2
    assert merged["resume_reused_row_count"] == 1
    assert merged["resume_repeat_policy"] == (
        "fresh_complete_series_per_selected_row_no_repeat_splicing"
    )
    assert merged["rows"][0] == existing["rows"][0]
    assert json.dumps(
        merged["rows"][0], sort_keys=True, separators=(",", ":"),
    ).encode() == json.dumps(
        existing["rows"][0], sort_keys=True, separators=(",", ":"),
    ).encode()
    for item in merged["rows"][1:]:
        assert item["ok"] is True
        assert item["run"]["energy_aggregate_verified"] is True
        assert item["run"]["energy_aggregate"]["run_count"] == 3
        assert item["run"]["energy_aggregate"]["valid_postprocessed_runs"] == 3
        output = Path(item["row"]["measurement_output_dir"])
        assert "resume_attempts" in output.parts
        assert output.name.startswith("attempt_")
    history = Path(merged["resume_history_dir"])
    archived = history / "native_producer_energy_results.json"
    assert archived.read_bytes() == prior_raw
    assert merged["resume_previous_results_sha256"] == hashlib.sha256(
        prior_raw
    ).hexdigest()


def test_real_hailo8_resume_contract_replaces_one_and_reuses_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the script seam missed by the v2.72.4 Resume tests."""

    monkeypatch.setitem(
        _EXECUTION_CONTEXT,
        "remote_root",
        HAILO8_RESUME_REMOTE_ROOT,
    )
    namespace = _namespace()
    out, summary, existing, old_plan, selectors = (
        _single_hailo8_resume_fixture(tmp_path)
    )
    canonical = out / "native_producer_energy_results.json"
    prior_raw = canonical.read_bytes()
    reused_before = copy.deepcopy(existing["rows"][1:])

    def transform(
        row: dict[str, Any],
        index: int,
        plan_root: Path,
    ) -> dict[str, Any]:
        updated = dict(row)
        updated["comparison_backend"] = "hailo8"
        if index == 0:
            updated = _bind_archived_hailo8_contract(
                updated,
                plan_root=plan_root,
            )
        return updated

    measurement_calls = _install_fake_runner(
        namespace,
        out=out,
        old_plan=old_plan,
        row_transform=transform,
        stub_resume_preparation=False,
    )

    def exact_probe(requirements, **_kwargs):
        entries = [{
            "remote_path": requirement.remote_path,
            "roles": [requirement.role],
            "expected_sha256": requirement.sha256,
            "expected_size_bytes": requirement.size_bytes,
            "remote_status": "exact",
            "remote_sha256": requirement.sha256,
            "remote_size_bytes": requirement.size_bytes,
            "exact": True,
            "safe_to_rehydrate": False,
        } for requirement in requirements]
        return {
            "ok": True,
            "read_only": True,
            "requirement_count": len(entries),
            "exact_count": len(entries),
            "rehydration_required_count": 0,
            "all_exact": True,
            "entries": entries,
        }

    def verified_preflight(
        _command_template: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        return {
            "ok": True,
            "status": "verified",
            "expected_command_contract_sha256": kwargs[
                "expected_command_contract_sha256"
            ],
            "collector_started": False,
            "workload_started": False,
            "validation": {
                "ok": True,
                "status": "verified",
                "reasons": [],
            },
        }, str(kwargs["workload_command_template"])

    monkeypatch.setattr(
        preparation,
        "probe_remote_requirements",
        exact_probe,
    )
    monkeypatch.setattr(
        preparation,
        "build_resume_artifact_stage_map",
        lambda *_args, **_kwargs: pytest.fail(
            "all exact artifacts must not be restaged"
        ),
    )
    monkeypatch.setattr(
        preparation,
        "rehydrate_remote_stage_map",
        lambda *_args, **_kwargs: pytest.fail(
            "all exact artifacts must not mutate remote state"
        ),
    )
    monkeypatch.setattr(
        cohort,
        "_run_energy_preflight",
        verified_preflight,
    )
    namespace["main"].__globals__[
        "_prepare_resume_measurement_cohort"
    ] = preparation.prepare_resume_measurement_cohort
    monkeypatch.setattr(
        sys,
        "argv",
        _resume_argv(out, summary, selectors),
    )

    assert namespace["main"]() == 0
    assert measurement_calls == selectors
    merged = json.loads(canonical.read_text(encoding="utf-8"))
    assert merged["ok"] is True
    assert merged["status"] == "completed"
    assert merged["resume_existing"] is True
    assert merged["resume_selected_row_count"] == 1
    assert merged["resume_reused_row_count"] == 2
    assert merged["resume_selected_rows_ok"] is True
    assert merged["resume_merge_published"] is True
    assert merged["measurement_wrapper_started_count"] == 1
    assert merged["started_measurement_count"] == 1
    assert merged["rows"][0]["ok"] is True
    assert merged["rows"][0]["run"]["energy_aggregate_verified"] is True
    assert merged["rows"][1:] == reused_before
    assert json.dumps(
        merged["rows"][1:],
        sort_keys=True,
        separators=(",", ":"),
    ).encode() == json.dumps(
        reused_before,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert namespace["_resume_candidates"](merged) == []
    history = Path(merged["resume_history_dir"])
    assert (
        history / "native_producer_energy_results.json"
    ).read_bytes() == prior_raw
    assert merged["resume_previous_results_sha256"] == hashlib.sha256(
        prior_raw
    ).hexdigest()


def test_resume_failure_keeps_old_success_and_archives_prior_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    out, summary, existing, old_plan, selectors = _resume_fixture(tmp_path)
    canonical = out / "native_producer_energy_results.json"
    prior_raw = canonical.read_bytes()
    measurement_calls = _install_fake_runner(
        namespace, out=out, old_plan=old_plan, fail_selector=selectors[0],
    )
    monkeypatch.setattr(sys, "argv", _resume_argv(out, summary, selectors))

    assert namespace["main"]() == 2
    assert measurement_calls == selectors
    assert canonical.read_bytes() == prior_raw
    attempt_dirs = list((out / "resume_attempts").glob("resume_*"))
    assert len(attempt_dirs) == 1
    failed = json.loads(
        (
            attempt_dirs[0] / "resume_failed_results_no_merge.json"
        ).read_text(encoding="utf-8")
    )
    assert failed["ok"] is False
    assert failed["status"] == "resume_selected_measurement_failed_no_merge"
    assert failed["resume_merge_published"] is False
    assert failed["rows"][0] == existing["rows"][0]
    assert failed["rows"][1]["run"]["energy_aggregate_bound"] is True
    assert failed["rows"][1]["run"]["energy_aggregate_complete"] is False
    assert failed["rows"][1]["run"]["energy_aggregate_verified"] is False
    assert (
        Path(failed["resume_history_dir"])
        / "native_producer_energy_results.json"
    ).read_bytes() == prior_raw


def test_resume_preparation_failure_starts_no_measurement_and_keeps_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    out, summary, _existing, old_plan, selectors = _resume_fixture(tmp_path)
    canonical = out / "native_producer_energy_results.json"
    prior_raw = canonical.read_bytes()
    measurement_calls = _install_fake_runner(
        namespace, out=out, old_plan=old_plan,
    )
    namespace["main"].__globals__[
        "_prepare_resume_measurement_cohort"
    ] = lambda *args, **kwargs: {
        "ok": False,
        "status": "resume_cohort_preflight_failed",
        "measurement_wrapper_allowed": False,
        "report_path": str(out / "blocked_preparation.json"),
        "cohort_preflight": {
            "ok": False,
            "measurement_wrapper_allowed": False,
        },
    }
    monkeypatch.setattr(sys, "argv", _resume_argv(out, summary, selectors))

    assert namespace["main"]() == 2
    assert measurement_calls == []
    assert canonical.read_bytes() == prior_raw
    [attempt] = list((out / "resume_attempts").glob("resume_*"))
    failure = json.loads(
        (attempt / "resume_failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == (
        "resume_preparation_or_cohort_preflight_failed"
    )
    assert failure["resume_merge_published"] is False
    assert failure["measurement_wrapper_started_count"] == 0
    assert failure["started_measurement_count"] == 0
    assert failure["collector_started_repeat_count"] == 0
    assert failure["workload_started_repeat_count"] == 0


def test_resume_contract_drift_fails_before_measurement(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    out, _summary, existing, old_plan, selectors = _resume_fixture(tmp_path)
    fresh = copy.deepcopy(old_plan)
    fresh["rows"][1]["successful_command_contract_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="resume contract drift"):
        namespace["_validate_resume_selection"](
            existing, fresh,
            [namespace["_parse_resume_selector"](value) for value in selectors],
        )


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--remote-root", "/remote/not-the-canonical-run"),
        ("--remote-tool-dir", "/remote/other-tool"),
        ("--hailo10-ssh", "nx@other-hailo10"),
        ("--hailo10-env", "source /env/other/bin/activate"),
    ],
)
def test_resume_rejects_explicit_execution_context_drift_before_any_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    namespace = _namespace()
    out, summary, _existing, _old_plan, selectors = _resume_fixture(tmp_path)
    canonical = out / "native_producer_energy_results.json"
    before = canonical.read_bytes()
    namespace["main"].__globals__["_run"] = lambda *args, **kwargs: pytest.fail(
        "execution-context drift must fail before plan or remote commands"
    )
    monkeypatch.setattr(
        sys, "argv", _resume_argv(out, summary, selectors) + [flag, value],
    )

    with pytest.raises(SystemExit) as exc:
        namespace["main"]()
    assert exc.value.code == 2
    assert canonical.read_bytes() == before


def test_resume_derives_omitted_execution_context_from_canonical_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    out, summary, _existing, old_plan, selectors = _resume_fixture(tmp_path)
    canonical = out / "native_producer_energy_results.json"
    legacy_report = json.loads(canonical.read_text(encoding="utf-8"))
    legacy_report["plan_payload"].pop("execution_context")
    canonical.write_text(json.dumps(legacy_report), encoding="utf-8")
    observed_plan_commands: list[list[str]] = []

    def fake_run(
        cmd: list[str], timeout: int | None = None, label: str = "",
    ) -> dict[str, Any]:
        del timeout
        if label != "energy_plan":
            pytest.fail("only the local plan command should be reached")
        observed_plan_commands.append(list(cmd))
        plan_dir = Path(cmd[cmd.index("--out-dir") + 1])
        fresh_rows = copy.deepcopy(old_plan["rows"])
        for fresh_row in fresh_rows:
            fresh_row["measurement_plan_attempt_id"] = "fresh"
        fresh_rows[1]["command_file"] = str(plan_dir / "missing-command.sh")
        payload = _plan(fresh_rows, attempt="fresh")
        payload.pop("execution_context")
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "native_producer_energy_plan.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )
        return {"rc": 0, "elapsed_s": 0.01}

    namespace["main"].__globals__["_run"] = fake_run
    monkeypatch.setattr(sys, "argv", _resume_argv(out, summary, selectors))

    # Selection reaches the local plan, then fails only because this focused
    # fake deliberately does not materialize fresh relocated command files.
    assert namespace["main"]() == 2
    assert len(observed_plan_commands) == 1
    command = observed_plan_commands[0]
    for field, flag in (
        ("remote_root", "--remote-root"),
        ("remote_tool_dir", "--remote-tool-dir"),
        ("hailo8_ssh", "--hailo8-ssh"),
        ("hailo10_ssh", "--hailo10-ssh"),
        ("deepx_ssh", "--deepx-ssh"),
        ("hailo8_env", "--hailo8-env"),
        ("hailo10_env", "--hailo10-env"),
        ("deepx_env", "--deepx-env"),
    ):
        assert command[command.index(flag) + 1] == _EXECUTION_CONTEXT[field]


def test_resume_rejects_normalized_remote_command_drift(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    _out, _summary, existing, old_plan, selectors = _resume_fixture(tmp_path)
    fresh = copy.deepcopy(old_plan)
    source = Path(fresh["rows"][1]["command_file"])
    tampered = source.with_name(source.stem + ".tampered.sh")
    tampered.write_text(
        source.read_text(encoding="utf-8").replace(
            "cd /remote/tool", "cd /remote/other-tool",
        ),
        encoding="utf-8",
    )
    fresh["rows"][1]["command_file"] = str(tampered)

    with pytest.raises(ValueError, match="resume remote command drift"):
        namespace["_validate_resume_selection"](
            existing, fresh,
            [namespace["_parse_resume_selector"](value) for value in selectors],
        )

    unbound = copy.deepcopy(old_plan)
    unbound["rows"][1]["measure_command"] = unbound["rows"][1][
        "measure_command"
    ].replace(str(source), str(tampered))
    with pytest.raises(ValueError, match="resume remote command drift"):
        namespace["_validate_resume_selection"](
            existing, unbound,
            [namespace["_parse_resume_selector"](value) for value in selectors],
        )


def test_resume_rejects_zero_ambiguous_and_already_complete_selections(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    _out, _summary, existing, old_plan, selectors = _resume_fixture(tmp_path)
    parsed = namespace["_parse_resume_selector"](selectors[0])

    missing = ("missing", "model", "case", "setup")
    with pytest.raises(ValueError, match="did not match"):
        namespace["_validate_resume_selection"](existing, old_plan, [missing])

    duplicate_plan = copy.deepcopy(old_plan)
    duplicate_plan["rows"].append(copy.deepcopy(duplicate_plan["rows"][0]))
    with pytest.raises(ValueError, match="duplicate resume identity"):
        namespace["_validate_resume_selection"](
            existing, duplicate_plan, [parsed],
        )

    complete_selector = namespace["_resume_row_identity"](
        old_plan["rows"][0]
    )
    with pytest.raises(ValueError, match="complete verified result"):
        namespace["_validate_resume_selection"](
            existing, old_plan, [complete_selector],
        )

    with pytest.raises(ValueError, match="every incomplete existing row"):
        namespace["_validate_resume_selection"](
            existing, old_plan, [parsed],
        )


def test_list_resume_candidates_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    namespace = _namespace()
    out, summary, _existing, _old_plan, selectors = _resume_fixture(tmp_path)
    before = {
        path.relative_to(out): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    namespace["main"].__globals__["_run"] = lambda *args, **kwargs: pytest.fail(
        "read-only listing must not invoke plan or measurement commands"
    )
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--summary", str(summary),
        "--out-dir", str(out),
        "--list-resume-candidates",
    ])

    assert namespace["main"]() == 0
    listed = json.loads(capsys.readouterr().out)
    assert [item["selector"] for item in listed["candidates"]] == selectors
    after = {
        path.relative_to(out): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "raw_result",
    [
        "{}",
        '{"rows": []}',
        "{",
    ],
)
def test_list_resume_candidates_rejects_empty_or_malformed_report_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_result: str,
) -> None:
    namespace = _namespace()
    out = tmp_path / "native_energy_measurements"
    out.mkdir()
    canonical = out / "native_producer_energy_results.json"
    canonical.write_text(raw_result, encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text('{"rows": []}', encoding="utf-8")
    before = {
        path.relative_to(out): (
            path.read_bytes(), path.stat().st_mtime_ns
        )
        for path in out.rglob("*")
        if path.is_file()
    }
    namespace["main"].__globals__["_run"] = (
        lambda *args, **kwargs: pytest.fail(
            "invalid read-only listing must not invoke commands"
        )
    )
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--summary", str(summary),
        "--out-dir", str(out),
        "--list-resume-candidates",
    ])

    with pytest.raises(SystemExit) as exc:
        namespace["main"]()
    assert exc.value.code == 2
    after = {
        path.relative_to(out): (
            path.read_bytes(), path.stat().st_mtime_ns
        )
        for path in out.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_expected_selected_row_count_mismatch_fails_before_any_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    out, summary, _existing, _old_plan, selectors = _resume_fixture(tmp_path)
    namespace["main"].__globals__["_run"] = lambda *args, **kwargs: pytest.fail(
        "selector count mismatch must fail before plan or measurement commands"
    )
    argv = _resume_argv(out, summary, selectors)
    argv[argv.index("--expected-selected-rows") + 1] = "1"
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        namespace["main"]()
    assert exc.value.code == 2
