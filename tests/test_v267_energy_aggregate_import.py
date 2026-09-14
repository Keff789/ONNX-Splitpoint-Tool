from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import shlex
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_native_producer_energy_from_summary.py"


def _namespace() -> dict[str, Any]:
    return runpy.run_path(str(SCRIPT))


def _command(
    out: Path,
    *,
    requested_runs: int = 3,
    setup_id: str = "setup-a",
    run_id: str = "native-row-a",
    equals_out: bool = False,
) -> str:
    out_option = (
        f"--out={shlex.quote(str(out))}"
        if equals_out else f"--out {shlex.quote(str(out))}"
    )
    return (
        "python energy_measurement_cli.py measure "
        f"--setup-id {shlex.quote(setup_id)} "
        f"--run-id {shlex.quote(run_id)} "
        f"--runs {requested_runs} {out_option} "
        "--command 'runner --out nested.json --runs 99'"
    )


def _aggregate(
    out: Path,
    *,
    requested_runs: int = 3,
    effective_runs: int | None = None,
    setup_id: str = "setup-a",
    run_id: str = "native-row-a",
) -> dict[str, Any]:
    effective = requested_runs if effective_runs is None else effective_runs
    return {
        "ok": True,
        "status": "ok",
        "setup_id": setup_id,
        "run_id": run_id,
        "out_dir": str(out.resolve()),
        "run_count": effective,
        "requested_valid_repeat_count": effective,
        "materialized_logical_repeat_count": effective,
        "valid_postprocessed_runs": effective,
        "scientific_primary_valid_run_count": effective,
        "scientific_primary_method": "command_marker_window",
        "scientific_primary_energy_status": "available",
        "repeat_contract_complete": True,
        "energy_window_method_ab": {
            "requested_run_count": requested_runs,
            "effective_run_count": effective,
            "scientific_primary_method": "command_marker_window",
        },
        "runs": [{"run_index": index} for index in range(effective)],
        "scientific_primary_energy_statistics": {
            "energy_j": {
                "n": effective,
                "mean": 10.0,
                "ci_low": 9.8,
                "ci_high": 10.2,
            }
        },
    }


def _write_aggregate(out: Path, aggregate: dict[str, Any]) -> tuple[Path, int]:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "energy_aggregate.json"
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    return path, path.stat().st_mtime_ns


def _attach_valid(
    namespace: dict[str, Any],
    out: Path,
    *,
    command: str | None = None,
    requested_runs: int = 3,
    setup_id: str = "setup-a",
    run_id: str = "native-row-a",
    started_ns: int | None = None,
) -> dict[str, Any]:
    command = command or _command(
        out, requested_runs=requested_runs, setup_id=setup_id, run_id=run_id,
    )
    path = out / "energy_aggregate.json"
    return namespace["_attach_energy_aggregate"](
        {"rc": 0, "stdout": "bounded tail"},
        command,
        out,
        expected_runs=requested_runs,
        expected_run_id=run_id,
        expected_setup_id=setup_id,
        execution_started_ns=(path.stat().st_mtime_ns if started_ns is None else started_ns),
        aggregate_absent_before_execution=True,
    )


def test_energy_aggregate_binds_outer_options_and_embeds_exact_repeats(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / "measurement with spaces"
    aggregate = _aggregate(measurement)
    aggregate_path, _mtime = _write_aggregate(measurement, aggregate)

    result = _attach_valid(namespace, measurement)

    assert result["energy_aggregate_embedded"] is True
    assert result["energy_aggregate_bound"] is True
    assert result["energy_aggregate_complete"] is True
    assert result["energy_aggregate_verified"] is True
    assert result["energy_aggregate_import_status"] == "verified"
    assert result["energy_aggregate_planned_request_repeat_count"] == 3
    assert result["energy_aggregate_effective_repeat_count"] == 3
    assert result["energy_aggregate_requested_repeat_count"] == 3
    assert result["energy_aggregate_valid_repeat_count"] == 3
    assert result["energy_aggregate_materialized_repeat_count"] == 3
    assert result["energy_aggregate_sha256"] == hashlib.sha256(
        aggregate_path.read_bytes()
    ).hexdigest()


def test_energy_aggregate_accepts_equals_out_and_legitimate_ab_expansion(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / "measurement"
    _write_aggregate(
        measurement,
        _aggregate(measurement, requested_runs=1, effective_runs=3),
    )
    command = _command(measurement, requested_runs=1, equals_out=True)

    assert namespace["_command_option"](command, "--out") == str(measurement)
    result = _attach_valid(
        namespace, measurement, command=command, requested_runs=1,
    )
    assert result["energy_aggregate_verified"] is True
    assert result["energy_aggregate_planned_request_repeat_count"] == 1
    assert result["energy_aggregate_effective_repeat_count"] == 3


def test_frozen_effective_repeat_count_rejects_additional_local_expansion(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / "measurement"
    path, _mtime = _write_aggregate(
        measurement,
        _aggregate(measurement, requested_runs=3, effective_runs=4),
    )

    result = namespace["_attach_energy_aggregate"](
        {"rc": 0},
        _command(measurement, requested_runs=3),
        measurement,
        expected_runs=3,
        expected_effective_runs=3,
        expected_run_id="native-row-a",
        expected_setup_id="setup-a",
        execution_started_ns=path.stat().st_mtime_ns,
        aggregate_absent_before_execution=True,
    )

    assert result["energy_aggregate_verified"] is False
    assert any(
        "does_not_match_frozen_3" in error
        for error in result["energy_aggregate_validation_errors"]
    )


def test_v2721_prepare_binds_requested_and_effective_repeat_contract(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    root = tmp_path / "root"
    base = root / "measurements" / "row" / "plan"
    row, plan = _planned_row(base, runs=3)
    row.update({
        "measurement_profile_requested_repeats": 1,
        "measurement_effective_repeats": 3,
        "measurement_repeat_expansion_applied": True,
        "measurement_repeat_expansion_reason": (
            "frozen_window_method_ab_minimum"
        ),
    })
    plan.update({
        "energy_profile_requested_runs_per_row": 1,
        "energy_effective_runs_per_row": 3,
        "energy_repeat_expansion_applied": True,
        "energy_repeat_expansion_reason": (
            "frozen_window_method_ab_minimum"
        ),
    })

    prepared = namespace["_prepare_measurement_execution"](
        row, plan, allowed_root=root, attempt_id="frozen-repeats",
    )
    assert prepared["expected_runs"] == 3
    assert prepared["expected_effective_runs"] == 3

    drifted = dict(row)
    drifted["measurement_effective_repeats"] = 4
    with pytest.raises(
        ValueError,
        match="requested/effective repeat contract drift",
    ):
        namespace["_prepare_measurement_execution"](
            drifted,
            plan,
            allowed_root=root,
            attempt_id="drifted-repeats",
        )


@pytest.mark.parametrize("second_value", ["true", "false"])
def test_energy_aggregate_rejects_identical_and_conflicting_duplicate_keys(
    tmp_path: Path, second_value: str,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / "measurement"
    aggregate = json.dumps(_aggregate(measurement))
    aggregate = aggregate.replace(
        '{"ok": true,', f'{{"ok": true, "ok": {second_value},', 1,
    )
    measurement.mkdir(parents=True)
    (measurement / "energy_aggregate.json").write_text(
        aggregate, encoding="utf-8",
    )

    result = _attach_valid(namespace, measurement)

    assert result["energy_aggregate_verified"] is False
    assert result["energy_aggregate_embedded"] is False
    assert "aggregate_invalid_json" in result[
        "energy_aggregate_validation_errors"
    ]


@pytest.mark.parametrize(
    ("command", "name"),
    [
        ("tool --out old --out new", "--out"),
        ("tool --out=old --out new", "--out"),
        ("tool --out --runs 3", "--out"),
        ("tool --runs 3 --runs=4", "--runs"),
        ("tool --setup-id a --setup-id b", "--setup-id"),
        ("tool --run-id a --run-id=b", "--run-id"),
        ("tool -- --out late", "--out"),
    ],
)
def test_top_level_option_parser_rejects_ambiguous_or_inactive_options(
    command: str, name: str,
) -> None:
    with pytest.raises(ValueError):
        _namespace()["_command_option"](command, name)


@pytest.mark.parametrize("value", ["0", "-1", "1.0", "03", "three"])
def test_repeat_parser_rejects_noncanonical_positive_integers(value: str) -> None:
    with pytest.raises(ValueError):
        _namespace()["_positive_cli_int"](value, "--runs")


def _planned_row(base: Path, *, runs: int = 3) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt = "plan-attempt-a"
    setup = "setup-a"
    run_id = "native-row-a"
    row = {
        "backend": "hailo8_to_trt",
        "model": "model-a",
        "case": "case-a",
        "setup_id": setup,
        "comparison_backend": "hailo8",
        "precision": "fp16",
        "measurement_output_dir": str(base),
        "measurement_output_base_dir": str(base),
        "measurement_plan_attempt_id": attempt,
        "measurement_setup_id": setup,
        "measurement_run_id": run_id,
        "measurement_requested_repeats": runs,
        "measurement_profile_requested_repeats": runs,
        "measurement_effective_repeats": runs,
        "measurement_repeat_expansion_applied": False,
        "measurement_repeat_expansion_reason": "none",
        "measure_command": _command(
            base, requested_runs=runs, setup_id=setup, run_id=run_id,
        ),
    }
    plan = {
        "measurement_plan_attempt_id": attempt,
        "energy_runs_per_row": runs,
        "energy_profile_requested_runs_per_row": runs,
        "energy_effective_runs_per_row": runs,
        "energy_repeat_expansion_applied": False,
        "energy_repeat_expansion_reason": "none",
        "preflight_status": "passed",
        "preflight": {
            "status": "passed",
            "ok": True,
            "measurement_start_allowed": True,
            "energy_plan_coverage_contract_valid": True,
            "technical_measurement_contract_valid": True,
        },
        "rows": [row],
    }
    return row, plan


def test_prepare_execution_materializes_fresh_attempt_and_rewrites_outer_out(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    root = tmp_path / "native_energy_measurements"
    base = root / "measurements" / "row-a" / "plan-a"
    row, plan = _planned_row(base)

    prepared = namespace["_prepare_measurement_execution"](
        row, plan, allowed_root=root, attempt_id="execution-a",
    )
    actual = Path(prepared["output_dir"])
    assert actual == base / "attempt_execution-a"
    assert actual.is_dir() and list(actual.iterdir()) == []
    assert namespace["_command_option"](prepared["command"], "--out") == str(actual)
    assert prepared["row"]["measurement_planned_output_dir"] == str(base.resolve())
    assert prepared["row"]["measurement_output_dir"] == str(actual)
    assert prepared["row"]["measurement_execution_attempt_id"] == "execution-a"

    with pytest.raises(FileExistsError):
        namespace["_prepare_measurement_execution"](
            row, plan, allowed_root=root, attempt_id="execution-a",
        )


@pytest.mark.parametrize(
    "mutation",
    ["row_out", "command_out", "row_runs", "plan_runs", "setup", "run_id", "plan_attempt"],
)
def test_prepare_execution_rejects_plan_row_command_mismatches(
    tmp_path: Path, mutation: str,
) -> None:
    namespace = _namespace()
    root = tmp_path / "root"
    base = root / "measurements" / "row" / "plan"
    row, plan = _planned_row(base)
    if mutation == "row_out":
        row["measurement_output_dir"] = str(root / "other")
    elif mutation == "command_out":
        row["measure_command"] = row["measure_command"].replace(str(base), str(root / "other"))
    elif mutation == "row_runs":
        row["measurement_requested_repeats"] = 2
    elif mutation == "plan_runs":
        plan["energy_runs_per_row"] = 2
    elif mutation == "setup":
        row["measurement_setup_id"] = "other-setup"
    elif mutation == "run_id":
        row["measurement_run_id"] = "other-run"
    elif mutation == "plan_attempt":
        plan["measurement_plan_attempt_id"] = "other-attempt"

    with pytest.raises(ValueError):
        namespace["_prepare_measurement_execution"](
            row, plan, allowed_root=root, attempt_id=f"bad-{mutation}",
        )


def test_declared_output_mismatch_rejects_old_aggregate_without_embedding(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_aggregate(old, _aggregate(old))
    new.mkdir()

    result = namespace["_attach_energy_aggregate"](
        {"rc": 0},
        _command(new),
        old,
        expected_runs=3,
        expected_run_id="native-row-a",
        expected_setup_id="setup-a",
        execution_started_ns=1,
        aggregate_absent_before_execution=True,
    )
    assert result["energy_aggregate_verified"] is False
    assert result["energy_aggregate_embedded"] is False
    assert "energy_aggregate" not in result
    assert "declared_output_does_not_match_command_out" in result[
        "energy_aggregate_validation_errors"
    ]


def test_stale_aggregate_is_rejected_and_never_embedded(tmp_path: Path) -> None:
    namespace = _namespace()
    measurement = tmp_path / "measurement"
    path, mtime_ns = _write_aggregate(measurement, _aggregate(measurement))

    result = _attach_valid(
        namespace,
        measurement,
        started_ns=(mtime_ns + namespace["_AGGREGATE_MTIME_TOLERANCE_NS"] + 1),
    )
    assert path.is_file()
    assert result["energy_aggregate_verified"] is False
    assert result["energy_aggregate_embedded"] is False
    assert "energy_aggregate" not in result
    assert "aggregate_predates_measurement_execution" in result[
        "energy_aggregate_validation_errors"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ok", None),
        ("ok", False),
        ("repeat_contract_complete", None),
        ("repeat_contract_complete", False),
        ("run_count", 1),
        ("run_count", 4),
        ("requested_valid_repeat_count", True),
        ("materialized_logical_repeat_count", "3"),
        ("valid_postprocessed_runs", 1),
        ("scientific_primary_valid_run_count", 4),
    ],
)
def test_aggregate_contract_fields_are_exact_and_fail_closed(
    tmp_path: Path, field: str, value: Any,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / f"measurement-{field}-{value}"
    aggregate = _aggregate(measurement)
    if value is None:
        aggregate.pop(field)
    else:
        aggregate[field] = value
    _write_aggregate(measurement, aggregate)

    result = _attach_valid(namespace, measurement)
    assert result["energy_aggregate_verified"] is False
    diagnostic_incomplete = (
        (field == "ok" and value is False)
        or (field == "repeat_contract_complete" and value is False)
        or (field == "valid_postprocessed_runs" and value == 1)
    )
    assert result["energy_aggregate_embedded"] is diagnostic_incomplete
    assert result["energy_aggregate_bound"] is diagnostic_incomplete
    assert result["energy_aggregate_complete"] is False
    if diagnostic_incomplete:
        assert result["energy_aggregate_import_status"] == "bound_incomplete"
        assert result["energy_aggregate_completion_errors"]
    else:
        assert "energy_aggregate" not in result
        assert result["energy_aggregate_import_status"] == "rejected_fail_closed"


@pytest.mark.parametrize("identity", ["setup_id", "run_id", "out_dir"])
def test_aggregate_identity_mismatch_is_rejected(
    tmp_path: Path, identity: str,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / f"measurement-{identity}"
    aggregate = _aggregate(measurement)
    aggregate[identity] = str(tmp_path / "wrong") if identity == "out_dir" else "wrong"
    _write_aggregate(measurement, aggregate)

    result = _attach_valid(namespace, measurement)
    assert result["energy_aggregate_verified"] is False
    assert result["energy_aggregate_embedded"] is False
    assert result["energy_aggregate_bound"] is False


def test_command_runs_three_rejects_self_consistent_stale_one_repeat(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / "measurement"
    _write_aggregate(
        measurement,
        _aggregate(measurement, requested_runs=1, effective_runs=1),
    )

    result = _attach_valid(namespace, measurement, requested_runs=3)
    assert result["energy_aggregate_verified"] is False
    assert result["energy_aggregate_embedded"] is False
    assert any(
        "does_not_match_planned_3" in error
        for error in result["energy_aggregate_validation_errors"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scientific_primary_method", "chapter4_baseline"),
        ("scientific_primary_energy_status", "unavailable"),
    ],
)
def test_pre_v267_or_unavailable_primary_role_is_rejected(
    tmp_path: Path, field: str, value: str,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / field
    aggregate = _aggregate(measurement)
    aggregate[field] = value
    _write_aggregate(measurement, aggregate)
    result = _attach_valid(namespace, measurement)
    assert result["energy_aggregate_verified"] is False
    if field == "scientific_primary_energy_status":
        assert result["energy_aggregate_embedded"] is True
        assert result["energy_aggregate_bound"] is True
        assert result["energy_aggregate_complete"] is False
        assert result["energy_aggregate_import_status"] == "bound_incomplete"
    else:
        assert result["energy_aggregate_embedded"] is False


def test_pre_v267_ab_primary_role_is_rejected(tmp_path: Path) -> None:
    namespace = _namespace()
    measurement = tmp_path / "ab-role"
    aggregate = _aggregate(measurement)
    aggregate["energy_window_method_ab"]["scientific_primary_method"] = (
        "chapter4_baseline"
    )
    _write_aggregate(measurement, aggregate)
    result = _attach_valid(namespace, measurement)
    assert result["energy_aggregate_verified"] is False
    assert result["energy_aggregate_embedded"] is False


def test_nonzero_measurement_rc_binds_fresh_aggregate_as_incomplete_diagnostic(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / "measurement"
    path, _mtime = _write_aggregate(measurement, _aggregate(measurement))
    result = namespace["_attach_energy_aggregate"](
        {"rc": 1},
        _command(measurement),
        measurement,
        expected_runs=3,
        expected_run_id="native-row-a",
        expected_setup_id="setup-a",
        execution_started_ns=path.stat().st_mtime_ns,
        aggregate_absent_before_execution=True,
    )
    assert result["energy_aggregate_verified"] is False
    assert result["energy_aggregate_embedded"] is True
    assert result["energy_aggregate_bound"] is True
    assert result["energy_aggregate_complete"] is False
    assert result["energy_aggregate_import_status"] == "bound_incomplete"
    assert "measurement_process_rc_nonzero" in result[
        "energy_aggregate_completion_errors"
    ]


def test_fresh_two_of_three_aggregate_is_retained_without_becoming_verified(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    measurement = tmp_path / "measurement"
    aggregate = _aggregate(measurement, requested_runs=1, effective_runs=3)
    aggregate.update({
        "ok": False,
        "status": "incomplete_valid_repeats",
        "repeat_contract_complete": False,
        "valid_postprocessed_runs": 2,
        "scientific_primary_valid_run_count": 2,
    })
    aggregate["scientific_primary_energy_statistics"]["energy_j"]["n"] = 2
    path, _mtime = _write_aggregate(measurement, aggregate)

    result = namespace["_attach_energy_aggregate"](
        {"rc": 1},
        _command(measurement, requested_runs=1),
        measurement,
        expected_runs=1,
        expected_run_id="native-row-a",
        expected_setup_id="setup-a",
        execution_started_ns=path.stat().st_mtime_ns,
        aggregate_absent_before_execution=True,
    )

    assert result["energy_aggregate_embedded"] is True
    assert result["energy_aggregate_bound"] is True
    assert result["energy_aggregate_complete"] is False
    assert result["energy_aggregate_verified"] is False
    assert result["energy_aggregate"]["valid_postprocessed_runs"] == 2
    assert result["energy_aggregate_import_status"] == "bound_incomplete"
    assert result["energy_aggregate_validation_errors"] == []


def test_successful_main_removes_partial_and_records_actual_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    out = tmp_path / "reports" / "native_energy_measurements"
    base = out / "measurements" / "row" / "plan-a"
    row, plan = _planned_row(base)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    out.mkdir(parents=True)
    (out / "native_producer_energy_results.json").write_text("stale", encoding="utf-8")
    (out / "native_producer_energy_results.partial.json").write_text("stale", encoding="utf-8")

    def fake_run(cmd: list[str], timeout: int | None = None, label: str = "") -> dict[str, Any]:
        del timeout
        if label == "energy_plan":
            plan_dir = out / "plan"
            plan_dir.mkdir(parents=True, exist_ok=True)
            (plan_dir / "native_producer_energy_plan.json").write_text(
                json.dumps(plan), encoding="utf-8",
            )
            return {"rc": 0, "elapsed_s": 0.01}
        command = shlex.join(cmd)
        actual_out = Path(namespace["_command_option"](command, "--out"))
        requested = int(namespace["_command_option"](command, "--runs"))
        setup = namespace["_command_option"](command, "--setup-id")
        run_id = namespace["_command_option"](command, "--run-id")
        _write_aggregate(
            actual_out,
            _aggregate(
                actual_out, requested_runs=requested,
                setup_id=setup, run_id=run_id,
            ),
        )
        return {"rc": 0, "elapsed_s": 0.01}

    namespace["main"].__globals__["_run"] = fake_run
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--summary", str(summary), "--out-dir", str(out),
        "--runs", "3",
    ])
    assert namespace["main"]() == 0
    final = json.loads(
        (out / "native_producer_energy_results.json").read_text(encoding="utf-8")
    )
    assert final["ok"] is True and final["status"] == "completed"
    assert not (out / "native_producer_energy_results.partial.json").exists()
    actual = Path(final["rows"][0]["row"]["measurement_output_dir"])
    assert actual.parent == base
    assert actual.name.startswith("attempt_")
    assert final["rows"][0]["run"]["energy_aggregate_verified"] is True


def test_plan_failure_clears_stale_partial_before_terminal_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace()
    out = tmp_path / "native_energy_measurements"
    out.mkdir()
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    (out / "native_producer_energy_results.json").write_text("stale", encoding="utf-8")
    partial = out / "native_producer_energy_results.partial.json"
    partial.write_text("stale", encoding="utf-8")
    namespace["main"].__globals__["_run"] = lambda *args, **kwargs: {"rc": 1}
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--summary", str(summary), "--out-dir", str(out),
    ])

    assert namespace["main"]() == 2
    final = json.loads(
        (out / "native_producer_energy_results.json").read_text(encoding="utf-8")
    )
    assert final["status"] == "native_energy_plan_failed"
    assert not partial.exists()
