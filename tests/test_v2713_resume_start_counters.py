from __future__ import annotations

import json
from pathlib import Path
import runpy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_native_producer_energy_from_summary.py"


def _observation(
    output_dir: Path,
    *,
    aggregate_verified: bool = False,
) -> dict[str, Any]:
    namespace = runpy.run_path(str(SCRIPT))
    return namespace["_measurement_start_observation"](
        output_dir,
        aggregate_verified=aggregate_verified,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_normal_aggregate_infers_workload_start_from_timing_evidence(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "energy_summary.json", {
        "ok": True,
        "runs": [{
            "run_index": 0,
            "collector_started": True,
            # The frozen collector does not currently materialize a top-level
            # workload_started=True field on successful repeats.
            "workload_timing": {
                "start_ns": 100,
                "end_ns": 200,
                "rc": 0,
            },
            "workload_command_rc": 0,
        }],
    })

    observed = _observation(tmp_path, aggregate_verified=True)

    assert observed["measurement_started"] is True
    assert observed["collector_started_repeat_count"] == 1
    assert observed["workload_started_repeat_count"] == 1
    assert observed["aggregate_run_count"] == 1


def test_abrupt_wrapper_failure_uses_repeat_local_start_evidence(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run_000"
    _write_json(run / "energy_summary.json", {
        "run_index": 0,
        "status": "collector_running",
        "collector_started": True,
    })
    (run / "workload_timing.txt").write_text(
        "start_ns=123456789\n",
        encoding="utf-8",
    )

    observed = _observation(tmp_path)

    assert observed["energy_summary_present"] is False
    assert observed["repeat_summary_file_count"] == 1
    assert observed["workload_timing_start_count"] == 1
    assert observed["measurement_started"] is True
    assert observed["collector_started_repeat_count"] == 1
    assert observed["workload_started_repeat_count"] == 1


def test_aggregate_and_repeat_summary_are_not_double_counted(
    tmp_path: Path,
) -> None:
    repeat = {
        "run_index": 0,
        "collector_started": True,
        "workload_timing": {"start_ns": 1, "end_ns": 2, "rc": 0},
        "workload_command_rc": 0,
    }
    _write_json(tmp_path / "energy_summary.json", {
        "ok": True,
        "runs": [repeat],
    })
    _write_json(tmp_path / "run_000" / "energy_summary.json", repeat)
    (tmp_path / "run_000" / "workload_timing.txt").write_text(
        "start_ns=1\nend_ns=2\nrc=0\n",
        encoding="utf-8",
    )

    observed = _observation(tmp_path, aggregate_verified=True)

    assert observed["collector_started_repeat_count"] == 1
    assert observed["workload_started_repeat_count"] == 1
    assert observed["repeat_summary_file_count"] == 1


def test_failed_preflight_remains_zero_physical_starts(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "energy_summary.json", {
        "ok": False,
        "status": "preflight_failed",
        "collector_started": False,
        "workload_started": False,
    })
    _write_json(tmp_path / "run_000" / "energy_summary.json", {
        "run_index": 0,
        "status": "preflight_failed",
        "collector_started": False,
        "workload_started": False,
        "workload_command_rc": None,
    })

    observed = _observation(tmp_path)

    assert observed["measurement_started"] is False
    assert observed["collector_started_repeat_count"] == 0
    assert observed["workload_started_repeat_count"] == 0

