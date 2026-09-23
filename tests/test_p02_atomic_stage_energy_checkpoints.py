from __future__ import annotations

"""Short, hardware-free P0.2 checkpoint and Resume fixtures."""

import importlib.util
import copy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.workflow import checkpoints
from onnx_splitpoint_tool.workflow.checkpoints import (
    AtomicRowJournal,
    load_stage_checkpoint,
    load_reusable_stage_checkpoint,
    write_stage_checkpoint,
)
from onnx_splitpoint_tool.workflow.contracts import StageResult


def _load_script(name: str, module_name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


COORDINATOR = _load_script(
    "run_evalrun_native_producer_variants.py",
    "p02_native_variant_coordinator",
)
ENERGY = _load_script(
    "run_native_producer_energy_from_summary.py",
    "p02_native_energy_runner",
)


def _load_runner_module():
    try:
        from onnx_splitpoint_tool.workflow import runner
        return runner
    except ModuleNotFoundError as exc:
        if exc.name != "jsonschema":
            raise
        fixture_jsonschema = types.ModuleType("jsonschema")

        class _FixtureValidator:
            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def check_schema(*_args, **_kwargs):
                return None

            def iter_errors(self, *_args, **_kwargs):
                return []

        fixture_jsonschema.Draft202012Validator = _FixtureValidator
        fixture_jsonschema.validate = lambda *_args, **_kwargs: None
        sys.modules["jsonschema"] = fixture_jsonschema
        from onnx_splitpoint_tool.workflow import runner
        return runner


RUNNER = _load_runner_module()


def _energy_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "backend": "hailo8_to_trt",
            "model": "fixture_model",
            "case": f"b{index:03d}",
            "setup_id": "fixture_hailo8_setup",
            "comparison_backend": "hailo8",
            "precision": "uint8_dequant_fp16",
            "measure_command": "fixture-energy-command",
        }
        for index in range(count)
    ]


def _journal(root: Path, count: int) -> tuple[
    AtomicRowJournal, list[dict[str, object]], list[str], list[str]
]:
    rows = _energy_rows(count)
    identities = [ENERGY._managed_row_identity(row) for row in rows]
    row_hashes = [
        checkpoints.canonical_json_sha256({"row": row}) for row in rows
    ]
    return (
        AtomicRowJournal.create(
            root,
            plan_hash="frozen-plan",
            rows=rows,
            identities=identities,
            row_contract_hashes=row_hashes,
        ),
        rows,
        identities,
        row_hashes,
    )


def _planned_performance_rows_63() -> list[dict[str, object]]:
    setup_contracts = (
        ("deepx", "deepx_to_trt", "fixture_deepx_setup", "deepx"),
        ("hailo8", "hailo8_to_trt", "fixture_hailo8_setup", "hailo8"),
        (
            "hailo10h", "hailo10h_to_trt",
            "fixture_hailo10h_setup", "hailo10h",
        ),
    )
    rows: list[dict[str, object]] = []
    for backend_key, _pipeline, setup_id, comparison in setup_contracts:
        for model in ("resnet50", "yolo26s", "yolov7"):
            for case_index in range(5):
                rows.append({
                    "execution_mode": "native_split",
                    "backend_key": backend_key,
                    "backend": backend_key,
                    "model": model,
                    "case": f"b{case_index:03d}",
                    "setup_id": setup_id,
                    "variant": f"{backend_key}_{model}_{case_index}",
                })
            for full_backend in (backend_key, "tensorrt"):
                rows.append({
                    "execution_mode": "native_full_baseline",
                    "backend_key": backend_key,
                    "backend": f"native_full_{full_backend}",
                    "model": model,
                    "case": "full",
                    "setup_id": setup_id,
                    "comparison_backend": comparison,
                    "variant": f"{backend_key}_{model}_full",
                })
    if len(rows) != 63:
        raise AssertionError(len(rows))
    return rows


def _observed_performance_rows_63() -> list[dict[str, object]]:
    pipeline_by_producer = {
        "deepx": "deepx_to_trt",
        "hailo8": "hailo8_to_trt",
        "hailo10h": "hailo10h_to_trt",
    }
    observed: list[dict[str, object]] = []
    for planned in _planned_performance_rows_63():
        if planned["execution_mode"] == "native_split":
            observed.append({
                "backend": pipeline_by_producer[str(planned["backend_key"])],
                "model": planned["model"],
                "case": planned["case"],
                "precision": "uint8_dequant_fp16",
                "setup_id": planned["setup_id"],
                "comparison_backend": planned["backend_key"],
                "ok": True,
                "status": "ok",
            })
        else:
            observed.append({
                "backend": planned["backend"],
                "model": planned["model"],
                "case": "full",
                "precision": "legacy_comparison_fp16",
                "execution_mode": "native_full_baseline",
                "setup_id": planned["setup_id"],
                "comparison_backend": planned["comparison_backend"],
                "ok": True,
                "status": "ok",
            })
    return observed


def _write_performance_reports(
    reports: Path,
    planned: list[dict[str, object]],
    observed: list[dict[str, object]],
) -> dict[str, object]:
    matrix = COORDINATOR._native_performance_expected_matrix(
        planned, observed,
    )
    summary = {"row_count": len(observed), "rows": observed}
    _write_json(reports / "native_producer_summary.json", summary)
    _write_json(reports / "native_producer_combined_summary.json", summary)
    _write_json(reports / "native_expected_matrix.json", matrix)
    return matrix


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.atomic_write_json(path, payload)


def _freeze_performance_checkpoint(
    run_dir: Path,
    *,
    cfg: dict[str, object],
    variants: list[dict[str, object]],
    authority: dict[str, object],
) -> tuple[Path, str, dict[str, object]]:
    reports = run_dir / "reports"
    planned_rows = _planned_performance_rows_63()
    _write_performance_reports(
        reports, planned_rows, _observed_performance_rows_63(),
    )
    stage = {
        "schema": "onnx-splitpoint/native-producer-variant-stage",
        "schema_version": 2,
        "status": "running",
        "state": "running",
        "complete": False,
        "started_at": "2026-08-01T00:00:00+0000",
        "run_id": run_dir.name,
        "config": cfg,
        "variant_results": [
            {"id": f"variant_{index}", "rc": 0, "ok": True}
            for index in range(len(variants))
        ],
        "final_report": {"rc": 0},
    }
    completion = COORDINATOR._native_performance_completion(
        reports=reports,
        expected_rows=planned_rows,
        variant_count=len(variants),
        stage=stage,
        final_report=stage["final_report"],
    )
    if completion.get("complete") is not True:
        raise AssertionError(completion)
    checkpoint_path, snapshot_path = (
        COORDINATOR._native_performance_checkpoint_paths(run_dir)
    )
    input_hash = COORDINATOR._native_performance_input_hash(
        run_dir=run_dir,
        cfg=cfg,
        variants=variants,
        expected_rows=planned_rows,
        split_quality_authority=authority,
    )
    _write_json(snapshot_path, {
        "schema": "onnx-splitpoint/native-performance-stage-snapshot",
        "schema_version": 1,
        "input_hash": input_hash,
        "completion": completion,
        "stage": stage,
    })
    COORDINATOR._freeze_native_performance_artifacts(
        reports, snapshot_path,
    )
    write_stage_checkpoint(
        checkpoint_path,
        stage="native_performance",
        state="completed",
        complete=True,
        input_hash=input_hash,
        run_root=run_dir,
        artifacts=COORDINATOR._native_performance_artifacts(snapshot_path),
        details=completion,
    )
    return checkpoint_path, input_hash, stage


class AtomicCheckpointTests(unittest.TestCase):
    def test_packaged_checkpoint_scripts_are_byte_identical(self) -> None:
        for name in (
            "run_evalrun_native_producer_variants.py",
            "run_native_producer_energy_from_summary.py",
        ):
            self.assertEqual(
                (ROOT / "scripts" / name).read_bytes(),
                (
                    ROOT / "onnx_splitpoint_tool" / "resources"
                    / "remote_scripts" / name
                ).read_bytes(),
                name,
            )

    def test_atomic_replace_failure_preserves_previous_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage_result.json"
            checkpoints.atomic_write_json(path, {"generation": 1})
            before = path.read_bytes()
            with mock.patch.object(
                checkpoints.os,
                "replace",
                side_effect=OSError("fixture replace failure"),
            ):
                with self.assertRaises(OSError):
                    checkpoints.atomic_write_json(path, {"generation": 2})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(
                list(path.parent.glob(f".{path.name}.tmp-*")), []
            )

    def test_checkpoint_writers_reject_self_invalidating_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact.json"
            _write_json(artifact, {"ok": True})
            checkpoint = root / "stage_result.json"
            with self.assertRaisesRegex(ValueError, "not unique"):
                write_stage_checkpoint(
                    checkpoint,
                    stage="fixture",
                    state="completed",
                    complete=True,
                    input_hash="fixture-input",
                    run_root=root,
                    artifacts=[artifact, artifact],
                )
            self.assertFalse(checkpoint.exists())

            journal, _rows, _identities, _hashes = _journal(
                root / "journal", 1,
            )
            with self.assertRaisesRegex(ValueError, "nonterminal"):
                journal.transition(
                    0, "running", result={"ok": True},
                )
            self.assertEqual(journal.rows[0]["state"], "not_started")

            write_stage_checkpoint(
                checkpoint,
                stage="fixture",
                state="completed",
                complete=True,
                input_hash="fixture-input",
                run_root=root,
                artifacts=[artifact],
            )
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            payload.pop("checkpoint_sha256")
            payload["schema_version"] = 2
            payload["checkpoint_sha256"] = checkpoints.canonical_json_sha256(
                payload,
            )
            _write_json(checkpoint, payload)
            loaded, reason = load_stage_checkpoint(
                checkpoint,
                stage="fixture",
                input_hash="fixture-input",
                run_root=root,
            )
            self.assertIsNone(loaded)
            self.assertEqual(reason, "checkpoint_schema_invalid")

    def test_journal_create_is_atomic_and_rejects_parallel_running_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            destination = parent / "native_energy"
            rows = _energy_rows(2)
            identities = [
                ENERGY._managed_row_identity(row) for row in rows
            ]
            row_hashes = [
                checkpoints.canonical_json_sha256({"row": row})
                for row in rows
            ]
            real_replace = os.replace

            def fail_directory_publish(source, target):
                if Path(source).is_dir():
                    raise OSError("fixture directory publish failure")
                return real_replace(source, target)

            with mock.patch.object(
                checkpoints.os,
                "replace",
                side_effect=fail_directory_publish,
            ):
                with self.assertRaises(OSError):
                    AtomicRowJournal.create(
                        destination,
                        plan_hash="frozen-plan",
                        rows=rows,
                        identities=identities,
                        row_contract_hashes=row_hashes,
                    )
            self.assertFalse(destination.exists())
            self.assertEqual(
                list(parent.glob(".native_energy.create-*")), []
            )

            journal = AtomicRowJournal.create(
                destination,
                plan_hash="frozen-plan",
                rows=rows,
                identities=identities,
                row_contract_hashes=row_hashes,
            )
            journal.transition(0, "running")
            with self.assertRaisesRegex(ValueError, "only one"):
                journal.transition(1, "running")

    def test_cancel_after_n_has_exact_partition_and_no_broken_pipe_fanout(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal, _rows, identities, row_hashes = _journal(
                root / "native_energy", 8,
            )
            completed_count = 3
            for index in range(completed_count):
                journal.transition(index, "running")
                journal.transition(
                    index,
                    "completed",
                    result={
                        "row": _energy_rows(8)[index],
                        "ok": True,
                    },
                )
            journal.transition(completed_count, "running")
            journal.transition(
                completed_count,
                "cancelled",
                reason="signal:SIGTERM",
            )
            journal.mark_run_state("cancelled")

            resumed = AtomicRowJournal.open_for_resume(
                root / "native_energy",
                plan_hash="frozen-plan",
                identities=identities,
                row_contract_hashes=row_hashes,
            )
            states = [row["state"] for row in resumed.rows]
            self.assertEqual(states.count("completed"), completed_count)
            self.assertEqual(states.count("cancelled"), 1)
            self.assertEqual(
                states.count("not_started"), 8 - completed_count - 1,
            )
            self.assertEqual(states.count("running"), 0)
            self.assertEqual(states.count("failed"), 0)
            for row in resumed.rows[completed_count + 1:]:
                self.assertEqual(row["reason"], "")
                self.assertNotIn("BrokenPipeError", json.dumps(row))

            stage_path = root / "native_energy" / "stage_result.json"
            write_stage_checkpoint(
                stage_path,
                stage="native_energy",
                state="cancelled",
                complete=False,
                input_hash="frozen-plan",
                run_root=root,
                artifacts=[resumed.manifest_path],
            )
            stage = json.loads(stage_path.read_text(encoding="utf-8"))
            self.assertEqual(stage["state"], "cancelled")
            self.assertIs(stage["complete"], False)

    def test_resume_begins_at_n_plus_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal, rows, identities, row_hashes = _journal(
                root / "native_energy", 7,
            )
            completed_count = 4
            for index in range(completed_count):
                journal.transition(index, "running")
                journal.transition(
                    index,
                    "completed",
                    result={"row": rows[index], "ok": True},
                )
            resumed = AtomicRowJournal.open_for_resume(
                root / "native_energy",
                plan_hash="frozen-plan",
                identities=identities,
                row_contract_hashes=row_hashes,
            )
            self.assertEqual(resumed.pending_indexes()[0], completed_count)
            resumed.transition(completed_count, "running")
            self.assertEqual(
                resumed.rows[completed_count]["attempt_count"], 1,
            )
            for index in range(completed_count):
                self.assertEqual(resumed.rows[index]["attempt_count"], 1)

    def test_finished_child_is_recovered_without_second_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal, rows, _identities, _row_hashes = _journal(
                root / "native_energy", 2,
            )
            output_dir = root / "measurement_0"
            _write_json(output_dir / "energy_aggregate.json", {"ok": True})
            journal.transition(0, "running")
            journal.update_running(0, execution={
                "runtime_row": rows[0],
                "output_dir": str(output_dir),
                "command": "fixture-energy-command",
                "argv": ["fixture-energy-command"],
                "process_result": {"rc": 0, "cmd": ["fixture-energy-command"]},
                "expected_runs": 1,
                "expected_effective_runs": 1,
                "expected_run_id": "fixture_run",
                "expected_setup_id": "fixture_hailo8_setup",
                "expected_command_contract_sha256": "a" * 64,
                "execution_started_ns": 1,
                "aggregate_absent_before_execution": True,
            })
            with mock.patch.object(
                ENERGY,
                "_attach_energy_aggregate",
                side_effect=lambda result, *_args, **_kwargs: {
                    **result,
                    "energy_aggregate_verified": True,
                },
            ), mock.patch.object(
                ENERGY,
                "_measurement_start_observation",
                return_value={
                    "measurement_started": True,
                    "collector_started_repeat_count": 1,
                    "workload_started_repeat_count": 1,
                },
            ):
                recovered = ENERGY._recover_managed_running_result(
                    journal.rows[0]
                )
            self.assertIsNotNone(recovered)
            journal.recover_terminal(0, result=recovered)
            self.assertEqual(journal.rows[0]["state"], "completed")
            self.assertEqual(journal.rows[0]["attempt_count"], 1)
            self.assertEqual(journal.pending_indexes(), [1])

    def test_unverified_child_aggregate_is_never_recovered_terminally(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal, rows, _identities, _row_hashes = _journal(
                root / "native_energy", 1,
            )
            output_dir = root / "measurement_0"
            _write_json(output_dir / "energy_aggregate.json", {"ok": True})
            journal.transition(0, "running")
            journal.update_running(0, execution={
                "runtime_row": rows[0],
                "output_dir": str(output_dir),
                "command": "fixture-energy-command",
                "argv": ["fixture-energy-command"],
                "process_result": {"rc": 0, "cmd": ["fixture-energy-command"]},
                "expected_runs": 1,
                "expected_effective_runs": 1,
                "expected_run_id": "fixture_run",
                "expected_setup_id": "fixture_hailo8_setup",
                "expected_command_contract_sha256": "a" * 64,
                "execution_started_ns": 1,
                "aggregate_absent_before_execution": True,
                "execution_attempt_id": "attempt-1",
            })
            with mock.patch.object(
                ENERGY,
                "_attach_energy_aggregate",
                return_value={"rc": 0, "energy_aggregate_verified": False},
            ):
                self.assertIsNone(
                    ENERGY._recover_managed_running_result(journal.rows[0])
                )
            with self.assertRaisesRegex(ValueError, "verified child"):
                journal.recover_terminal(0, result={
                    "row": rows[0],
                    "ok": True,
                    "run": {
                        "energy_aggregate_verified": False,
                        "execution_attempt_id": "attempt-1",
                    },
                })

    def test_stage_contract_forces_cancelled_complete_false(self) -> None:
        payload = StageResult(
            stage="run_native_producers",
            model_id=None,
            status="cancelled",
            state="cancelled",
            complete=True,
            started_at="start",
            finished_at="finish",
        ).to_dict()
        self.assertEqual(payload["state"], "cancelled")
        self.assertIs(payload["complete"], False)

    def test_standard_and_final_campaigns_cannot_weaken_63_row_gate(
        self,
    ) -> None:
        required = COORDINATOR._native_performance_required_campaign_rows
        with self.assertRaisesRegex(ValueError, "must be 63"):
            required({
                "native_performance_checkpoint": {"required_row_count": 62},
                "_workflow_context": {
                    "execution_preset": {"id": "standard"},
                },
            })
        self.assertEqual(required({
            "_workflow_context": {"campaign": {"mode": "final"}},
        }), 63)
        self.assertEqual(required({
            "native_performance_checkpoint": {"required_row_count": 45},
            "_workflow_context": {"campaign": {"mode": "diagnostic"}},
        }), 45)
        self.assertIsNone(required({
            "_workflow_context": {"campaign": {"mode": "diagnostic"}},
        }))

        supplement = {
            "native_performance_checkpoint": {
                "scope": "bounded_supplement",
                "required_row_count": 24,
            },
            "case_policy": "case_map_only",
            "full_baselines": {"enabled": False},
            "variants": [{
                "id": "targeted_singles",
                "full_baselines": {"enabled": False},
            }],
            "_workflow_context": {
                "execution_preset": {"id": "standard"},
            },
        }
        self.assertEqual(
            required(supplement, expected_row_count=24), 24,
        )
        with self.assertRaisesRegex(ValueError, "drifts"):
            required(supplement, expected_row_count=23)
        no_scope = copy.deepcopy(supplement)
        del no_scope["native_performance_checkpoint"]["scope"]
        with self.assertRaisesRegex(ValueError, "must be 63"):
            required(no_scope, expected_row_count=24)

    def test_performance_commit_requires_exact_unique_63_row_import(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reports = Path(temporary)
            planned = _planned_performance_rows_63()
            observed = _observed_performance_rows_63()
            stage = {"variant_results": [{"rc": 0}] * 3}

            def evaluate(
                actual_rows: list[dict[str, object]],
                expected_rows: list[dict[str, object]] | None = None,
                required_campaign_rows: int | None = 63,
            ) -> dict[str, object]:
                frozen_plan = (
                    expected_rows if expected_rows is not None else planned
                )
                _write_performance_reports(
                    reports, frozen_plan, actual_rows,
                )
                return COORDINATOR._native_performance_completion(
                    reports=reports,
                    expected_rows=frozen_plan,
                    variant_count=3,
                    stage=stage,
                    final_report={"rc": 0},
                    required_campaign_rows=required_campaign_rows,
                )

            complete = evaluate(observed)
            self.assertIs(complete["complete"], True)
            self.assertEqual(
                complete["expected_rows_by_setup"],
                {
                    "fixture_deepx_setup": 21,
                    "fixture_hailo10h_setup": 21,
                    "fixture_hailo8_setup": 21,
                },
            )
            self.assertIs(
                complete["standard_63_campaign_shape_complete"], True,
            )

            missing_row = evaluate(observed[:-1])
            self.assertIs(missing_row["complete"], False)
            self.assertEqual(missing_row["summary_row_count"], 62)
            self.assertEqual(missing_row["missing_expected_row_count"], 1)

            shortened_plan = evaluate(observed[:-1], planned[:-1])
            self.assertIs(shortened_plan["complete"], False)
            self.assertEqual(
                shortened_plan["configured_expected_row_count"], 62,
            )

            extra_planned = {
                **planned[0],
                "case": "b999",
                "variant": "deepx_resnet50_extra",
            }
            extra_observed = {
                **observed[0],
                "case": "b999",
            }
            oversized_plan = evaluate(
                [*observed, extra_observed],
                [*planned, extra_planned],
            )
            self.assertIs(oversized_plan["complete"], False)
            self.assertEqual(
                oversized_plan["configured_expected_row_count"], 64,
            )

            identity_drift_rows = [dict(row) for row in observed]
            identity_drift_rows[0]["case"] = "b999"
            identity_mismatch = evaluate(identity_drift_rows)
            self.assertIs(identity_mismatch["complete"], False)
            self.assertIs(
                identity_mismatch["planned_identity_match"], False,
            )

            setup_drift_rows = [dict(row) for row in observed]
            setup_drift_rows[0]["setup_id"] = "wrong_setup"
            setup_mismatch = evaluate(setup_drift_rows)
            self.assertIs(setup_mismatch["complete"], False)

            comparison_drift = [dict(row) for row in observed]
            comparison_drift[0]["comparison_backend"] = "hailo8"
            comparison_mismatch = evaluate(comparison_drift)
            self.assertIs(comparison_mismatch["complete"], False)

            duplicate_rows = [dict(row) for row in observed]
            duplicate_rows[-1] = {
                **duplicate_rows[0],
                "precision": "a_second_precision_same_identity",
            }
            duplicate = evaluate(duplicate_rows)
            self.assertIs(duplicate["complete"], False)

            full_case_drift = [dict(row) for row in observed]
            full_case_drift[5]["case"] = "b999"
            self.assertIs(evaluate(full_case_drift)["complete"], False)

            model_alias_conflict = [dict(row) for row in observed]
            model_alias_conflict[0]["model_id"] = "conflicting_model"
            self.assertIs(
                evaluate(model_alias_conflict)["complete"], False,
            )

            backend_alias_conflict = [dict(row) for row in observed]
            backend_alias_conflict[0]["producer_backend"] = (
                "hailo10h_to_trt"
            )
            self.assertIs(
                evaluate(backend_alias_conflict)["complete"], False,
            )

            string_false = [dict(row) for row in observed]
            string_false[0]["ok"] = "false"
            string_false_result = evaluate(string_false)
            self.assertIs(string_false_result["complete"], True)
            self.assertEqual(string_false_result["successful_row_count"], 62)

            malformed_plan = [dict(row) for row in planned]
            malformed_observed = [dict(row) for row in observed]
            malformed_plan[0]["setup_id"] = "unexpected_fourth_setup"
            malformed_observed[0]["setup_id"] = "unexpected_fourth_setup"
            wrong_campaign_shape = evaluate(
                malformed_observed, malformed_plan,
            )
            self.assertIs(wrong_campaign_shape["complete"], False)
            self.assertIs(
                wrong_campaign_shape[
                    "standard_63_campaign_shape_complete"
                ],
                False,
            )

            duplicate_producer_plan = [dict(row) for row in planned]
            duplicate_producer_observed = [dict(row) for row in observed]
            for index in range(42, 63):
                planned_row = duplicate_producer_plan[index]
                observed_row = duplicate_producer_observed[index]
                planned_row["backend_key"] = "hailo8"
                planned_row["setup_id"] = "fixture_hailo8_setup_b"
                observed_row["setup_id"] = "fixture_hailo8_setup_b"
                observed_row["comparison_backend"] = "hailo8"
                if planned_row["execution_mode"] == "native_split":
                    planned_row["backend"] = "hailo8"
                    observed_row["backend"] = "hailo8_to_trt"
                else:
                    planned_row["comparison_backend"] = "hailo8"
                    if planned_row["backend"] != "native_full_tensorrt":
                        planned_row["backend"] = "native_full_hailo8"
                        observed_row["backend"] = "native_full_hailo8"
            duplicate_producer = evaluate(
                duplicate_producer_observed, duplicate_producer_plan,
            )
            self.assertIs(duplicate_producer["complete"], False)

            model_set_plan = [dict(row) for row in planned]
            model_set_observed = [dict(row) for row in observed]
            for index in range(42, 63):
                if model_set_plan[index]["model"] == "yolov7":
                    model_set_plan[index]["model"] = "yolov7_alt"
                    model_set_observed[index]["model"] = "yolov7_alt"
            model_set_drift = evaluate(
                model_set_observed, model_set_plan,
            )
            self.assertIs(model_set_drift["complete"], False)

            generic_subset = evaluate(
                observed[:45], planned[:45], required_campaign_rows=None,
            )
            self.assertIs(generic_subset["complete"], True)

    def test_resume_reuses_performance_checkpoint_without_variant_child(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "fixture_run"
            reports = run_dir / "reports"
            reports.mkdir(parents=True)
            _write_json(run_dir / "run_manifest.json", {
                "resume_contract": {
                    "resume_contract_sha256": "b" * 64,
                },
            })
            variants = [{"id": f"variant_{index}"} for index in range(3)]
            source_cfg: dict[str, object] = {
                "variants": variants,
                "energy": {"enabled": False},
                "validation": {"enabled": False},
            }
            effective_cfg = {
                **source_cfg,
                "trt_quality_producer_sets_by_setup": {},
                "native_split_quality_binding_sets_by_variant": {},
                "variants": variants,
            }
            authority = {
                "valid": True,
                "workflow_version": "fixture",
                "tool_version": "fixture",
            }
            checkpoint_path, input_hash, _stage = (
                _freeze_performance_checkpoint(
                    run_dir,
                    cfg=effective_cfg,
                    variants=variants,
                    authority=authority,
                )
            )
            reusable, reason = load_reusable_stage_checkpoint(
                checkpoint_path,
                stage="native_performance",
                input_hash=input_hash,
                run_root=run_dir,
            )
            self.assertIsNotNone(reusable)
            self.assertEqual(reason, "reusable")

            config_path = run_dir / "native_config.json"
            _write_json(config_path, source_cfg)
            labels: list[str] = []

            def fake_run(*_args, label: str = "", **_kwargs):
                labels.append(label)
                return {"rc": 0, "stdout_tail": "", "stderr_tail": ""}

            with mock.patch.object(
                COORDINATOR,
                "_materialize_trt_quality_producer_sets",
                return_value=({}, {}),
            ), mock.patch.object(
                COORDINATOR,
                "_quality_first_variant_plan",
                return_value=variants,
            ), mock.patch.object(
                COORDINATOR,
                "_variant_expected_energy_rows",
                return_value=(
                    _planned_performance_rows_63(), {}, False, True,
                ),
            ), mock.patch.object(
                COORDINATOR,
                "build_native_energy_preflight",
                return_value={"status": "passed", "plan_viable": True},
            ), mock.patch.object(
                COORDINATOR,
                "native_energy_preflight_blocks_streaming",
                return_value=False,
            ), mock.patch.object(
                COORDINATOR,
                "resolve_native_split_quality_authority",
                return_value=authority,
            ), mock.patch.object(
                COORDINATOR,
                "native_split_quality_required_for_row",
                return_value=False,
            ), mock.patch.object(
                COORDINATOR,
                "_select_report_python",
                return_value=(sys.executable, {"fixture": True}),
            ), mock.patch.object(
                COORDINATOR,
                "_run_window_method_probe",
                return_value=({"status": "not_applicable"}, False),
            ), mock.patch.object(
                COORDINATOR,
                "_standard_quality_enforced_policy",
                return_value=False,
            ), mock.patch.object(
                COORDINATOR,
                "_run",
                side_effect=fake_run,
            ), mock.patch.object(
                sys,
                "argv",
                [
                    str(COORDINATOR.__file__),
                    "--eval-run-dir", str(run_dir),
                    "--config", str(config_path),
                    "--resume",
                ],
            ):
                self.assertEqual(COORDINATOR.main(), 0)

            self.assertFalse(
                any(label.startswith("variant:") for label in labels),
                labels,
            )
            self.assertNotIn("final_report", labels)
            restored = json.loads(
                (reports / "native_producer_summary.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(len(restored["rows"]), 63)

            labels.clear()
            with mock.patch.object(
                COORDINATOR,
                "_run",
                side_effect=AssertionError(
                    "terminal coordinator handoff started a child"
                ),
            ) as run_mock, mock.patch.object(
                sys,
                "argv",
                [
                    str(COORDINATOR.__file__),
                    "--eval-run-dir", str(run_dir),
                    "--config", str(config_path),
                    "--resume",
                ],
            ):
                self.assertEqual(COORDINATOR.main(), 0)
            run_mock.assert_not_called()

    @staticmethod
    def _outer_runner_fixture(
        run_dir: Path, config: dict[str, object],
    ):
        class _RemoteRegistry:
            cancelled = False

            @staticmethod
            def journal_environment():
                return {}

        runner = RUNNER.EvaluationWorkflowRunner(
            RUNNER.WorkflowOptions(profile="", out=str(run_dir.parent), resume=True)
        )
        runner._native_producer_config = lambda: config
        runner.profile_payload = {}
        runner.profile_path = "fixture-profile"
        runner.run_dir = run_dir
        runner.run_id = run_dir.name
        runner.session_id = "fixture-session"
        runner._remote_process_registry = _RemoteRegistry()
        runner._process_registry = None
        runner._cancel_event = threading.Event()
        runner.report_paths = []
        runner.warnings = []
        runner.log = lambda *_args, **_kwargs: None
        runner._stop_requested = False
        return runner

    def test_outer_parent_imports_terminal_handoff_without_child_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "fixture_run"
            reports = run_dir / "reports"
            reports.mkdir(parents=True)
            _write_json(run_dir / "run_manifest.json", {
                "resume_contract": {
                    "resume_contract_sha256": "c" * 64,
                },
            })
            config: dict[str, object] = {
                "enabled": True,
                "variants": [{"id": "fixture"}],
                "energy": {"enabled": False},
                "validation": {"enabled": False},
            }
            canonical_stage = reports / "native_producer_stage.json"
            _write_json(canonical_stage, {
                "schema": "onnx-splitpoint/native-producer-variant-stage",
                "status": "failed",
                "state": "failed",
                "complete": True,
                "failure_class": "fixture_terminal_decision",
                "failure_reason": "fixture_terminal_decision",
                "summary": {},
            })
            canonical_before = canonical_stage.read_bytes()
            coordinator_checkpoint = (
                run_dir / "stages" / "run_native_producers"
                / "native_coordinator" / "stage_result.json"
            )
            write_stage_checkpoint(
                coordinator_checkpoint,
                stage="native_coordinator",
                state="failed",
                complete=True,
                input_hash="fixed-coordinator-input",
                run_root=run_dir,
                artifacts=[canonical_stage],
                details={
                    "return_code": 2,
                    "stage_status": "failed",
                    "stage_state": "failed",
                    "stage_complete": True,
                },
            )
            runner = self._outer_runner_fixture(run_dir, config)
            with mock.patch.object(
                RUNNER,
                "native_coordinator_input_hash",
                return_value="fixed-coordinator-input",
            ), mock.patch.object(
                RUNNER,
                "run_streaming",
                side_effect=AssertionError(
                    "outer parent restarted terminal coordinator"
                ),
            ) as stream_mock:
                _artifacts, details, _message, status = (
                    runner._stage_run_native_producers()
                )
            stream_mock.assert_not_called()
            self.assertEqual(status, "failed")
            self.assertIs(details["coordinator_handoff_reused"], True)
            self.assertEqual(canonical_stage.read_bytes(), canonical_before)
            parent_import = json.loads(
                (reports / "native_producer_parent_import.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertIs(
                parent_import["parent_import"]["handoff_reused"], True,
            )

    def test_outer_parent_keeps_incomplete_child_stage_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "fixture_run"
            reports = run_dir / "reports"
            reports.mkdir(parents=True)
            _write_json(run_dir / "run_manifest.json", {
                "resume_contract": {
                    "resume_contract_sha256": "d" * 64,
                },
            })
            config: dict[str, object] = {
                "enabled": True,
                "variants": [{"id": "fixture"}],
                "energy": {"enabled": False},
                "validation": {"enabled": False},
            }
            canonical_stage = reports / "native_producer_stage.json"
            _write_json(canonical_stage, {
                "schema": "onnx-splitpoint/native-producer-variant-stage",
                "status": "running",
                "state": "running",
                "complete": False,
            })
            canonical_before = canonical_stage.read_bytes()
            coordinator_checkpoint = (
                run_dir / "stages" / "run_native_producers"
                / "native_coordinator" / "stage_result.json"
            )
            write_stage_checkpoint(
                coordinator_checkpoint,
                stage="native_coordinator",
                state="running",
                complete=False,
                input_hash="fixed-coordinator-input",
                run_root=run_dir,
                artifacts=[canonical_stage],
            )
            runner = self._outer_runner_fixture(run_dir, config)
            completed = RUNNER.StreamingCompletedProcess(
                ["fixture"], 2, "", "", 0.0,
            )
            with mock.patch.object(
                RUNNER,
                "native_coordinator_input_hash",
                return_value="fixed-coordinator-input",
            ), mock.patch.object(
                RUNNER,
                "run_streaming",
                return_value=completed,
            ) as stream_mock:
                _artifacts, details, _message, status = (
                    runner._stage_run_native_producers()
                )
            stream_mock.assert_called_once()
            self.assertEqual(status, "cancelled")
            self.assertIs(details["child_complete"], False)
            self.assertIs(runner._stop_requested, True)
            self.assertEqual(canonical_stage.read_bytes(), canonical_before)

    def test_managed_plan_hash_ignores_new_attempt_id(self) -> None:
        rows = _energy_rows(3)
        first = {
            "measurement_plan_attempt_id": "attempt-a",
            "duration_s": 60.0,
            "runs": 3,
        }
        second = {
            **first,
            "measurement_plan_attempt_id": "attempt-b",
        }
        first_contract = ENERGY._managed_journal_contract(rows, first)
        second_contract = ENERGY._managed_journal_contract(rows, second)
        self.assertEqual(first_contract, second_contract)

    def test_cancel_during_energy_planning_commits_incomplete_stage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "energy"
            summary = root / "summary.json"
            _write_json(summary, {"rows": []})
            with mock.patch.object(
                ENERGY,
                "_run",
                side_effect=ENERGY._ManagedEnergyCancelled(
                    signal.SIGTERM, "signal:SIGTERM",
                ),
            ), mock.patch.object(
                sys,
                "argv",
                [
                    str(ENERGY.__file__),
                    "--summary", str(summary),
                    "--out-dir", str(out),
                ],
            ):
                self.assertEqual(ENERGY.main(), 130)
            stage = json.loads(
                (
                    out / "stages" / "native_energy"
                    / "stage_result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(stage["state"], "cancelled")
            self.assertIs(stage["complete"], False)
            self.assertEqual(stage["details"]["return_code"], 130)

    def test_cancel_during_resume_recovery_cancels_only_running_row(
        self,
    ) -> None:
        cases = (
            (
                "SIGTERM",
                lambda: ENERGY._ManagedEnergyCancelled(
                    signal.SIGTERM, "signal:SIGTERM",
                ),
                "signal:SIGTERM",
            ),
            (
                "SIGINT",
                lambda: ENERGY._ManagedEnergyCancelled(
                    signal.SIGINT, "signal:SIGINT",
                ),
                "signal:SIGINT",
            ),
            (
                "BrokenPipeError",
                lambda: BrokenPipeError("fixture parent pipe closed"),
                "parent_pipe_closed",
            ),
        )
        for label, exception_factory, expected_reason in cases:
            with (
                self.subTest(interruption=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
                out = Path(temporary) / "energy"
                journal_root = out / "checkpoints" / "native_energy"
                journal, _rows, identities, row_hashes = _journal(
                    journal_root, 5,
                )
                journal.transition(0, "running", execution={
                    "phase": "measurement_child_starting",
                    "row_index": 0,
                })
                stage_path = (
                    out / "stages" / "native_energy" / "stage_result.json"
                )
                write_stage_checkpoint(
                    stage_path,
                    stage="native_energy",
                    state="running",
                    complete=False,
                    input_hash="frozen-input",
                    run_root=out,
                    artifacts=[journal.manifest_path],
                )

                def interrupt_resume_recovery():
                    ENERGY._ACTIVE_MANAGED_STAGE_CONTEXT = {
                        "out": out,
                        "checkpoint": stage_path,
                        "journal": journal_root,
                        "input_hash": "frozen-input",
                        "resume_checkpoint": True,
                        "managed_plan_hash": "frozen-plan",
                        "managed_identities": identities,
                        "managed_row_hashes": row_hashes,
                    }
                    raise exception_factory()

                with mock.patch.object(
                    ENERGY,
                    "_main_impl",
                    side_effect=interrupt_resume_recovery,
                ):
                    self.assertEqual(ENERGY.main(), 130)

                resumed = AtomicRowJournal.open_for_resume(
                    journal_root,
                    plan_hash="frozen-plan",
                    identities=identities,
                    row_contract_hashes=row_hashes,
                )
                states = [row["state"] for row in resumed.rows]
                self.assertEqual(states, [
                    "cancelled",
                    "not_started",
                    "not_started",
                    "not_started",
                    "not_started",
                ])
                self.assertEqual(
                    resumed.rows[0]["reason"], expected_reason,
                )
                self.assertNotIn(
                    "BrokenPipeError", json.dumps(resumed.rows[1:]),
                )
                stage = json.loads(stage_path.read_text(encoding="utf-8"))
                self.assertEqual(stage["state"], "cancelled")
                self.assertIs(stage["complete"], False)
                self.assertEqual(
                    stage["details"]["journal_cancel_error"], "",
                )

    def test_blocked_energy_plan_journals_every_row_not_started(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "energy"
            summary = root / "summary.json"
            _write_json(summary, {"rows": []})
            rows = _energy_rows(5)

            def blocked_plan(_cmd, *, label: str = "", **_kwargs):
                self.assertEqual(label, "energy_plan")
                _write_json(out / "plan" / "native_producer_energy_plan.json", {
                    "measurement_plan_attempt_id": "blocked-fixture",
                    "preflight_status": "blocked",
                    "preflight": {
                        "status": "blocked",
                        "ok": False,
                        "measurement_start_allowed": False,
                        "energy_plan_coverage_contract_valid": False,
                        "blocked_reason": "fixture_preflight_block",
                    },
                    "rows": rows,
                })
                return {"rc": 0, "elapsed_s": 0.0}

            with mock.patch.object(
                ENERGY, "_run", side_effect=blocked_plan,
            ), mock.patch.object(
                sys,
                "argv",
                [
                    str(ENERGY.__file__),
                    "--summary", str(summary),
                    "--out-dir", str(out),
                ],
            ):
                self.assertEqual(ENERGY.main(), 4)
            manifest = json.loads(
                (
                    out / "checkpoints" / "native_energy" / "journal.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["row_count"], len(rows))
            self.assertEqual(manifest["counts"]["not_started"], len(rows))
            self.assertEqual(manifest["counts"]["running"], 0)
            self.assertEqual(manifest["state"], "failed")
            self.assertIs(manifest["complete"], False)

    def test_real_energy_loop_breaks_at_n_and_resumes_exact_next_row(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "energy"
            summary = root / "summary.json"
            _write_json(summary, {"rows": []})
            rows = _energy_rows(7)
            completed_before_cancel = 3
            invocation = {"number": 1, "energy_calls": 0}
            actual_execution_indexes: list[int] = []

            def plan_payload() -> dict[str, object]:
                return {
                    "measurement_plan_attempt_id": (
                        f"attempt-{invocation['number']}"
                    ),
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

            def fake_run(_cmd, *, label: str = "", **_kwargs):
                if label == "energy_plan":
                    _write_json(
                        out / "plan" / "native_producer_energy_plan.json",
                        plan_payload(),
                    )
                    return {"rc": 0, "elapsed_s": 0.0}
                if label.startswith("energy:"):
                    invocation["energy_calls"] += 1
                    if (
                        invocation["number"] == 1
                        and invocation["energy_calls"]
                        == completed_before_cancel + 1
                    ):
                        raise BrokenPipeError("fixture parent closed")
                    return {"rc": 0, "elapsed_s": 0.001}
                raise AssertionError(label)

            def prepare(
                row, _plan, *, allowed_root, validate_only=False,
            ):
                index = int(str(row["case"])[1:])
                if not validate_only:
                    actual_execution_indexes.append(index)
                output_dir = Path(allowed_root) / f"measurement_{index}"
                output_dir.mkdir(parents=True, exist_ok=True)
                return {
                    "row": dict(row),
                    "output_dir": output_dir,
                    "command": "fixture-energy-command",
                    "argv": ["fixture-energy-command", str(index)],
                    "expected_runs": 1,
                    "expected_effective_runs": 1,
                    "expected_run_id": "fixture_run",
                    "expected_setup_id": "fixture_hailo8_setup",
                    "expected_command_contract_sha256": "a" * 64,
                    "execution_attempt_id": (
                        f"execution-{invocation['number']}-{index}"
                    ),
                    "split_quality_binding_status": "fixture",
                }

            def attach(result, *_args, **_kwargs):
                return {**result, "energy_aggregate_verified": True}

            observation = {
                "measurement_started": True,
                "collector_started_repeat_count": 1,
                "workload_started_repeat_count": 1,
            }
            base_argv = [
                str(ENERGY.__file__),
                "--summary", str(summary),
                "--out-dir", str(out),
                "--duration-s", "1",
            ]
            with mock.patch.object(
                ENERGY, "_run", side_effect=fake_run,
            ), mock.patch.object(
                ENERGY,
                "_prepare_measurement_execution",
                side_effect=prepare,
            ), mock.patch.object(
                ENERGY, "_attach_energy_aggregate", side_effect=attach,
            ), mock.patch.object(
                ENERGY,
                "_measurement_start_observation",
                return_value=observation,
            ), mock.patch.object(sys, "argv", base_argv):
                self.assertEqual(ENERGY.main(), 130)

            journal_root = out / "checkpoints" / "native_energy"
            first_manifest = json.loads(
                (journal_root / "journal.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first_manifest["counts"], {
                "cancelled": 1,
                "completed": completed_before_cancel,
                "failed": 0,
                "not_started": len(rows) - completed_before_cancel - 1,
                "running": 0,
            })
            for entry in first_manifest["rows"][completed_before_cancel + 1:]:
                checkpoint = json.loads(
                    (journal_root / entry["checkpoint"]).read_text(
                        encoding="utf-8",
                    )
                )
                self.assertEqual(checkpoint["state"], "not_started")
                self.assertNotIn("BrokenPipeError", json.dumps(checkpoint))

            invocation.update(number=2, energy_calls=0)
            actual_execution_indexes.clear()
            with mock.patch.object(
                ENERGY, "_run", side_effect=fake_run,
            ), mock.patch.object(
                ENERGY,
                "_prepare_measurement_execution",
                side_effect=prepare,
            ), mock.patch.object(
                ENERGY, "_attach_energy_aggregate", side_effect=attach,
            ), mock.patch.object(
                ENERGY,
                "_measurement_start_observation",
                return_value=observation,
            ), mock.patch.object(
                sys, "argv", [*base_argv, "--resume-checkpoint"],
            ):
                self.assertEqual(ENERGY.main(), 0)

            self.assertEqual(
                actual_execution_indexes[0], completed_before_cancel,
            )
            final_manifest = json.loads(
                (journal_root / "journal.json").read_text(encoding="utf-8")
            )
            self.assertTrue(final_manifest["complete"])
            self.assertEqual(final_manifest["counts"]["completed"], len(rows))
            self.assertEqual(final_manifest["counts"]["not_started"], 0)
            self.assertEqual(final_manifest["counts"]["cancelled"], 0)

            with mock.patch.object(
                ENERGY,
                "_run",
                side_effect=AssertionError(
                    "terminal Energy checkpoint reran the planner"
                ),
            ) as run_mock, mock.patch.object(
                sys, "argv", [*base_argv, "--resume-checkpoint"],
            ):
                self.assertEqual(ENERGY.main(), 0)
            run_mock.assert_not_called()

            with mock.patch.object(
                ENERGY,
                "_run",
                side_effect=AssertionError("drifted resume started planner"),
            ) as run_mock, mock.patch.object(
                sys,
                "argv",
                [
                    *base_argv,
                    "--hailo8-ssh", "fixture-drift",
                    "--resume-checkpoint",
                ],
            ):
                with self.assertRaises(SystemExit):
                    ENERGY.main()
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
