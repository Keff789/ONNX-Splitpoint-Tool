from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from onnx_splitpoint_tool.run_modes import RUN_MODE_SCHEMA_VERSION, apply_run_mode, default_run_modes_config, load_run_modes_config, validate_run_modes_config
from onnx_splitpoint_tool.workflow.hailo_remote_binding import _discover_hefs, _infer_backend
from onnx_splitpoint_tool.workflow.runner import (
    _benchmark_stage_status_v60p,
    _campaign_stage_status_v60p,
)


class V60PSmokeWarningFixTests(unittest.TestCase):
    def _profile(self) -> dict:
        cfg = default_run_modes_config()
        return {
            "name": "smoke-v60p",
            "model_suite": {"primary": [{"id": "resnet50", "task": "classification", "enabled": True}]},
            "selection_policy": {
                "max_accepted_cases_per_model": 1,
                "preferred_shortlist": 1,
                "selection_strategy": "stratified_windows",
            },
            "run_profiles": [{"id": "hailo8_to_trt", "stage1": "hailo8", "stage2": "tensorrt", "enabled": True}],
            "execution_preset": {
                "id": "smoke",
                "follow_tool_config": False,
                "snapshot": cfg["modes"]["smoke"],
                "overrides": {"native_enabled": False, "energy_enabled": False},
            },
        }

    def test_smoke_defaults_use_small_registered_validation_and_safe_timeout(self) -> None:
        profile, _audit = apply_run_mode(self._profile())
        snap = profile["execution_preset"]["snapshot"]
        self.assertTrue(snap["data"]["use_final_dataset_registry"])
        self.assertEqual(snap["data"]["validation_items"], {"classification": 16, "detection": 12})
        self.assertEqual(snap["quality"]["bootstrap_repetitions"], 25)
        self.assertEqual(snap["runtime"]["benchmark"]["timeout_s"], 3600)

    def test_v60o_smoke_defaults_migrate_without_touching_custom_values(self) -> None:
        old = default_run_modes_config()
        old["schema_version"] = 1
        smoke = old["modes"]["smoke"]
        smoke["campaign"]["auto_bind_dataset_registry"] = False
        smoke["data"]["use_final_dataset_registry"] = False
        smoke["data"]["validation_items"] = {"classification": 32, "detection": 25}
        smoke["quality"]["bootstrap_repetitions"] = 50
        smoke["quality"]["profile_id"] = "task_quality_development_50"
        smoke["ranking"]["bootstrap_repetitions"] = 100
        smoke["runtime"]["benchmark"]["timeout_s"] = 600
        migrated = validate_run_modes_config(old)
        self.assertEqual(migrated["schema_version"], RUN_MODE_SCHEMA_VERSION)
        out = migrated["modes"]["smoke"]
        self.assertTrue(out["data"]["use_final_dataset_registry"])
        self.assertEqual(out["data"]["validation_items"], {"classification": 16, "detection": 12})
        self.assertEqual(out["quality"]["bootstrap_repetitions"], 25)
        self.assertEqual(out["runtime"]["benchmark"]["timeout_s"], 3600)

    def test_schema_v2_smoke_defaults_are_migrated_to_v60q_budgets(self) -> None:
        old = default_run_modes_config()
        old["schema_version"] = 2
        smoke = old["modes"]["smoke"]
        smoke["campaign"]["auto_bind_dataset_registry"] = False
        smoke["data"]["use_final_dataset_registry"] = False
        smoke["data"]["validation_items"] = {"classification": 32, "detection": 25}
        smoke["quality"]["bootstrap_repetitions"] = 50
        smoke["quality"]["profile_id"] = "task_quality_development_50"
        smoke["ranking"]["bootstrap_repetitions"] = 100
        smoke["runtime"]["benchmark"]["timeout_s"] = 600
        migrated = validate_run_modes_config(old)
        self.assertEqual(migrated["schema_version"], RUN_MODE_SCHEMA_VERSION)
        out = migrated["modes"]["smoke"]
        self.assertTrue(out["campaign"]["auto_bind_dataset_registry"])
        self.assertTrue(out["data"]["use_final_dataset_registry"])
        self.assertEqual(out["data"]["validation_items"], {"classification": 16, "detection": 12})
        self.assertEqual(out["quality"]["bootstrap_repetitions"], 25)
        self.assertEqual(out["ranking"]["bootstrap_repetitions"], 50)
        self.assertEqual(out["runtime"]["benchmark"]["timeout_s"], 3600)

    def test_schema_v8_untouched_smoke_timeout_migrates_without_overwriting_custom_value(self) -> None:
        old = default_run_modes_config()
        old["schema_version"] = 8
        old["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"] = 1800
        self.assertEqual(
            validate_run_modes_config(old)["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"],
            3600,
        )

        old["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"] = 2700
        self.assertEqual(
            validate_run_modes_config(old)["modes"]["smoke"]["runtime"]["benchmark"]["timeout_s"],
            2700,
        )

    def test_schema_v2_migration_is_persisted_only_by_explicit_save(self) -> None:
        import yaml
        from onnx_splitpoint_tool.run_modes import save_run_modes_config, run_modes_revision
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run_modes.yaml"
            old = default_run_modes_config()
            old["schema_version"] = 2
            smoke = old["modes"]["smoke"]
            smoke["data"]["validation_items"] = {"classification": 32, "detection": 25}
            smoke["quality"]["bootstrap_repetitions"] = 50
            smoke["ranking"]["bootstrap_repetitions"] = 100
            smoke["runtime"]["benchmark"]["timeout_s"] = 600
            path.write_text(yaml.safe_dump(old, sort_keys=False), encoding="utf-8")
            before = path.read_bytes()
            loaded = load_run_modes_config(path)
            self.assertEqual(loaded["schema_version"], RUN_MODE_SCHEMA_VERSION)
            self.assertEqual(path.read_bytes(), before)
            save_run_modes_config(path, loaded, expected_revision=run_modes_revision(loaded))
            persisted = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["schema_version"], RUN_MODE_SCHEMA_VERSION)
            self.assertEqual(persisted["modes"]["smoke"]["data"]["validation_items"], {"classification": 16, "detection": 12})

    def test_schema_v7_smoke_local_default_migrates_to_management_quality(self) -> None:
        old = default_run_modes_config()
        old["schema_version"] = 7
        old["modes"]["smoke"]["quality"]["execution_location"] = "local"
        old["modes"]["smoke"]["quality"]["workers"] = 1

        migrated = validate_run_modes_config(old)

        quality = migrated["modes"]["smoke"]["quality"]
        self.assertEqual(migrated["schema_version"], RUN_MODE_SCHEMA_VERSION)
        self.assertEqual(quality["execution_location"], "central_management")
        self.assertEqual(quality["workers"], 4)

    def test_schema_v7_smoke_explicit_custom_execution_contract_is_preserved(self) -> None:
        old = default_run_modes_config()
        old["schema_version"] = 7
        old["modes"]["smoke"]["quality"]["execution_location"] = "local"
        old["modes"]["smoke"]["quality"]["workers"] = 6

        quality = validate_run_modes_config(old)["modes"]["smoke"]["quality"]

        self.assertEqual(quality["execution_location"], "local")
        self.assertEqual(quality["workers"], 6)

    def test_development_ready_is_not_a_job_warning(self) -> None:
        self.assertEqual(_campaign_stage_status_v60p("development_ready"), "ok")
        self.assertEqual(_campaign_stage_status_v60p("screening_ready"), "ok")
        self.assertEqual(_campaign_stage_status_v60p("blocked"), "warn")

    def test_partial_executor_remains_partial_even_with_measured_rows(self) -> None:
        self.assertEqual(
            _benchmark_stage_status_v60p(normalized_row_count=3, executor_status="partial", executor_metrics={}),
            "partial",
        )
        self.assertEqual(
            _benchmark_stage_status_v60p(normalized_row_count=3, executor_status="ok", executor_metrics={}),
            "ok",
        )
        self.assertEqual(
            _benchmark_stage_status_v60p(
                normalized_row_count=2,
                executor_status="ok",
                executor_metrics={"remote_dispatch_failed": True},
            ),
            "partial",
        )

    def test_specific_hailo_architecture_wins_over_generic_parent_directory(self) -> None:
        self.assertEqual(_infer_backend(Path("suite/b052/hailo/hailo10/part1/compiled.hef")), "hailo10h")
        self.assertEqual(_infer_backend(Path("suite/b052/hailo/hailo8/part1/compiled.hef")), "hailo8")

    def test_discovered_hefs_keep_hailo8_and_hailo10_separate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            suite = root / "suite"
            for arch in ("hailo8", "hailo10"):
                p = suite / "b052" / "hailo" / arch / "part1" / "compiled.hef"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(arch.encode("ascii"))
            rows = _discover_hefs(suite, root, [])
            self.assertEqual({row["backend"] for row in rows}, {"hailo8", "hailo10h"})


if __name__ == "__main__":
    unittest.main()
