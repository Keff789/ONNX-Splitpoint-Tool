from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
    save_evaluation_profile_yaml,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.gui.run_mode_editor import _SECTIONS
from onnx_splitpoint_tool.run_modes import (
    apply_run_mode,
    default_run_modes_config,
    load_run_modes_config,
    mode_summary,
    save_run_modes_config,
    validate_run_modes_config,
)
from onnx_splitpoint_tool.workflow.hardware_matrix import normalize_hardware_targets
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import _mode_task_item_count


class RunModesV60NTests(unittest.TestCase):
    def _profile(self, mode: str = "standard", *, native: bool = True, energy: bool = False) -> dict:
        cfg = default_run_modes_config()
        return {
            "name": f"test_{mode}",
            "purpose": "unit test",
            "selection_policy": {
                "max_accepted_cases_per_model": 2,
                "preferred_shortlist": 5,
                "min_gap": 1,
                "candidate_search_pool": "auto",
                "selection_strategy": "stratified_windows",
            },
            "model_suite": {"primary": [{"id": "resnet50", "task": "classification", "evaluation_role": "development"}]},
            "run_profiles": [
                {"id": "ort_tensorrt", "type": "same_backend_reference", "full": "tensorrt", "stage1": "tensorrt", "stage2": "tensorrt", "required": True},
                {"id": "hailo8_to_trt", "type": "mixed_backend", "stage1": "hailo8", "stage2": "tensorrt", "required": False},
            ],
            "execution_preset": {
                "id": mode,
                "follow_tool_config": False,
                "snapshot": cfg["modes"][mode],
                "overrides": {"native_enabled": native, "energy_enabled": energy},
            },
        }

    def test_packaged_modes_are_distinct_and_ordered(self) -> None:
        config = validate_run_modes_config(default_run_modes_config())
        self.assertEqual(list(config["modes"]), ["smoke", "standard", "final"])
        self.assertEqual(config["default_mode"], "standard")
        self.assertEqual(config["modes"]["smoke"]["data"]["calibration_items"]["classification"], 8)
        self.assertEqual(config["modes"]["standard"]["data"]["validation_items"]["classification"], 500)
        self.assertEqual(config["modes"]["final"]["data"]["validation_items"]["classification"], 5000)
        self.assertEqual(config["modes"]["final"]["quality"]["bootstrap_repetitions"], 5000)

    def test_run_mode_editor_exposes_every_packaged_leaf(self) -> None:
        mode = default_run_modes_config()["modes"]["standard"]
        leaves: set[str] = set()
        def walk(value, prefix=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, f"{prefix}.{key}" if prefix else key)
            else:
                leaves.add(prefix)
        walk(mode)
        shown = {spec[1] for _title, specs in _SECTIONS for spec in specs}
        optional = {
            spec[1] for _title, specs in _SECTIONS for spec in specs
            if spec[2] == "optional_manifest"
        }
        # This explicitly selectable field must be visible without inserting
        # a default empty value: omission inherits a legacy environment opt-in,
        # whereas a user-selected empty value deliberately disables it.
        self.assertEqual(optional, {
            "build.hailo.compute_by_family.hailo8.dependency_manifest",
        })
        self.assertFalse(leaves & optional)
        self.assertEqual(leaves, shown - optional)

    def test_smoke_standard_final_materialisation(self) -> None:
        smoke, _ = apply_run_mode(self._profile("smoke", native=False, energy=False))
        standard, _ = apply_run_mode(self._profile("standard", native=True, energy=False))
        final, _ = apply_run_mode(self._profile("final", native=True, energy=True))
        for profile in (smoke, standard, final):
            native_detection = profile["quality_gate"]["native_contract"][
                "detection_self_reference"
            ]
            self.assertEqual(
                native_detection["min_reference_match_ratio"], 0.80,
            )
            self.assertEqual(
                native_detection["min_mean_matched_iou"], 0.85,
            )
        self.assertEqual(smoke["benchmark_execution"]["runs"], 1)
        self.assertLessEqual(smoke["quality_gate"]["statistics"]["bootstrap_repetitions"], 50)
        self.assertEqual(smoke["quality_gate"]["statistics"]["execution_location"], "central_management")
        self.assertEqual(smoke["quality_gate"]["statistics"]["workers"], 4)
        self.assertEqual(standard["validation_execution"]["max_items"], {"classification": 500, "detection": 500})
        self.assertEqual(final["validation_execution"]["max_items"], {"classification": 5000, "detection": 5000})
        self.assertEqual(smoke["integrity_policy"]["mode"], "fast")
        self.assertEqual(standard["integrity_policy"]["mode"], "relaxed")
        self.assertEqual(final["integrity_policy"]["mode"], "relaxed")
        self.assertFalse(smoke["workflow"]["no_model_hash"])
        self.assertFalse(standard["workflow"]["no_model_hash"])
        self.assertFalse(final["workflow"]["no_model_hash"])
        self.assertEqual(final["campaign"]["mode"], "development")
        self.assertEqual(final["campaign"]["enforcement"], "warn")
        self.assertFalse(final["official_coco_evaluation"]["required_for_final"])

    def test_global_energy_switch_always_disables_native_energy(self) -> None:
        profile, _ = apply_run_mode(self._profile("final", native=True, energy=False))
        self.assertFalse(profile["energy"]["enabled"])
        self.assertTrue(profile["native_producers"]["enabled"])
        self.assertFalse(profile["native_producers"]["energy"]["enabled"])
        self.assertEqual(profile["native_producers"]["energy"]["mode"], "plan")

    def test_profile_does_not_embed_remote_setup_complexity(self) -> None:
        source = self._profile("standard")
        source["remote_execution"] = {"host": "should-not-survive"}
        source["hardware_setups"] = [{"id": "inline"}]
        source["build_environments"] = [{"id": "inline-build"}]
        profile, _ = apply_run_mode(source)
        for key in ("remote_execution", "hardware_setups", "hardware_groups", "hardware_targets", "build_environments"):
            self.assertNotIn(key, profile)
        self.assertEqual(profile["hardware"], {"selected_setups": [], "selected_groups": []})

    def test_central_run_mode_file_can_be_customised(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run_modes.yaml"
            config = default_run_modes_config()
            config["modes"]["smoke"]["quality"]["bootstrap_repetitions"] = 7
            config["modes"]["smoke"]["data"]["validation_items"]["classification"] = 9
            save_run_modes_config(path, config)
            loaded = load_run_modes_config(path)
            source = self._profile("smoke", native=False, energy=False)
            source["execution_preset"] = {"id": "smoke", "follow_tool_config": True, "overrides": {"native_enabled": False, "energy_enabled": False}}
            resolved, audit = apply_run_mode(source, config=loaded, config_path=path)
            self.assertEqual(resolved["quality_gate"]["statistics"]["bootstrap_repetitions"], 7)
            self.assertEqual(resolved["validation_execution"]["max_items"]["classification"], 9)
            self.assertEqual(audit["mode_id"], "smoke")

    def test_hardware_target_is_derived_from_central_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            registry = Path(temp) / "hardware.yaml"
            registry.write_text(yaml.safe_dump({
                "hardware_setups": [{
                    "id": "central_h8",
                    "accelerator": "hailo8",
                    "enabled": True,
                    "host": {"address": "192.0.2.4", "user": "nx", "port": 22},
                    "runtime": {"provider": "hailo8", "activate": "source ~/venv/bin/activate"},
                    "build": {"environment_id": "h8"},
                }],
                "build_environments": [{"id": "h8", "kind": "hailo8_dfc"}],
            }, sort_keys=False), encoding="utf-8")
            profile, _ = apply_run_mode(self._profile("smoke", native=False, energy=False))
            profile["hardware"]["setups_file"] = str(registry)
            targets = normalize_hardware_targets(profile)
            self.assertEqual([row["id"] for row in targets], ["central_h8"])
            self.assertEqual(targets[0]["runtime"]["host"], "192.0.2.4")

    def test_task_specific_calibration_count_follows_mode(self) -> None:
        profile, _ = apply_run_mode(self._profile("standard"))
        self.assertEqual(_mode_task_item_count(profile, "classification", kind="calibration_items", fallback=1), 500)
        self.assertEqual(_mode_task_item_count(profile, "detection", kind="calibration_items", fallback=1), 500)

    def test_materialized_profile_is_schema_valid(self) -> None:
        for mode in ("smoke", "standard", "final"):
            profile, _ = apply_run_mode(self._profile(mode))
            validated = validate_evaluation_profile_payload(profile, source=f"test {mode}")
            self.assertEqual(validated["execution_preset"]["id"], mode)

    def test_saved_profile_reloads_current_central_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mode_file = root / "run_modes.yaml"
            profile_file = root / "profile.yaml"
            config = default_run_modes_config()
            save_run_modes_config(mode_file, config)
            source = self._profile("standard")
            source["execution_preset"] = {"id": "standard", "follow_tool_config": True, "config_path": str(mode_file), "overrides": {"native_enabled": True, "energy_enabled": False}}
            with mock.patch.dict("os.environ", {"ONNX_SPLITPOINT_RUN_MODES_FILE": str(mode_file)}, clear=False):
                save_evaluation_profile_yaml(profile_file, source)
                config["modes"]["standard"]["runtime"]["benchmark"]["runs"] = 2
                save_run_modes_config(mode_file, config)
                loaded = load_evaluation_profile(profile_file)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.raw_profile["benchmark_execution"]["runs"], 2)

    def test_summary_hides_internal_complexity_but_reports_effort(self) -> None:
        text = mode_summary("final", default_run_modes_config())
        self.assertIn("Validation: CLS 5000 / DET 5000", text)
        self.assertIn("Bootstrap: 5000", text)
        self.assertIn("Reproducibility: relaxed", text)


if __name__ == "__main__":
    unittest.main()
