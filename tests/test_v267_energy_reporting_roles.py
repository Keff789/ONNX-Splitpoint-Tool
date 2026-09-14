from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from onnx_splitpoint_tool.campaign import build_campaign_readiness
from onnx_splitpoint_tool.native_energy_reporting import (
    build_native_energy_ab_aggregates,
    collect_native_energy,
)


def _write_report(root: Path, aggregate: dict) -> None:
    path = (
        root / "reports" / "native_energy_measurements"
        / "native_producer_energy_results.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "rows": [{
            "ok": True,
            "row": {
                "backend": "hailo8_to_trt",
                "model": "resnet50",
                "case": "b1",
                "precision": "p",
                "setup_id": "h8",
                "claim_ok": True,
                "contract_consistent": True,
                "task": "classification",
                "prepared_feed_task": "classification",
                "prepared_feed_preprocess_mode": "resize",
                "prepared_feed_letterbox_pad_value": 0,
                "prepared_feed_source_image_sha256": "a" * 64,
            },
            "run": {
                "rc": 0,
                "energy_aggregate": aggregate,
                "energy_aggregate_embedded": True,
            },
        }],
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def _measurement_contract() -> dict:
    return {
        "status": "ok",
        "run_count": 3,
        "valid_postprocessed_runs": 3,
        "raw_postprocessed_run_count": 3,
        "confidence_level": 0.95,
        "avg_energy_work_units_used": 100,
        "energy_work_units_source": "runtime_completed_work_units",
        "runtime_completed_work_unit_run_count": 3,
        "energy_efficiency_claim_eligible": True,
        "final_energy_gate_status": "pass",
        "postprocess_status": "ok",
        "energy_window_effective_values": ["command_window"],
        "energy_window_requested": "command",
        "energy_physical_scope": "MB",
        "energy_primary_metric": "calibrated_input_energy_unsubtracted",
        "energy_calibrated_input_unsubtracted": True,
        "energy_raw_primary": True,
    }


class EnergyReportingRoleTests(unittest.TestCase):
    def test_final_profile_and_readiness_use_role_explicit_methods(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile_path = (
            root / "onnx_splitpoint_tool" / "resources" / "evaluation_profiles"
            / "thesis_final_campaign_v1.yaml"
        )
        schema_path = (
            root / "onnx_splitpoint_tool" / "resources" / "schemas"
            / "evaluation_profile.schema.json"
        )
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        contract = profile["energy"]["window_method_ab"]
        self.assertEqual(contract["primary_method"], "command_marker_window")
        self.assertEqual(contract["shadow_method"], "chapter4_legacy_window")
        self.assertNotIn("baseline_method", contract)
        self.assertNotIn("candidate_method", contract)

        # Locate the schema block directly without depending on the optional
        # jsonschema package in minimal source-only installations.
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        energy_properties = schema["properties"]["energy"]["properties"]
        method_properties = energy_properties["window_method_ab"]["properties"]
        self.assertEqual(
            method_properties["primary_method"]["enum"], ["command_marker_window"]
        )
        self.assertEqual(
            method_properties["shadow_method"]["enum"], ["chapter4_legacy_window"]
        )
        self.assertTrue(method_properties["baseline_method"]["deprecated"])
        self.assertTrue(method_properties["candidate_method"]["deprecated"])

        readiness = build_campaign_readiness(profile, profile_path=profile_path)
        check = next(
            item for item in readiness["checks"]
            if item.get("id") == "energy_window_method_ab"
        )
        self.assertEqual(check["status"], "pass")
        self.assertIn("non-blocking sensitivity shadow", check["detail"])

    def test_marker_primary_remains_claimable_when_chapter4_shadow_fails(self) -> None:
        aggregate = {
            **_measurement_contract(),
            "scientific_primary_method_frozen": True,
            "scientific_primary_method": "command_marker_window",
            "scientific_primary_energy_status": "available",
            "scientific_primary_claim_eligible": True,
            "scientific_primary_energy_total_j": 10.0,
            "scientific_primary_energy_per_work_unit_j": 0.1,
            "scientific_primary_active_duration_s": 2.0,
            "scientific_primary_avg_power_w": 5.0,
            "scientific_primary_energy_statistics": {
                "energy_j": {"n": 3, "mean": 10.0, "ci_low": 9.8, "ci_high": 10.2},
                "energy_per_work_unit_j": {"n": 3, "mean": 0.1},
                "avg_power_w": {"n": 3, "mean": 5.0},
            },
            "scientific_shadow_method": "chapter4_legacy_window",
            "scientific_shadow_energy_status": "incomplete",
            "legacy_window_comparison_requested": True,
            "legacy_window_comparison_attempted_runs": 3,
            "legacy_window_comparison_successful_runs": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_report(root, aggregate)
            row = collect_native_energy(root)[0]

        self.assertEqual(row["scientific_primary_method"], "command_marker_window")
        self.assertEqual(row["energy_total_j"], 10.0)
        self.assertEqual(row["scientific_shadow_method"], "chapter4_legacy_window")
        self.assertEqual(row["scientific_shadow_energy_status"], "incomplete")
        self.assertEqual(row["energy_ab_status"], "no_valid_ab_repeats")
        self.assertTrue(row["claim_eligible"], row["claim_exclusion_reasons"])
        self.assertFalse(row["scientific_shadow_affects_primary_claim"])
        exported = build_native_energy_ab_aggregates([row])[0]
        self.assertEqual(exported["scientific_primary_method"], "command_marker_window")
        self.assertEqual(exported["scientific_shadow_method"], "chapter4_legacy_window")
        self.assertFalse(exported["scientific_shadow_affects_primary_claim"])

    def test_v266_chapter4_primary_is_not_reinterpreted_as_a_shadow(self) -> None:
        aggregate = {
            **_measurement_contract(),
            "scientific_primary_method_frozen": True,
            "scientific_primary_method": "chapter4_baseline",
            "scientific_primary_energy_status": "available",
            "scientific_primary_claim_eligible": True,
            "scientific_primary_energy_total_j": 12.0,
            "scientific_primary_energy_per_work_unit_j": 0.12,
            "scientific_primary_active_duration_s": 2.0,
            "scientific_primary_avg_power_w": 6.0,
            "scientific_primary_energy_statistics": {
                "energy_j": {"n": 3, "mean": 12.0},
                "energy_per_work_unit_j": {"n": 3, "mean": 0.12},
                "avg_power_w": {"n": 3, "mean": 6.0},
            },
            "candidate_method": "candidate_v263",
            "candidate_role": "shadow_only",
            "candidate_eligible_for_auto_switch": False,
            "candidate_v263_shadow_energy_total_j": 9.0,
            "candidate_v263_shadow_energy_per_work_unit_j": 0.09,
            "avg_energy_total_j": 9.0,
            "avg_energy_per_work_unit_j": 0.09,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_report(root, aggregate)
            row = collect_native_energy(root)[0]

        self.assertEqual(row["scientific_primary_method"], "chapter4_baseline")
        self.assertEqual(row["scientific_primary_contract_generation"], "v2.63-v2.66_chapter4_primary")
        self.assertEqual(row["energy_total_j"], 12.0)
        self.assertEqual(row["energy_per_work_j"], 0.12)
        self.assertEqual(row["scientific_shadow_method"], "candidate_v263")
        self.assertEqual(row["candidate_v263_shadow_energy_total_j"], 9.0)
        self.assertIsNone(row["chapter4_legacy_shadow_energy_total_j"])
        self.assertTrue(row["claim_eligible"], row["claim_exclusion_reasons"])


if __name__ == "__main__":
    unittest.main()
