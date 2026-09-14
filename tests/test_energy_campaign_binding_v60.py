from __future__ import annotations

import unittest

from onnx_splitpoint_tool.workflow.execution_binding import (
    _energy_confidence_level_for_profile,
    _energy_physical_scope_for_profile,
    _energy_randomization_seed_for_profile,
    _energy_randomize_target_order_for_profile,
    _energy_repeat_override,
    _energy_window_label_for_profile,
    _profile_energy,
)


class _Options:
    energy_repeat_override = 0
    energy_confidence_level = None
    energy_physical_scope = ""
    energy_window_label = ""
    energy_randomize_target_order = None
    energy_randomization_seed = None


class EnergyCampaignBindingV60Tests(unittest.TestCase):
    def test_final_system_power_contract_reaches_remote_energy_options(self) -> None:
        profile = {
            "campaign": {"mode": "final", "enforcement": "strict"},
            "measurement_campaign": {
                "system_power": {
                    "scope": "FS",
                    "window": "command",
                    "repeats": 5,
                    "confidence_level": 0.95,
                    "randomize_run_order": True,
                    "randomization_seed": 1234,
                }
            },
        }
        cfg = _profile_energy(profile)
        self.assertTrue(cfg["enabled"])
        self.assertEqual(_energy_repeat_override(_Options(), profile), 5)
        self.assertAlmostEqual(_energy_confidence_level_for_profile(_Options(), profile), 0.95)
        self.assertEqual(_energy_physical_scope_for_profile(_Options(), profile), "FS")
        self.assertEqual(_energy_window_label_for_profile(_Options(), profile), "command")
        self.assertTrue(_energy_randomize_target_order_for_profile(_Options(), profile))
        self.assertEqual(_energy_randomization_seed_for_profile(_Options(), profile), 1234)


if __name__ == "__main__":
    unittest.main()
