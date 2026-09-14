"""Historical hardware-independent release smoke for version 2.79.12.

The literal release identity is intentionally frozen here. The current
maintenance-line smoke lives in :mod:`onnx_splitpoint_tool.v27913_smoke`.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import __build_features__
from .ranking_methods import RANKING_METHOD_IMPLEMENTATION, WORKFLOW_RANKING_METHOD
from .release_identity import (
    BYTECODE_ISOLATION_CONTRACT,
    REMOTE_ENERGY_PRIMARY_ADMISSION_CONTRACT,
    SOURCE_INTEGRITY_CONTRACT,
)
from .v27911_smoke import REQUIRED_FEATURES as V27911_REQUIRED_FEATURES

VERSION = "2.79.12"
LINEAGE = "v2.79"
BUILD_ID = "v2.79.12-platform-power-calibration-provenance-closure"
NEW_FEATURES = {
    BYTECODE_ISOLATION_CONTRACT,
    SOURCE_INTEGRITY_CONTRACT,
    REMOTE_ENERGY_PRIMARY_ADMISSION_CONTRACT,
    "m2_dot_udp_command_default_and_legacy_migration",
    "full_system_energy_method_preflight_before_power_mutation",
    "m2_idle_energy_calibration_gate_binding",
    "m2_idle_off_on_evidence_sha256_binding",
    "platform_power_energy_gate_reason_projection",
    "installed_acceptance_preserved_profile_scope",
    "energy_method_runtime_binary_content_binding",
    "hardware_registry_stale_writer_compare_and_swap",
    "m2_idle_calibration_config_and_commit_compare_and_swap",
    "generated_idle_binding_v2_self_verification",
    "physical_state_transition_binding",
    "current_energy_method_exact_pairing",
    "exact_method_eight_artifact_admission",
    "pre_mutation_udp_configuration_and_ssh_target_validation",
    "standalone_platform_registry_race_guard",
    "duplicate_hardware_setup_id_rejection",
    "accelerator_and_effective_jetson_endpoint_binding",
    "energy_data_port_runtime_binding",
    "energy_claim_duplicate_setup_rejection",
    "malformed_current_platform_identity_rejection",
    "accelerator_env_hidden_target_field_preservation",
    "calibration_effective_ssh_target_cas",
    "raw_energy_registry_claim_admission",
}
REQUIRED_FEATURES = set(V27911_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    checks = {
        "historical_identity": (
            VERSION == "2.79.12"
            and LINEAGE == "v2.79"
            and BUILD_ID
            == "v2.79.12-platform-power-calibration-provenance-closure"
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "ranking_freeze": (
            WORKFLOW_RANKING_METHOD == "cut_bytes_only"
            and RANKING_METHOD_IMPLEMENTATION
            == "v277-cut-bytes-only-workflow-freeze-1"
        ),
        "historical_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/v27912_smoke.py",
                "scripts/run_v27912_small_acceptance.sh",
                "scripts/run_v27912_seven_model_long_overnight.sh",
                "scripts/verify_v27912_yolov7_claim_gate_32.py",
                "scripts/launcher_status_v27912.py",
                "profiles/complete_set_7models_v27912_b500_audit20.yaml",
                "tests/test_v27912_release_provenance.py",
                "tests/test_v27912_seven_model_overnight_launcher.py",
                "tests/test_v27912_yolov7_claim_gate_32.py",
            )
        ),
        "historical_entrypoints": all(
            marker in pyproject
            for marker in (
                'onnx-splitpoint-smoke-v27912 = "onnx_splitpoint_tool.v27912_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-12 = "onnx_splitpoint_tool.v27912_smoke:main"',
            )
        )
        and all(
            marker in updater
            for marker in (
                "onnx-splitpoint-smoke-v27912=onnx_splitpoint_tool.v27912_smoke:main",
                "onnx-splitpoint-smoke-v2-79-12=onnx_splitpoint_tool.v27912_smoke:main",
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(
            json.dumps(
                {"status": "FAIL", "failed": failed, "checks": checks},
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {"status": "PASS", "version": VERSION, "checks": checks},
            indent=2,
        )
    )
    print("PASS v2.79.12 historical smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
