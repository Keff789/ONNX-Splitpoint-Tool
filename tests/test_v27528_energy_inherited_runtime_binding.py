from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import onnx_splitpoint_tool.energy.collector as energy_collector
from onnx_splitpoint_tool.campaign import create_energy_calibration_manifest
from onnx_splitpoint_tool.energy.collector import (
    _verify_calibration_manifest,
    run_fast_firmware_measurement,
)
from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup
from onnx_splitpoint_tool.energy.method_manifest import (
    EXACT_IMPLEMENTATION_REUSE_POLICY,
    _canonical_implementation_artifacts,
)
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool.source_integrity import create_source_integrity_binding
from onnx_splitpoint_tool.workflow.artifacts import sha256_json


SETUP_BINDINGS = {
    "orin_nx_hailo8_01": "192.168.0.197",
    "orin_nx_hailo10_01": "192.168.0.176",
    "orin_nx_deepx_m1_01": "192.168.0.185",
}
SETUP_ID = "orin_nx_hailo8_01"
URECS_ADDRESS = SETUP_BINDINGS[SETUP_ID]


def _source_integrity_report() -> dict[str, object]:
    return {
        "ok": True,
        "status": "verified",
        "package_version": VERSION,
        "build_id": BUILD_ID,
        "manifest_path": "/opt/onnx-splitpoint-tool/SOURCE_MANIFEST.json",
        "manifest_sha256": "a" * 64,
        "sha256sums_path": "/opt/onnx-splitpoint-tool/SHA256SUMS.txt",
        "sha256sums_sha256": "b" * 64,
    }


def _inherited_manifest(tmp_path: Path) -> tuple[Path, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    collector_binary = bin_dir / "urecs-data-collector"
    postprocessor_binary = bin_dir / "power_calculations"
    for executable in (collector_binary, postprocessor_binary):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    implementation_artifacts = _canonical_implementation_artifacts(
        EnergyDefaults(
            collector_binary=str(collector_binary),
            power_calculations_binary=str(postprocessor_binary),
        )
    )
    spec = {
        "evidence_mode": "inherited_validated_method",
        "channel_id": "urecs_fs_input_channel_0",
        "scope": "FS",
        "measurement_point": "complete_system_input",
        "sample_rate_hz": 2000,
        "locked": True,
        "method": {
            "collector": "urecs-data-collector",
            "collector_mode": "fast_firmware",
            "postprocessor": "power_calculations",
            "sample_rate_hz": 2000,
            "data_port": 3000,
            "output_semantics": "calibrated_input_energy_unsubtracted",
            "implementation_policy": EXACT_IMPLEMENTATION_REUSE_POLICY,
        },
        "validation_reference": {
            "reference_id": "wachsmuth2026masterarbeit",
            "author": "Wachsmuth, Joris",
            "title": (
                "Entwicklung und Validierung eines Messsystems zur "
                "energetischen Bewertung eingebetteter KI-Beschleuniger"
            ),
            "institution": "Bielefeld University",
            "work_type": "Master's thesis",
            "internal_identifier": "M99",
            "year": 2026,
        },
        "reuse_attestation": {
            "validated_method_accepted": True,
            "exact_implementation_reused": True,
            "new_calibration_required": False,
            "attested_by": "Kevin Mika",
            "attested_at": "2026-08-10T00:00:00+02:00",
        },
        "source_release_integrity": create_source_integrity_binding(
            _source_integrity_report()
        ),
        "channel_bindings": [
            {
                "setup_id": setup_id,
                "urecs_address": urecs_address,
                "data_port": 3000,
                "channel": 0,
                "sample_rate_hz": 2000,
                "scope": "FS",
                "measurement_point": "complete_system_input",
            }
            for setup_id, urecs_address in SETUP_BINDINGS.items()
        ],
        "artifacts": [
            {
                "id": artifact_id,
                "kind": "measurement_implementation",
                "path": str(artifact_path),
            }
            for artifact_id, artifact_path in implementation_artifacts
        ],
    }
    spec_path = tmp_path / "energy_method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    manifest = create_energy_calibration_manifest(
        spec=spec_path,
        output=tmp_path / "urecs_fs_input_provenance_manifest.json",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return manifest, digest


def _verify_inherited(
    manifest: Path,
    digest: str,
    **overrides: object,
) -> dict[str, object]:
    runtime: dict[str, object] = {
        "setup_id": SETUP_ID,
        "urecs_address": URECS_ADDRESS,
        "data_port": 3000,
        "channel": 0,
        "sample_rate_hz": 2000,
        "physical_scope": "FS",
        "collector_mode": "fast_firmware",
    }
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = {
        row.get("id"): row
        for row in payload.get("artifacts", [])
        if isinstance(row, dict)
    }
    runtime["collector_binary"] = artifacts.get(
        "urecs_data_collector_binary", {}
    ).get("path", "missing-collector")
    runtime["postprocessor_binary"] = artifacts.get(
        "power_calculations_binary", {}
    ).get("path", "missing-postprocessor")
    runtime["expected_channel_bindings"] = (
        payload.get("channel_bindings")
        or payload.get("bindings")
        or payload.get("target_bindings")
        or []
    )
    runtime.update(overrides)
    return _verify_calibration_manifest(manifest, digest, **runtime)


class EnergyInheritedRuntimeBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch(
            "onnx_splitpoint_tool.energy.method_manifest.verify_installed_source_integrity",
            side_effect=lambda: dict(_source_integrity_report()),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_inherited_method_manifest_matches_exact_runtime_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, digest = _inherited_manifest(Path(temp))
            for setup_id, urecs_address in SETUP_BINDINGS.items():
                with self.subTest(setup_id=setup_id):
                    verification = _verify_inherited(
                        manifest,
                        digest,
                        setup_id=setup_id,
                        urecs_address=urecs_address,
                    )
                    self.assertIs(verification["verified"], True)
                    self.assertEqual(
                        verification["status"],
                        "inherited_validated_method_verified",
                    )
                    self.assertIs(
                        verification["inherited_content_verified"], True
                    )
                    self.assertIs(
                        verification["inherited_binding_verified"], True
                    )
                    self.assertEqual(
                        verification["runtime_binding_id"], setup_id
                    )
                    self.assertEqual(
                        verification["runtime_binding_errors"], []
                    )

    def test_historical_bindings_alias_is_not_admitted_for_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, _digest = _inherited_manifest(Path(temp))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["bindings"] = payload.pop("channel_bindings")
            payload["manifest_payload_sha256"] = sha256_json(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "manifest_payload_sha256"
                }
            )
            manifest.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

            verification = _verify_inherited(manifest, digest)
            self.assertIs(verification["verified"], False)
            self.assertEqual(
                verification["status"], "inherited_method_manifest_invalid"
            )
            self.assertEqual(
                verification["runtime_binding_errors"],
                ["runtime_channel_binding_rows_not_exact"],
            )

    def test_resigned_underbound_inherited_method_is_not_admitted(self) -> None:
        mutations = ("missing_policy", "single_artifact", "nonmapping_binding")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temp:
                    manifest, _digest = _inherited_manifest(Path(temp))
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    if mutation == "missing_policy":
                        payload["method"].pop("implementation_policy")
                    elif mutation == "single_artifact":
                        payload["artifacts"] = payload["artifacts"][:1]
                        payload["artifact_set_sha256"] = sha256_json(
                            payload["artifacts"]
                        )
                    elif mutation == "nonmapping_binding":
                        payload["channel_bindings"].append("hidden-extra-row")
                        payload["channel_binding_set_sha256"] = sha256_json(
                            payload["channel_bindings"]
                        )
                    payload["manifest_payload_sha256"] = sha256_json(
                        {
                            key: value
                            for key, value in payload.items()
                            if key != "manifest_payload_sha256"
                        }
                    )
                    manifest.write_text(
                        json.dumps(payload, indent=2), encoding="utf-8"
                    )
                    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

                    verification = _verify_inherited(manifest, digest)
                    self.assertIs(verification["verified"], False)
                    self.assertEqual(
                        verification["status"],
                        "inherited_method_manifest_invalid",
                    )

    def test_inherited_method_runtime_binding_mismatch_fails_closed(
        self,
    ) -> None:
        cases = [
            (
                "setup_id",
                "orin_nx_unknown",
                "runtime_setup_binding_missing",
            ),
            (
                "urecs_address",
                "192.168.0.176",
                "runtime_urecs_address_binding_mismatch",
            ),
            ("channel", 1, "runtime_channel_not_zero"),
            (
                "sample_rate_hz",
                1000,
                "runtime_sample_rate_not_2000_hz",
            ),
            (
                "physical_scope",
                "MB",
                "runtime_physical_scope_not_full_system",
            ),
            (
                "collector_mode",
                "legacy_firmware",
                "runtime_collector_mode_not_fast_firmware",
            ),
            (
                "collector_binary",
                "different-collector",
                "runtime_collector_binary_mismatch",
            ),
            (
                "postprocessor_binary",
                "different-postprocessor",
                "runtime_postprocessor_binary_mismatch",
            ),
        ]
        for override, value, reason in cases:
            with self.subTest(override=override, value=value):
                with tempfile.TemporaryDirectory() as temp:
                    manifest, digest = _inherited_manifest(Path(temp))
                    verification = _verify_inherited(
                        manifest,
                        digest,
                        **{override: value},
                    )
                    self.assertIs(verification["verified"], False)
                    self.assertIs(
                        verification["runtime_fail_closed_required"],
                        True,
                    )
                    self.assertEqual(
                        verification["status"],
                        "inherited_method_runtime_binding_mismatch",
                    )
                    self.assertIn(
                        reason, verification["runtime_binding_errors"]
                    )

    def test_inherited_method_semantic_manifest_mutation_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, _digest = _inherited_manifest(Path(temp))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["reuse_attestation"][
                "exact_implementation_reused"
            ] = False
            payload["manifest_payload_sha256"] = sha256_json(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "manifest_payload_sha256"
                }
            )
            manifest.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

            verification = _verify_inherited(manifest, digest)
            self.assertIs(verification["verified"], False)
            self.assertIs(
                verification["runtime_fail_closed_required"], True
            )
            self.assertEqual(
                verification["status"],
                "inherited_method_manifest_invalid",
            )
            self.assertIs(
                verification["inherited_content_verification"]
                ["inherited_reuse_attestation_ok"],
                False,
            )

    def test_binding_failure_stops_before_tools_collector_and_workload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            manifest, digest = _inherited_manifest(tmp_path)

            def tools_must_not_run(
                _defaults: EnergyDefaults,
            ) -> dict[str, object]:
                raise AssertionError(
                    "tool probe ran before inherited binding gate"
                )

            out_dir = tmp_path / "blocked_measurement"
            with mock.patch.object(
                energy_collector,
                "check_energy_tools",
                tools_must_not_run,
            ):
                result = run_fast_firmware_measurement(
                    "touch should_never_run",
                    out_dir,
                    setup=EnergySetup(
                        setup_id=SETUP_ID,
                        enabled=True,
                        urecs_address="192.168.0.176",
                    ),
                    defaults=EnergyDefaults(
                        collector_binary="urecs-data-collector",
                        power_calculations_binary="power_calculations",
                        mode="fast_firmware",
                        channel=0,
                        sample_rate=2000,
                    ),
                    setup_id=SETUP_ID,
                    duration_s=1.0,
                    physical_scope="FS",
                    calibration_manifest=str(manifest),
                    calibration_sha256=digest,
                )

            self.assertIs(result["ok"], False)
            self.assertIn(
                result["status"],
                {
                    "energy_input_registry_contract_blocked",
                    "energy_input_provenance_runtime_binding_blocked",
                },
            )
            self.assertIs(result["collector_started"], False)
            self.assertIs(result["workload_started"], False)
            self.assertFalse((out_dir / "probe").exists())
            self.assertEqual(list(out_dir.glob("run_*")), [])
            persisted = json.loads(
                (out_dir / "energy_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(persisted["status"], result["status"])

    def test_legacy_direct_calibration_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "legacy_direct_calibration.json"
            manifest.write_text(
                json.dumps({"evidence_mode": "direct_calibration"}),
                encoding="utf-8",
            )
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            verification = _verify_calibration_manifest(manifest, digest)
            self.assertIs(verification["verified"], True)
            self.assertEqual(verification["status"], "verified")
            self.assertIs(
                verification["runtime_fail_closed_required"], False
            )


if __name__ == "__main__":
    unittest.main()
