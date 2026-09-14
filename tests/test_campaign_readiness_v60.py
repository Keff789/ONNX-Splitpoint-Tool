from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml

from onnx_splitpoint_tool.campaign import (
    build_campaign_readiness,
    create_candidate_universe_manifest,
    create_dataset_manifest,
    create_energy_calibration_manifest,
    create_holdout_registry,
    create_pipeline_contract_manifest,
    create_prediction_freeze_approval,
    create_campaign_freeze,
    deterministic_audit_cases,
    validate_calibration_validation_separation,
    verify_dataset_manifest,
    verify_energy_calibration_manifest,
    verify_holdout_registry,
    verify_pipeline_contract_manifest,
    verify_prediction_freeze_approval,
    verify_campaign_freeze,
)
from onnx_splitpoint_tool.energy.collector import _repeat_statistics
from onnx_splitpoint_tool.workflow.artifacts import sha256_file, sha256_json, write_csv, write_json


class CampaignReadinessV60Tests(unittest.TestCase):
    def _image(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def _dataset_manifests(self, root: Path) -> dict[str, Path]:
        cls_cal = root / "datasets" / "cls_cal"
        cls_val = root / "datasets" / "cls_val"
        self._image(cls_cal / "cat" / "a.jpg", b"classification-calibration-a")
        self._image(cls_cal / "dog" / "b.jpg", b"classification-calibration-b")
        self._image(cls_val / "cat" / "c.jpg", b"classification-validation-c")
        self._image(cls_val / "dog" / "d.jpg", b"classification-validation-d")

        det_cal = root / "datasets" / "det_cal"
        det_val = root / "datasets" / "det_val"
        self._image(det_cal / "a.jpg", b"detection-calibration-a")
        self._image(det_val / "b.jpg", b"detection-validation-b")
        ann_cal = root / "datasets" / "det_cal.json"
        ann_val = root / "datasets" / "det_val.json"
        ann_cal.write_text(json.dumps({"images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}], "annotations": [], "categories": []}), encoding="utf-8")
        ann_val.write_text(json.dumps({"images": [{"id": 2, "file_name": "b.jpg", "width": 10, "height": 10}], "annotations": [], "categories": []}), encoding="utf-8")

        out = root / "campaign_inputs"
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "classification_calibration": create_dataset_manifest(task="classification", role="calibration", dataset_id="cls-cal", split="train", root=cls_cal, output=out / "cls_cal.json", selection_strategy="class_stratified", max_items=2),
            "classification_validation": create_dataset_manifest(task="classification", role="validation", dataset_id="cls-val", split="val", root=cls_val, output=out / "cls_val.json"),
            "detection_calibration": create_dataset_manifest(task="detection", role="calibration", dataset_id="det-cal", split="train2017", root=det_cal, annotations=ann_cal, output=out / "det_cal.json"),
            "detection_validation": create_dataset_manifest(task="detection", role="validation", dataset_id="det-val", split="val2017", root=det_val, annotations=ann_val, output=out / "det_val.json"),
        }
        return paths

    def _pipeline_contract(self, root: Path) -> Path:
        cfg = root / "configs"
        cfg.mkdir(parents=True, exist_ok=True)
        adapter_source = root / "adapters" / "endpoint_adapter.py"
        adapter_source.parent.mkdir(parents=True, exist_ok=True)
        adapter_source.write_text(
            "def adapt(endpoint_contract, tensors):\n"
            "    return tensors\n",
            encoding="utf-8",
        )
        entries = []
        for name, kind in (
            ("classification_preprocessing", "preprocessing"),
            ("detection_preprocessing", "preprocessing"),
            ("detection_decoder", "decoder"),
            ("detection_nms", "nms"),
            ("endpoint_adapters", "adapter"),
        ):
            path = cfg / f"{name}.yaml"
            path.write_text(yaml.safe_dump({"locked": True, "contract_id": name, "implementation": "test-v1"}, sort_keys=False), encoding="utf-8")
            entry = {"id": name, "kind": kind, "task": "classification" if name.startswith("classification") else ("all" if kind == "adapter" else "detection"), "path": str(path)}
            if kind == "adapter":
                entry["implementation_sources"] = [str(adapter_source)]
            entries.append(entry)
        spec = root / "pipeline_contract_sources.yaml"
        spec.write_text(yaml.safe_dump({"contracts": entries}, sort_keys=False), encoding="utf-8")
        return create_pipeline_contract_manifest(spec=spec, output=root / "campaign_inputs" / "pipeline_contract_manifest.json")

    def _energy_calibration(self, root: Path) -> Path:
        data = root / "energy_data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "cal.csv").write_text("raw,ampere\n0,0\n100,1\n", encoding="utf-8")
        (data / "verify.csv").write_text("raw,ampere\n50,0.5\n", encoding="utf-8")
        spec = {
            "channel_id": "urecs_fs_input",
            "scope": "FS",
            "measurement_point": "u.RECS input",
            "sample_rate_hz": 2000,
            "locked": True,
            "sensor_chain": {"shunt_ohm": 0.02, "amplifier": "INA225", "adc": "ADS7953", "adc_channel": 0, "firmware_version": "test"},
            "calibration": {"model": "affine", "coefficients": {"offset_raw": 0.0, "gain_si_per_raw": 0.01}, "fitted_at": "2026-07-10T00:00:00Z", "fitted_by": "tester", "calibration_point_count": 2},
            "verification": {"independent": True, "pass": True, "sample_count": 1, "verified_at": "2026-07-10T00:00:00Z", "verified_by": "tester", "mean_absolute_relative_error_percent": 0.1, "maximum_absolute_relative_error_percent": 0.2},
            "uncertainty": {"confidence_level": 0.95, "combined_relative_percent": 0.5, "expanded_relative_percent": 1.0},
            "artifacts": [
                {"id": "cal", "kind": "calibration_data", "path": str(data / "cal.csv")},
                {"id": "verify", "kind": "verification_data", "path": str(data / "verify.csv")},
            ],
        }
        spec_path = root / "energy_spec.yaml"
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        return create_energy_calibration_manifest(spec=spec_path, output=root / "campaign_inputs" / "energy_calibration.json")

    def test_dataset_contract_and_energy_manifests_are_hash_verified_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = self._dataset_manifests(root)
            loaded = [json.loads(path.read_text(encoding="utf-8")) for path in manifests.values()]
            self.assertTrue(all(verify_dataset_manifest(item, verify_files=True)["ok"] for item in loaded))
            self.assertTrue(validate_calibration_validation_separation(loaded)["ok"])

            contract_path = self._pipeline_contract(root)
            contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
            verification = verify_pipeline_contract_manifest(
                contract_payload, require_locked=True,
            )
            self.assertTrue(verification["ok"])
            self.assertTrue(verification["adapter_source_coverage_ok"])
            adapter_source = Path(
                next(
                    row for row in contract_payload["contracts"]
                    if row["kind"] == "adapter"
                )["implementation_sources"][0]["path"]
            )
            adapter_source.write_text("# outcome-informed mutation\n", encoding="utf-8")
            self.assertFalse(
                verify_pipeline_contract_manifest(
                    contract_payload, require_locked=True,
                )["ok"]
            )

            energy_path = self._energy_calibration(root)
            self.assertTrue(verify_energy_calibration_manifest(json.loads(energy_path.read_text(encoding="utf-8")), require_final=True)["ok"])

            # A byte-level mutation invalidates the manifest immediately.
            cls_val_manifest = json.loads(manifests["classification_validation"].read_text(encoding="utf-8"))
            sample = Path(cls_val_manifest["root"]) / cls_val_manifest["items"][0]["relative_path"]
            sample.write_bytes(b"mutated")
            self.assertFalse(verify_dataset_manifest(cls_val_manifest, verify_files=True)["ok"])

    def test_deterministic_audit_is_stable_and_score_independent(self) -> None:
        candidates = [
            {"case_id": f"b{i:03d}", "boundary": i, "cut_bytes": i * 1000, "imbalance_val": (i % 7) / 7.0, "n_cut_tensors": 1 + (i % 3), "strict_ok": True, "prediction_score": 1000 - i}
            for i in range(1, 41)
        ]
        first = deterministic_audit_cases(candidates, model_id="holdout", size=12, seed=17)
        for row in candidates:
            row["prediction_score"] = -float(row["prediction_score"])
        second = deterministic_audit_cases(candidates, model_id="holdout", size=12, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)

    def test_prediction_approval_remains_verifiable_after_portable_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            analysis = run_dir / "models" / "holdout" / "analysis"
            analysis.mkdir(parents=True)
            candidates = [
                {"case_id": "b001", "boundary": 1, "cut_bytes": 1000, "strict_ok": True},
                {"case_id": "b002", "boundary": 2, "cut_bytes": 2000, "strict_ok": True},
            ]
            prediction = write_json(analysis / "prediction.json", {"model_id": "holdout", "candidates": candidates})
            pred_csv = write_csv(analysis / "predictions_frozen.csv", candidates)
            ranking_csv = write_csv(analysis / "ranking_predictions_frozen.csv", [{"case_id": "b001", "method_id": "cut_bytes_only", "predicted_rank": 1}])
            universe_path, _, universe = create_candidate_universe_manifest(model_id="holdout", candidates=candidates, mode="all_feasible", output_dir=analysis, source_prediction_sha256=sha256_file(prediction) or "")
            manifest = {
                "schema": "onnx-splitpoint/prediction-freeze-manifest",
                "schema_version": 1,
                "model_id": "holdout",
                "evaluation_role": "holdout",
                "prospective": True,
                "valid_for_holdout": True,
                "freeze_status": "prospective_frozen",
                "prediction_json": prediction.name,
                "prediction_sha256": sha256_file(prediction),
                "prediction_csv": pred_csv.name,
                "prediction_csv_sha256": sha256_file(pred_csv),
                "ranking_prediction_csv": ranking_csv.name,
                "ranking_prediction_csv_sha256": sha256_file(ranking_csv),
                "candidate_universe_manifest": universe_path.name,
                "candidate_universe_sha256": universe["universe_sha256"],
            }
            write_json(analysis / "prediction_freeze_manifest.json", manifest)
            approval = create_prediction_freeze_approval(run_dir=run_dir, model_id="holdout", signer="Test Reviewer")
            self.assertTrue(verify_prediction_freeze_approval(approval)["ok"])

            portable = root / "portable"
            shutil.copytree(analysis, portable)
            shutil.rmtree(run_dir)
            self.assertTrue(verify_prediction_freeze_approval(portable / "prediction_freeze_approval.json")["ok"])

    def test_complete_strict_profile_can_reach_ready_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = self._dataset_manifests(root)
            contract = self._pipeline_contract(root)
            energy = self._energy_calibration(root)
            model_dev = root / "dev.onnx"
            model_holdout = root / "holdout.onnx"
            model_dev.write_bytes(b"dev-model")
            model_holdout.write_bytes(b"holdout-model")

            profile = {
                "name": "synthetic-final",
                "campaign": {
                    "id": "synthetic-final",
                    "mode": "final",
                    "enforcement": "strict",
                    "frozen_before_final_campaign": True,
                    "dataset_manifests": {
                        "classification": {"calibration": str(manifests["classification_calibration"]), "validation": str(manifests["classification_validation"])},
                        "detection": {"calibration": str(manifests["detection_calibration"]), "validation": str(manifests["detection_validation"])},
                    },
                    "pipeline_contract_manifest": str(contract),
                    "holdout_registry": str(root / "campaign_inputs" / "holdout_registry.json"),
                    "ranking_model_bundle": str(root / "campaign_inputs" / "ranking_model_bundle.json"),
                    "energy_calibration_manifest": str(energy),
                    "require_fitted_stage_time": True,
                    "require_native_handover_model": True,
                },
                "quality_gate": {"frozen_before_final_campaign": True},
                "model_suite": {
                    "primary": [
                        {"id": "dev", "task": "classification", "family_id": "resnet", "evaluation_role": "development", "generalization_scope": "development", "validation_tier": "final", "path": str(model_dev), "candidate_universe": {"mode": "declared_shortlist"}},
                        {"id": "holdout", "task": "detection", "family_id": "unseen_detector", "evaluation_role": "holdout", "generalization_scope": "model_family_holdout", "validation_tier": "final", "path": str(model_holdout), "holdout_group": "det", "candidate_universe": {"mode": "all_feasible", "minimum_valid_candidates": 10}, "unseen_attestation": {"unseen": True, "attested_by": "Reviewer", "attested_at": "2026-07-10T00:00:00Z"}},
                    ],
                    "reserve": [],
                },
                "ranking_validation": {
                    "cycle_time_no_handover": {},
                    "cycle_time_with_handover": {},
                },
                "native_producers": {"enabled": True, "full_baselines": {"enabled": True, "same_runtime_contract": True}},
                "measurement_campaign": {
                    "system_power": {
                        "scope": "FS",
                        "window": "command",
                        "raw_primary": True,
                        "repeats": 5,
                        "confidence_level": 0.95,
                        "randomize_run_order": True,
                        "randomization_seed": 9,
                        "idle_normalization": {"enabled": True, "method": "paired_fs_rail_on_off", "report_raw_and_normalized": True},
                    }
                },
            }
            profile_path = root / "profile.yaml"
            profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
            create_holdout_registry(profile=profile_path, output=root / "campaign_inputs" / "holdout_registry.json")
            bundle = {
                "schema": "onnx-splitpoint/ranking-model-bundle",
                "schema_version": 1,
                "fit_policy": {
                    "holdout_rows_excluded": True,
                    "explicit_development_role_required": True,
                    "unlabeled_rows_treated_as_development": False,
                },
                "development_model_ids": ["dev"],
                "stage_time_models": {"tensorrt": {"status": "fitted", "mode": "affine_flops", "intercept_ms": 0.1, "per_gflop_ms": 0.2}},
                "handover_models": {"native_fifo": {"hailo8_to_tensorrt": {"status": "fitted", "mode": "constant", "value_ms": 0.1}}},
                "fitted_stage_model_count": 1,
                "fitted_handover_model_count": 1,
                "profile_patch": {"ranking_validation": {}},
            }
            bundle["bundle_payload_sha256"] = sha256_json({k: v for k, v in bundle.items() if k != "bundle_payload_sha256"})
            write_json(root / "campaign_inputs" / "ranking_model_bundle.json", bundle)

            loaded = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            readiness = build_campaign_readiness(loaded, profile_path=profile_path)
            self.assertTrue(readiness["ready"], [row for row in readiness["checks"] if row["status"] == "fail"])

            # The same complete core evidence must also reach Final readiness
            # under the narrower explicit matrix claim without consuming the
            # hold-out registry or ranking bundle.
            loaded["campaign"]["claim_scope"] = "evaluated_matrix"
            loaded["campaign"]["holdout_registry"] = ""
            loaded["campaign"]["ranking_model_bundle"] = ""
            loaded["ranking_validation"]["enabled"] = False
            matrix = build_campaign_readiness(loaded, profile_path=profile_path)
            self.assertTrue(matrix["ready"], [row for row in matrix["checks"] if row["status"] == "fail"])
            matrix_checks = {row["id"]: row for row in matrix["checks"]}
            self.assertEqual(matrix_checks["holdout_registry"]["status"], "not_applicable")
            self.assertEqual(matrix_checks["ranking_model_bundle_integrity"]["status"], "not_applicable")

    def test_holdout_registry_uses_narrow_role_and_model_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dev = root / "dev.onnx"
            holdout = root / "holdout.onnx"
            dev.write_bytes(b"dev")
            holdout.write_bytes(b"holdout")
            profile = {
                "name": "registry-test",
                "model_suite": {
                    "primary": [
                        {"id": "dev", "evaluation_role": "development", "path": str(dev)},
                        {
                            "id": "holdout",
                            "evaluation_role": "holdout",
                            "path": str(holdout),
                            "holdout_group": "classifier",
                            "candidate_universe": {"mode": "all_feasible"},
                            "unseen_attestation": {"unseen": True, "attested_by": "Reviewer", "attested_at": "2026-07-10T00:00:00Z"},
                        },
                    ],
                    "reserve": [],
                },
                "ranking_validation": {"weighted_score": {"w_comm": 1.0}},
            }
            profile_path = root / "profile.yaml"
            profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
            registry_path = create_holdout_registry(profile=profile_path, output=root / "holdout_registry.json")
            registry = json.loads(registry_path.read_text(encoding="utf-8"))

            # Fitting/patching ranking coefficients must not invalidate the
            # already frozen model-role decision.
            profile["ranking_validation"]["weighted_score"]["w_comm"] = 2.0
            patched = root / "profile_fitted.yaml"
            patched.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
            verification = verify_holdout_registry(registry, profile=profile, profile_path=patched, require_attestation=True)
            self.assertTrue(verification["ok"], verification)
            self.assertTrue(verification["model_role_projection_hash_ok"])
            self.assertFalse(verification["source_profile_hash_ok"])

            # The exact model artefact remains bound and tamper-evident.
            holdout.write_bytes(b"changed-holdout")
            verification = verify_holdout_registry(registry, profile=profile, profile_path=patched, require_attestation=True)
            self.assertFalse(verification["ok"])
            self.assertTrue(any(row.get("field") == "model_sha256" for row in verification["mismatches"]))

    def test_campaign_freeze_verifier_detects_archive_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile_path = root / "profile.yaml"
            profile_path.write_text(yaml.safe_dump({"name": "dev-freeze", "campaign": {"mode": "development"}}, sort_keys=False), encoding="utf-8")
            archive = create_campaign_freeze(profile=profile_path, output=root / "freeze.zip", signer="")
            self.assertTrue(verify_campaign_freeze(archive)["ok"])

            tampered = root / "freeze_tampered.zip"
            with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as target:
                for info in source.infolist():
                    data = source.read(info.filename)
                    if info.filename == "profile.yaml":
                        data += b"\n# tampered\n"
                    target.writestr(info, data)
            verification = verify_campaign_freeze(tampered)
            self.assertFalse(verification["ok"])
            self.assertFalse(verification["profile_hash_ok"])

    @unittest.skipUnless(shutil.which("openssl"), "OpenSSL not available")
    def test_campaign_freeze_optional_signature_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private_key = root / "private.pem"
            public_key = root / "public.pem"
            subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(private_key)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["openssl", "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            profile_path = root / "profile.yaml"
            profile_path.write_text(yaml.safe_dump({"name": "signed-dev-freeze", "campaign": {"mode": "development"}}, sort_keys=False), encoding="utf-8")
            archive = create_campaign_freeze(
                profile=profile_path,
                output=root / "signed_freeze.zip",
                signer="Test Signer",
                private_key=private_key,
                public_key=public_key,
            )
            verification = verify_campaign_freeze(archive)
            self.assertTrue(verification["ok"], verification)
            self.assertEqual(verification["signature_status"], "valid")

    def test_repeat_statistics_emit_small_sample_confidence_interval(self) -> None:
        stats = _repeat_statistics([10.0, 11.0, 9.0, 10.5, 9.5], 0.95)
        self.assertEqual(stats["n"], 5)
        self.assertAlmostEqual(stats["mean"], 10.0)
        self.assertLess(stats["ci_low"], stats["mean"])
        self.assertGreater(stats["ci_high"], stats["mean"])
        self.assertGreater(stats["sample_stddev"], 0.0)


if __name__ == "__main__":
    unittest.main()
