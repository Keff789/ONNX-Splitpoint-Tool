from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, SequenceNode

# The project declares jsonschema as a normal runtime dependency.  Keep this
# source-only regression test runnable in minimal packaging environments where
# dependencies have deliberately not been installed.
if importlib.util.find_spec("jsonschema") is None:
    jsonschema_stub = types.ModuleType("jsonschema")

    class _Draft202012ValidatorStub:
        def __init__(self, *_args, **_kwargs):
            pass

    jsonschema_stub.Draft202012Validator = _Draft202012ValidatorStub
    sys.modules["jsonschema"] = jsonschema_stub

from onnx_splitpoint_tool.benchmark.evaluation_profiles import profile_model_entries
from onnx_splitpoint_tool.campaign import (
    build_campaign_readiness,
    create_prediction_freeze_approval,
    stable_candidate_identity,
)
from onnx_splitpoint_tool.workflow.artifacts import read_json, write_json
from onnx_splitpoint_tool.workflow.analysis_pack import create_analysis_pack
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.scientific_reporting import _load_prediction_freeze
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _annotate_candidate_execution_contract_v261e,
)


def _checks(report: dict) -> dict[str, dict]:
    return {str(row.get("id")): row for row in list(report.get("checks") or [])}


def _candidate(index: int) -> dict:
    return {
        "case_id": f"b{index:03d}",
        "boundary": index,
        "split_index": index,
        "rank": index,
        "source_rank": index,
        "cut_bytes": index * 4096,
        "n_cut_tensors": 1 + (index % 3),
        "imbalance_val": (index % 11) / 11.0,
        "flops_left_abs": float(index),
        "flops_right_abs": float(40 - index),
        "strict_ok": True,
        "source": "onnx_real_boundary_hardware_aware",
        "predicted_total_latency_ms": 1.0 + index / 10.0,
        "score_pred": float(index),
    }


class HoldoutAuditFreezeContractV61eTests(unittest.TestCase):
    def test_analysis_pack_includes_pre_execution_holdout_and_approval_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "EvaluationRun_paused"
            scientific = run / "reports" / "scientific"
            scientific.mkdir(parents=True)
            write_json(scientific / "scientific_report.json", {"schema": "test-report"})
            (scientific / "row_eligibility.csv").write_text("model_id,ranking_eligible\n", encoding="utf-8")
            profile = {
                "profile_id": "holdout_pre_execution",
                "model_suite": {
                    "primary": [
                        {"id": "holdout_paused", "enabled": True},
                        {"id": "profile_only", "enabled": True},
                    ],
                    "reserve": [{"id": "disabled_reserve", "enabled": False}],
                }
            }
            (run / "profile.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
            write_json(
                run / "run_manifest.json",
                {
                    "schema": "onnx-splitpoint/evaluation-run-manifest",
                    "schema_version": 1,
                    "run_id": run.name,
                    "profile_id": "holdout_pre_execution",
                    "tool_version": "2.75.17",
                    "workflow_version": (
                        "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair"
                    ),
                    "status": "partial",
                    "created_at": "2026-07-18T00:00:00Z",
                    "model_count": 2,
                    "models": {
                        "holdout_paused": {"model_id": "holdout_paused"},
                        "profile_only": {"model_id": "profile_only"},
                    },
                },
            )
            analysis = run / "models" / "holdout_paused" / "analysis"
            analysis.mkdir(parents=True)
            write_json(analysis / "prediction_freeze_approval.json", {"approved": True})
            write_json(analysis / "prediction_freeze_approval_verification.json", {"ok": True})
            (analysis / "prediction_freeze_approval_request.md").write_text("approval request", encoding="utf-8")

            out = root / "analysis.zip"
            result = create_analysis_pack(
                run, out, materialize_missing_report=True
            )
            self.assertEqual(result["model_count"], 2)
            with zipfile.ZipFile(out) as archive:
                names = set(archive.namelist())
            self.assertIn("04_model_details/holdout_paused/analysis/prediction_freeze_approval.json", names)
            self.assertIn("04_model_details/holdout_paused/analysis/prediction_freeze_approval_verification.json", names)
            self.assertIn("04_model_details/holdout_paused/analysis/prediction_freeze_approval_request.md", names)
            self.assertIn("04_model_details/profile_only/README.md", names)
            self.assertNotIn("04_model_details/disabled_reserve/README.md", names)

    def test_final_template_has_unique_yaml_keys_and_fixed_campaign_matrix(self) -> None:
        path = Path(__file__).resolve().parents[1] / "onnx_splitpoint_tool" / "resources" / "evaluation_profiles" / "thesis_final_campaign_v1.yaml"
        document = yaml.compose(path.read_text(encoding="utf-8"))

        def _check_unique(node: object, location: str = "root") -> None:
            if isinstance(node, MappingNode):
                seen: set[str] = set()
                for key_node, value_node in node.value:
                    key = str(getattr(key_node, "value", ""))
                    self.assertNotIn(key, seen, f"duplicate YAML key {key!r} at {location}")
                    seen.add(key)
                    _check_unique(value_node, f"{location}.{key}")
            elif isinstance(node, SequenceNode):
                for index, value_node in enumerate(node.value):
                    _check_unique(value_node, f"{location}[{index}]")

        _check_unique(document)
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(profile["ranking_validation"]["audit_size"], 20)
        primary_ids = [row["id"] for row in profile["model_suite"]["primary"]]
        self.assertEqual(
            primary_ids,
            [
                "yolov7_paper",
                "resnet50",
                "yolo26s",
                "regnet_x_1_6gf",
                "yolo26m",
            ],
        )
        self.assertFalse(profile["model_suite"]["reserve"][0]["enabled"])
        run_profiles = {row["id"]: row for row in profile["run_profiles"]}
        self.assertTrue({"hailo8_to_trt", "hailo10_to_tensorrt", "deepx_m1_to_tensorrt"}.issubset(run_profiles))
        self.assertTrue(all(run_profiles[run_id]["required"] is False for run_id in ("hailo8_to_trt", "hailo10_to_tensorrt", "deepx_m1_to_tensorrt")))
        native = profile["native_producers"]
        self.assertEqual(native["backends"], ["hailo8", "hailo10h", "deepx"])
        self.assertEqual(set(native["full_baselines"]["backends_by_producer"]), {"hailo8", "hailo10h", "deepx"})

    def _runner(self, root: Path) -> EvaluationWorkflowRunner:
        runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(root)))
        runner.run_dir = root / "EvaluationRun_test"
        runner.run_dir.mkdir(parents=True, exist_ok=True)
        runner.profile_path = ""
        runner.profile_payload = {
            "campaign": {
                "mode": "final",
                "prediction_freeze_enabled": True,
                "require_prediction_freeze_approval": False,
            },
            "execution_preset": {"id": "standard"},
            "selection_policy": {
                "max_accepted_cases_per_model": 1,
                "preferred_shortlist": 10,
                "min_gap": 0,
                "selection_strategy": "predicted_topk",
            },
            # Freezing is a Standard/Final protocol operation and must remain
            # active even when correlation statistics are disabled.
            "ranking_validation": {
                "enabled": False,
                "audit_size": 20,
                "minimum_valid_audit_candidates": 10,
                "audit_seed": 17,
            },
            "run_profiles": [],
        }
        return runner

    def _freeze_fixture(self, root: Path) -> tuple[EvaluationWorkflowRunner, dict, dict, Path]:
        runner = self._runner(root)
        model_path = root / "holdout.onnx"
        model_path.write_bytes(b"frozen-model-identity")
        row = {
            "id": "holdout",
            "resolved_path": str(model_path),
            "evaluation_role": "holdout",
            "family_id": "regnet",
            "generalization_scope": "model_family_holdout",
            "validation_tier": "final",
            "candidate_universe_complete": True,
            "candidate_universe": {
                "mode": "deterministic_audit",
                "audit_size": 20,
                "minimum_valid_candidates": 10,
                "seed": 17,
            },
        }
        prediction = {
            "schema": "onnx-splitpoint/split-prediction",
            "schema_version": 1,
            "artifact_id": "prediction-fixture",
            "model_id": "holdout",
            "selection_strategy": "predicted_topk",
            "analysis_parse_status": "ok",
            "analysis_candidate_source": "onnx_real_boundary_hardware_aware",
            "real_boundary_analysis_ok": True,
            "candidates": [_candidate(index) for index in range(1, 31)],
        }
        prediction_path = runner.run_dir / "models" / "holdout" / "analysis" / "prediction.json"
        write_json(prediction_path, prediction)
        return runner, row, prediction, prediction_path

    def test_stable_candidate_identity_separates_core_from_execution_stratum(self) -> None:
        base = {
            "case_id": "b017",
            "boundary": 17,
            "model_sha256": "a" * 64,
            "direction": "hailo8_to_tensorrt",
            "runner_regime": "generic",
            "setup_id": "setup-a",
            "precision": "fp16",
            "contract_id": "contract-v1",
        }
        first = stable_candidate_identity("model-a", base, 17)
        second = stable_candidate_identity("model-a", {**base, "direction": "deepx_to_tensorrt"}, 17)
        changed_model = stable_candidate_identity("model-a", {**base, "model_sha256": "b" * 64}, 17)
        self.assertEqual(first["candidate_id"], "model-a::b017")
        self.assertEqual(first["candidate_identity_sha256"], second["candidate_identity_sha256"])
        self.assertNotEqual(first["execution_candidate_identity_sha256"], second["execution_candidate_identity_sha256"])
        self.assertNotEqual(first["candidate_identity_sha256"], changed_model["candidate_identity_sha256"])

    def test_final_readiness_is_fail_closed_for_missing_evaluation_role(self) -> None:
        profile = {
            "campaign": {"mode": "final", "enforcement": "strict"},
            "model_suite": {
                "primary": [{
                    "id": "role_missing",
                    "family_id": "family-a",
                    "generalization_scope": "development",
                    "validation_tier": "final",
                    "candidate_universe": {"mode": "declared_shortlist"},
                }],
                "reserve": [],
            },
        }
        checks = _checks(build_campaign_readiness(profile))
        self.assertEqual(checks["model_role_explicit_role_missing"]["status"], "fail")
        self.assertEqual(checks["development_models_present"]["status"], "fail")

    def test_final_linter_distinguishes_family_and_within_family_holdout(self) -> None:
        dev = {
            "id": "yolo26s",
            "evaluation_role": "development",
            "family_id": "yolo26",
            "generalization_scope": "development",
            "validation_tier": "final",
            "candidate_universe": {"mode": "declared_shortlist"},
        }
        holdout = {
            "id": "yolo26m",
            "evaluation_role": "holdout",
            "family_id": "yolo26",
            "generalization_scope": "within_family_transfer",
            "validation_tier": "final",
            "candidate_universe": {
                "mode": "deterministic_audit",
                "audit_size": 20,
                "minimum_valid_candidates": 10,
            },
            "unseen_attestation": {
                "unseen": True,
                "attested_by": "test",
                "attested_at": "2026-07-18T00:00:00Z",
            },
        }
        profile = {
            "campaign": {"mode": "final", "enforcement": "strict"},
            "model_suite": {"primary": [dev, holdout], "reserve": []},
        }
        within_checks = _checks(build_campaign_readiness(profile))
        self.assertEqual(within_checks["within_family_anchor_yolo26m"]["status"], "pass")

        family_profile = copy.deepcopy(profile)
        family_profile["model_suite"]["primary"][1]["generalization_scope"] = "model_family_holdout"
        family_checks = _checks(build_campaign_readiness(family_profile))
        self.assertEqual(family_checks["family_holdout_disjoint_yolo26m"]["status"], "fail")

    def test_disabled_reserve_is_not_active_campaign_projection(self) -> None:
        profile = {
            "campaign": {"mode": "final", "enforcement": "strict"},
            "model_suite": {
                "primary": [{
                    "id": "dev",
                    "evaluation_role": "development",
                    "family_id": "dev-family",
                    "generalization_scope": "development",
                    "validation_tier": "final",
                    "candidate_universe": {"mode": "declared_shortlist"},
                }],
                "reserve": [{"id": "yolo26xl", "enabled": False}],
            },
        }
        self.assertNotIn("yolo26xl", [row["id"] for row in profile_model_entries(profile, include_reserve=True)])
        checks = _checks(build_campaign_readiness(profile))
        self.assertNotIn("model_role_explicit_yolo26xl", checks)

    def test_audit_and_deployment_roles_reach_normalized_result_rows(self) -> None:
        plan = {
            "artifact_id": "candidate-plan-a",
            "candidate_universe_sha256": "sha256:universe-a",
            "selected_candidates": [
                {
                    "case_id": "b007",
                    "candidate_id": "holdout::b007",
                    "candidate_identity_sha256": "sha256:core-a",
                    "candidate_execution_roles": ["audit", "deployment_shortlist"],
                    "audit_rank": 3,
                    "deployment_rank": 1,
                },
                {
                    "boundary": 9,
                    "candidate_id": "holdout::b009",
                    "candidate_execution_roles": ["audit"],
                    "audit_rank": 4,
                },
            ],
        }
        rows = _annotate_candidate_execution_contract_v261e(
            [
                {"case_id": "b007", "variant": "composed"},
                {"split_index": 9, "variant": "native_split"},
                {"case_id": "full", "variant": "full"},
            ],
            plan,
        )
        self.assertEqual(rows[0]["candidate_execution_roles"], ["audit", "deployment_shortlist"])
        self.assertTrue(rows[0]["selected_for_score_independent_audit"])
        self.assertTrue(rows[0]["selected_for_deployment_shortlist"])
        self.assertEqual(rows[1]["candidate_execution_roles"], ["audit"])
        self.assertTrue(rows[1]["selected_for_score_independent_audit"])
        self.assertNotIn("candidate_execution_roles", rows[2])

    def test_execution_identity_is_recomputed_for_each_measured_stratum(self) -> None:
        plan = {
            "artifact_id": "candidate-plan-a",
            "candidate_universe_sha256": "sha256:universe-a",
            "selected_candidates": [{
                "case_id": "b007",
                "candidate_id": "holdout::b007",
                "candidate_identity_sha256": "sha256:frozen-core",
                # A plan-side execution ID is necessarily incomplete and must
                # never leak into normalized measurements.
                "execution_candidate_id": "holdout::b007::plan-placeholder",
                "execution_candidate_identity_sha256": "sha256:plan-placeholder",
                "candidate_execution_roles": ["audit"],
            }],
        }
        measured = []
        for direction in ("hailo8_to_tensorrt", "deepx_to_tensorrt"):
            for setup in ("setup-a", "setup-b"):
                for precision in ("fp16", "int8"):
                    measured.append({
                        "case_id": "b007",
                        "direction": direction,
                        "setup_id": setup,
                        "precision": precision,
                        "contract_sha256": "sha256:interface-v1",
                    })

        rows = _annotate_candidate_execution_contract_v261e(measured, plan)
        self.assertEqual({row["candidate_id"] for row in rows}, {"holdout::b007"})
        self.assertEqual(
            {row["candidate_identity_sha256"] for row in rows},
            {"sha256:frozen-core"},
        )
        self.assertEqual(len({row["execution_candidate_id"] for row in rows}), 8)
        self.assertEqual(len({row["execution_candidate_identity_sha256"] for row in rows}), 8)
        self.assertNotIn("holdout::b007::plan-placeholder", {row["execution_candidate_id"] for row in rows})
        self.assertTrue(all(
            row["candidate_execution_identity_source"]
            == "normalized_result_plus_frozen_candidate_core"
            for row in rows
        ))

    def test_freeze_is_independent_of_ranking_statistics_and_audit_is_not_shortlist_limited(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner, row, prediction, prediction_path = self._freeze_fixture(Path(temp))
            _artifacts, metrics = runner._freeze_prediction_artifact(
                model_id="holdout", row=row, prediction=prediction, prediction_path=prediction_path
            )
            self.assertEqual(metrics["prediction_freeze_status"], "prospective_frozen")
            manifest = read_json(prediction_path.parent / "prediction_freeze_manifest.json", default={}) or {}
            self.assertFalse(manifest["ranking_statistics_enabled"])
            universe = read_json(prediction_path.parent / "candidate_universe_manifest.json", default={}) or {}
            self.assertEqual(universe["audit"]["selected_size"], 20)
            self.assertFalse(universe["audit"]["selection_uses_predictions"])
            loaded_freeze = _load_prediction_freeze(
                runner.run_dir / "models" / "holdout",
                prediction_path,
            )
            self.assertTrue(loaded_freeze["valid"])
            self.assertTrue(loaded_freeze["candidate_universe_valid"])
            self.assertEqual(len(loaded_freeze["candidate_universe_selected_case_ids"]), 20)
            self.assertEqual(loaded_freeze["candidate_universe_minimum_valid_candidates"], 10)

            _artifacts, selection_metrics, _message, status = runner._stage_select_split_candidates("holdout", row)
            self.assertEqual(status, "ok")
            self.assertEqual(selection_metrics["audit_candidate_count"], 20)
            self.assertEqual(selection_metrics["deployment_shortlist_count"], 1)
            self.assertGreaterEqual(selection_metrics["execution_union_count"], 20)
            self.assertLessEqual(selection_metrics["execution_union_count"], 21)
            audit_plan = read_json(prediction_path.parent / "audit_plan.json", default={}) or {}
            self.assertIn("limits only the deployment shortlist", audit_plan["methodological_rule"])

    def test_conflicting_resume_preserves_original_frozen_universe_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner, row, prediction, prediction_path = self._freeze_fixture(Path(temp))
            runner._freeze_prediction_artifact(
                model_id="holdout", row=row, prediction=prediction, prediction_path=prediction_path
            )
            protected_names = (
                "prediction_freeze_manifest.json",
                "predictions_frozen.csv",
                "ranking_predictions_frozen.csv",
                "candidate_universe_manifest.json",
                "candidate_universe.csv",
            )
            before = {name: (prediction_path.parent / name).read_bytes() for name in protected_names}

            changed = copy.deepcopy(prediction)
            changed["artifact_id"] = "changed-prediction"
            changed["candidates"][0]["cut_bytes"] += 1
            write_json(prediction_path, changed)
            _artifacts, metrics = runner._freeze_prediction_artifact(
                model_id="holdout", row=row, prediction=changed, prediction_path=prediction_path
            )
            self.assertEqual(metrics["prediction_freeze_status"], "hash_conflict")
            after = {name: (prediction_path.parent / name).read_bytes() for name in protected_names}
            self.assertEqual(before, after)
            conflict = json.loads((prediction_path.parent / "prediction_freeze_conflict.json").read_text(encoding="utf-8"))
            self.assertNotEqual(
                conflict["existing_manifest"]["candidate_universe_sha256"],
                conflict["current"]["candidate_universe_sha256"],
            )

    def test_holdout_freeze_requires_successful_real_boundary_analysis(self) -> None:
        scenarios = {
            "parse_failed": lambda prediction: prediction.update({"analysis_parse_status": "failed"}),
            "placeholder_candidates": lambda prediction: [
                candidate.update({"source": "workflow_contract_placeholder"})
                for candidate in prediction["candidates"]
            ],
            "empty_candidates": lambda prediction: prediction.update({"candidates": []}),
        }
        for name, mutate in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                runner, row, prediction, prediction_path = self._freeze_fixture(Path(temp))
                changed = copy.deepcopy(prediction)
                mutate(changed)
                write_json(prediction_path, changed)

                artifacts, metrics = runner._freeze_prediction_artifact(
                    model_id="holdout",
                    row=row,
                    prediction=changed,
                    prediction_path=prediction_path,
                )

                self.assertEqual(metrics["prediction_freeze_status"], "holdout_real_analysis_required")
                self.assertFalse(metrics["predictions_frozen"])
                self.assertFalse(metrics["predictions_frozen_valid_for_holdout"])
                self.assertFalse(metrics["prediction_freeze_approval_eligible"])
                self.assertFalse((prediction_path.parent / "prediction_freeze_manifest.json").exists())
                self.assertFalse((prediction_path.parent / "candidate_universe_manifest.json").exists())
                rejection = read_json(artifacts["prediction_freeze_rejection_json"], default={}) or {}
                self.assertFalse(rejection["approval_eligible"])
                with self.assertRaises(ValueError):
                    create_prediction_freeze_approval(
                        run_dir=runner.run_dir,
                        model_id="holdout",
                        signer="Must Not Succeed",
                    )

    def test_development_placeholder_prediction_remains_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner, row, prediction, prediction_path = self._freeze_fixture(Path(temp))
            development_row = {
                **row,
                "evaluation_role": "development",
                "candidate_universe": {"mode": "declared_shortlist"},
            }
            diagnostic = copy.deepcopy(prediction)
            diagnostic["analysis_parse_status"] = "failed"
            for candidate in diagnostic["candidates"]:
                candidate["source"] = "workflow_contract_placeholder"
            write_json(prediction_path, diagnostic)

            _artifacts, metrics = runner._freeze_prediction_artifact(
                model_id="holdout",
                row=development_row,
                prediction=diagnostic,
                prediction_path=prediction_path,
            )
            self.assertEqual(metrics["prediction_freeze_status"], "prospective_frozen")
            self.assertTrue(metrics["predictions_frozen"])
            self.assertFalse(metrics["predictions_frozen_valid_for_holdout"])
            manifest = read_json(prediction_path.parent / "prediction_freeze_manifest.json", default={}) or {}
            self.assertEqual(manifest["evaluation_role"], "development")
            self.assertFalse(manifest["valid_for_holdout"])

    def test_existing_placeholder_freeze_is_preserved_but_made_unapprovable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner, row, prediction, prediction_path = self._freeze_fixture(Path(temp))
            runner._freeze_prediction_artifact(
                model_id="holdout",
                row=row,
                prediction=prediction,
                prediction_path=prediction_path,
            )
            manifest_path = prediction_path.parent / "prediction_freeze_manifest.json"
            original_manifest = manifest_path.read_bytes()

            placeholder = copy.deepcopy(prediction)
            placeholder["analysis_parse_status"] = "failed"
            for candidate in placeholder["candidates"]:
                candidate["source"] = "workflow_contract_placeholder"
            write_json(prediction_path, placeholder)
            artifacts, metrics = runner._freeze_prediction_artifact(
                model_id="holdout",
                row=row,
                prediction=placeholder,
                prediction_path=prediction_path,
            )

            self.assertEqual(metrics["prediction_freeze_status"], "holdout_real_analysis_required")
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            self.assertTrue(artifacts["prediction_freeze_conflict_json"].is_file())
            with self.assertRaises(ValueError):
                create_prediction_freeze_approval(
                    run_dir=runner.run_dir,
                    model_id="holdout",
                    signer="Must Not Succeed",
                )


if __name__ == "__main__":
    unittest.main()
