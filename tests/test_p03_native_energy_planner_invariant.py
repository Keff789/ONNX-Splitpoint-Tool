from __future__ import annotations

import base64
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]

CURRENT_CASES = {
    "resnet50": ("b024", "b046", "b052", "b074", "b103"),
    "yolo26s": ("b021", "b024", "b026", "b035", "b038"),
    "yolov7_paper": ("b009", "b011", "b044", "b063", "b066"),
}
LEGACY_CASES = {
    model: ("b001", "b002", "b003") for model in CURRENT_CASES
}
SETUPS = (
    (
        "deepx", "deepx_to_trt", "native_full_deepx",
        "orin_nx_deepx_m1_01",
    ),
    (
        "hailo8", "hailo8_to_trt", "native_full_hailo8",
        "orin_nx_hailo8_01",
    ),
    (
        "hailo10h", "hailo10h_to_trt", "native_full_hailo10h",
        "orin_nx_hailo10_01",
    ),
)


def _load_script(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COORDINATOR = _load_script(
    "p03_native_coordinator",
    "scripts/run_evalrun_native_producer_variants.py",
)
PLANNER = _load_script(
    "p03_native_energy_planner",
    "scripts/native_producer_energy_plan.py",
)
ENERGY_RUNNER = _load_script(
    "p03_native_energy_runner",
    "scripts/run_native_producer_energy_from_summary.py",
)


def _split_precision(producer: str, model: str) -> str:
    if producer == "hailo8" and model != "resnet50":
        return "uint8_dequant_fp16"
    return "float32_layout_fp16"


def _matrix_rows(
    case_map: Mapping[str, tuple[str, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    theoretical: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    for producer, split_backend, vendor_full, setup_id in SETUPS:
        for model, cases in case_map.items():
            task = "classification" if model == "resnet50" else "detection"
            precision = _split_precision(producer, model)
            for case in cases:
                theoretical.append({
                    "execution_mode": "native_split",
                    "backend_key": producer,
                    "backend": producer,
                    "setup_id": setup_id,
                    "model": model,
                    "case": case,
                    "variant": f"{producer}-{model}",
                })
                observed.append({
                    "ok": True,
                    "backend": split_backend,
                    "model": model,
                    "case": case,
                    "precision": precision,
                    "setup_id": setup_id,
                    "comparison_backend": producer,
                    "task": task,
                    "fps_makespan": 10.0,
                    "native_command_contract": {
                        "contract_sha256": "a" * 64,
                    },
                })
            for backend, fps in (
                (vendor_full, 8.0),
                ("native_full_tensorrt", 12.0),
            ):
                theoretical.append({
                    "execution_mode": "native_full_baseline",
                    "backend_key": producer,
                    "backend": backend,
                    "setup_id": setup_id,
                    "comparison_backend": producer,
                    "model": model,
                    "case": "full",
                    "variant": f"{producer}-{model}",
                })
                observed.append({
                    "ok": True,
                    "execution_mode": "native_full_baseline",
                    "backend": backend,
                    "model": model,
                    "case": "full",
                    "precision": "uint8_cast_fp16",
                    "runtime_precision_identity": "uint8_cast_fp16",
                    "setup_id": setup_id,
                    "comparison_backend": producer,
                    "task": task,
                    "fps_makespan": fps,
                    "full_command_contract": {
                        "contract_sha256": "b" * 64,
                    },
                })
    return theoretical, observed


def _validation_rows(
    rows: list[dict[str, Any]], *, mixed: bool,
) -> list[dict[str, Any]]:
    states = ("pass", "fail", "inconclusive", "unavailable")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        # Deliberately remove both Full annotations for one setup/model so the
        # post-hoc pairing result is provably missing while plan membership is
        # unchanged.
        if (
            mixed
            and str(row.get("backend") or "").startswith("native_full_")
            and row.get("setup_id") == "orin_nx_deepx_m1_01"
            and row.get("model") == "resnet50"
        ):
            continue
        state = states[index % len(states)] if mixed else "pass"
        positive = state == "pass"
        result.append({
            **row,
            "ok": positive,
            "status": state,
            "quality_gate_status": state,
            "semantic_ok": positive,
            "contract_consistent": positive,
            "top1_match": positive,
            "claim_ok": positive,
            "central_quality_evidence_verified": positive,
            "precision_quality_verified": positive,
            "precision_quality_binding_verified": positive,
            "task_quality_observation_valid": positive,
            "accuracy_gate_pass": positive,
            "quality_claim_result_verified": positive,
        })
    return result


def _native_trt_meta() -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/native-trt-meta",
        "schema_version": 1,
        "variant": "part2",
        "build_ok": True,
        "inputs_static": True,
        "inputs": [{
            "name": "cut",
            "shape": [1, 8, 8, 16],
            "elem_type": "FLOAT",
            "has_dynamic": False,
        }],
    }


def _verified_contract(
    raw: Any, *, expected_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = {
        "contract_sha256": str(
            (raw or {}).get("contract_sha256") or "c" * 64
        ),
        "backend": expected_identity["backend"],
        "model": expected_identity["model"],
        "case": expected_identity["case"],
        "precision": expected_identity.get("precision") or "uint8_cast_fp16",
        "setup_id": expected_identity["setup_id"],
        "comparison_backend": expected_identity["comparison_backend"],
        "runtime_options": {},
        "energy_workload": {},
        "artifacts": {},
    }
    if not str(expected_identity["backend"]).startswith("native_full_"):
        meta = _native_trt_meta()
        contract["native_split_quality_binding"] = {
            "native_trt_meta": dict(meta),
            "native_trt_meta_payload": dict(meta),
        }
    return contract, "verified_test_contract"


def _portable_binding(
    value: Any, **_kwargs: Any,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(value, Mapping):
        return None, "test_binding_missing"
    return dict(value), (
        "portable_embedded_evidence_and_cross_links_verified_without_local_rehash"
    )


def _verified_part2_contract(
    _value: Any,
) -> tuple[dict[str, Any], str]:
    return _native_trt_meta(), "verified_test_single_static_part2_input"


def _membership(row: Mapping[str, Any]) -> tuple[str, ...]:
    identity = COORDINATOR._native_performance_identity(row)
    if identity is None:
        raise AssertionError(f"invalid membership row: {row}")
    return identity


def _raw_technical_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Test-local identity independent of the production normalizer."""

    backend = str(row.get("backend") or row.get("backend_key") or "").lower()
    backend = {
        "deepx": "deepx_to_trt",
        "hailo8": "hailo8_to_trt",
        "hailo10": "hailo10h_to_trt",
        "hailo10h": "hailo10h_to_trt",
    }.get(backend, backend)
    comparison = str(row.get("comparison_backend") or "").lower()
    if not comparison and backend.endswith("_to_trt"):
        comparison = {
            "deepx_to_trt": "deepx",
            "hailo8_to_trt": "hailo8",
            "hailo10h_to_trt": "hailo10h",
        }.get(backend, "")
    return (
        backend,
        str(row.get("model") or row.get("model_id") or "").lower(),
        str(row.get("case") or row.get("case_id") or "").lower(),
        str(row.get("setup_id") or row.get("setup") or "").lower(),
        comparison,
    )


class NativeEnergyPlannerInvariantTests(unittest.TestCase):
    maxDiff = None

    def _run_plan(
        self,
        case_map: Mapping[str, tuple[str, ...]],
        *,
        mixed_annotations: bool,
        observed_mutator: Any = None,
        split_runtime_builder: Any = None,
        split_preflight_builder: Any = None,
        split_quality_evidence_builder: Any = None,
        validation_text_override: str | None = None,
        invalid_detection_exclusion_contract: bool = False,
        require_matrix_presence_complete: bool = True,
        fixture_builder: Any = None,
    ) -> tuple[
        dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
    ]:
        theoretical, observed = _matrix_rows(case_map)
        if observed_mutator is not None:
            observed_mutator(observed)
        if fixture_builder is not None:
            theoretical, observed, matrix = fixture_builder(
                theoretical, observed,
            )
        else:
            matrix = COORDINATOR._native_performance_expected_matrix(
                theoretical, observed,
            )
        if require_matrix_presence_complete:
            self.assertTrue(matrix["row_presence_complete"], matrix)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = root / "native_producer_summary.json"
            validation = root / "native_producer_validation_summary.json"
            output = root / "energy-plan"
            summary.write_text(
                json.dumps({"rows": observed}), encoding="utf-8",
            )
            validation.write_text(
                validation_text_override
                if validation_text_override is not None
                else json.dumps({
                    "rows": _validation_rows(
                        observed, mixed=mixed_annotations,
                    ),
                }),
                encoding="utf-8",
            )
            (root / "native_expected_matrix.json").write_text(
                json.dumps(matrix), encoding="utf-8",
            )

            argv = [
                str(PLANNER.__file__),
                "--summary", str(summary),
                "--validation-summary", str(validation),
                "--out-dir", str(output),
                "--hailo8-ssh", "hailo8-host",
                "--hailo10-ssh", "hailo10-host",
                "--deepx-ssh", "deepx-host",
                "--duration-s", "1",
                "--screening-energy",
            ]
            if invalid_detection_exclusion_contract:
                invalid_exclusions = root / "invalid-exclusions.json"
                invalid_exclusions.write_text(
                    json.dumps({"schema": "not-the-contract"}),
                    encoding="utf-8",
                )
                argv.extend([
                    "--detection-claim-exclusions",
                    str(invalid_exclusions),
                    "--detection-claim-exclusions-sha256",
                    "0" * 64,
                ])
            patches = (
                mock.patch.object(
                    PLANNER, "_verify_full_command_contract",
                    _verified_contract,
                ),
                mock.patch.object(
                    PLANNER, "verify_native_energy_command_contract",
                    _verified_contract,
                ),
                mock.patch.object(
                    PLANNER, "verify_native_split_part2_input_contract",
                    _verified_part2_contract,
                ),
                mock.patch.object(
                    PLANNER, "validate_native_split_quality_binding",
                    _portable_binding,
                ),
                mock.patch.object(
                    PLANNER, "_split_quality_energy_evidence",
                    split_quality_evidence_builder or (
                        lambda *_args, **_kwargs: ({
                            "native_split_quality_required": False,
                            "native_split_energy_binding_valid": True,
                            "native_split_energy_binding_status": (
                                "runtime_measurement_quality_annotation"
                            ),
                        }, "runtime_measurement_quality_annotation")
                    ),
                ),
                mock.patch.object(
                    PLANNER, "split_energy_runtime_argv",
                    split_runtime_builder or (
                        lambda *_args, **_kwargs: [
                            "python", "split-energy.py",
                        ]
                    ),
                ),
                mock.patch.object(
                    PLANNER, "_full_runtime_argv",
                    lambda *_args, **_kwargs: [
                        "python", "full-energy.py",
                    ],
                ),
                mock.patch.object(
                    PLANNER, "_split_preflight_argv",
                    split_preflight_builder or (
                        lambda *_args, **_kwargs: [
                            "python", "split-preflight.py",
                        ]
                    ),
                ),
                mock.patch.object(
                    PLANNER, "_full_preflight_argv",
                    lambda *_args, **_kwargs: [
                        "python", "full-preflight.py",
                    ],
                ),
                mock.patch.object(
                    PLANNER, "_process_local_runtime_environment",
                    lambda _contract: {},
                ),
                mock.patch.object(sys, "argv", argv),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patches[5], patches[6], patches[7], patches[8], patches[9], \
                    patches[10]:
                self.assertEqual(PLANNER.main(), 0)
            payload = json.loads(
                (output / "native_producer_energy_plan.json").read_text(
                    encoding="utf-8",
                )
            )
        return payload, theoretical, observed

    def test_current_63_matrix_membership_is_annotation_invariant(self) -> None:
        all_pass, theoretical, all_pass_observed = self._run_plan(
            CURRENT_CASES, mixed_annotations=False,
        )

        def blank_one_full_display_precision(
            rows: list[dict[str, Any]],
        ) -> None:
            target = next(
                row for row in rows
                if row["backend"] == "native_full_deepx"
                and row["model"] == "resnet50"
            )
            target["precision"] = ""
            target["runtime_precision_identity"] = ""

        mixed, mixed_theoretical, mixed_observed = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=True,
            observed_mutator=blank_one_full_display_precision,
        )

        def blank_one_split_display_precision(
            rows: list[dict[str, Any]],
        ) -> None:
            target = next(
                row for row in rows
                if row["backend"] == "hailo8_to_trt"
                and row["model"] == "resnet50"
            )
            target["precision"] = ""

        missing_split_precision, _, _ = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            observed_mutator=blank_one_split_display_precision,
        )

        malformed_validation, _, _ = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            validation_text_override="{not-json",
        )
        invalid_validation_shape, _, _ = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            validation_text_override=json.dumps({"rows": [1]}),
        )
        invalid_claim_contract, _, _ = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            invalid_detection_exclusion_contract=True,
        )

        def raise_quality_annotation(*_args: Any, **_kwargs: Any):
            raise OSError("fixture downstream quality annotation unavailable")

        unavailable_quality_binding, _, _ = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            split_quality_evidence_builder=raise_quality_annotation,
        )

        def malformed_endpoint_annotation(
            rows: list[dict[str, Any]],
        ) -> None:
            target = next(
                row for row in rows
                if row["task"] == "detection"
                and not str(row["backend"]).startswith("native_full_")
            )
            target["completed_task_endpoint_attestation"] = {
                "completed_task_completion_mode": "fixture",
                "completed_task_comparison_endpoint_contract": 1,
            }

        malformed_endpoint, _, _ = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            observed_mutator=malformed_endpoint_annotation,
        )

        for payload in (
            all_pass,
            mixed,
            malformed_validation,
            invalid_validation_shape,
            invalid_claim_contract,
            unavailable_quality_binding,
            malformed_endpoint,
            missing_split_precision,
        ):
            self.assertEqual(payload["energy_matrix_expected_count"], 63)
            self.assertEqual(payload["energy_plan_included_count"], 63)
            self.assertEqual(payload["energy_plan_excluded_count"], 0)
            self.assertTrue(payload["energy_plan_coverage_contract_valid"])
            self.assertEqual(
                payload["energy_plan_invalid_membership_identity_count"], 0,
            )
            self.assertEqual(payload["energy_plan_ledger_overlap_identities"], [])
            self.assertEqual(payload["energy_plan_ledger_unexpected_identities"], [])
            self.assertEqual(payload["energy_plan_ledger_missing_identities"], [])
            rows = payload["rows"]
            self.assertEqual(Counter(row["setup_id"] for row in rows), {
                "orin_nx_deepx_m1_01": 21,
                "orin_nx_hailo8_01": 21,
                "orin_nx_hailo10_01": 21,
            })
            self.assertEqual(Counter(
                (row["setup_id"], row["model"]) for row in rows
            ), {
                (setup, model): 7
                for setup in (
                    "orin_nx_deepx_m1_01",
                    "orin_nx_hailo8_01",
                    "orin_nx_hailo10_01",
                )
                for model in CURRENT_CASES
            })
            self.assertEqual(sum(row["full_baseline"] for row in rows), 18)
            self.assertEqual(sum(
                not row["full_baseline"] for row in rows
            ), 45)
            for row in rows:
                admission = row["native_energy_planner_admission"]
                self.assertTrue(admission["runtime_success"])
                self.assertTrue(admission["energy_command_preflight_ok"])
                self.assertTrue(
                    admission["full_baseline"]
                    or admission["split_has_valid_part2_input"]
                )
                if admission["full_baseline"]:
                    self.assertFalse(
                        admission["split_has_valid_part2_input"]
                    )
                    self.assertIsNone(
                        admission["effective_part2_input_count"]
                    )

        recovered_split = next(
            row for row in missing_split_precision["rows"]
            if row["backend"] == "hailo8_to_trt"
            and row["model"] == "resnet50"
            and row["case"] == "b024"
        )
        self.assertEqual(
            recovered_split["split_boundary_precision"],
            "uint8_cast_fp16",
        )
        self.assertTrue(
            recovered_split["native_energy_planner_admission"][
                "split_has_valid_part2_input"
            ]
        )

        for payload in (malformed_validation, invalid_validation_shape):
            self.assertEqual(
                payload["validation_summary_status"],
                "unavailable_invalid_annotation_document",
            )
            self.assertTrue(
                payload["preflight"]["measurement_start_allowed"]
            )
        self.assertEqual(
            invalid_claim_contract["detection_claim_exclusions_status"],
            "file_or_sha256_invalid",
        )
        self.assertTrue(
            invalid_claim_contract["preflight"][
                "measurement_start_allowed"
            ]
        )
        detection_rows = [
            row for row in invalid_claim_contract["rows"]
            if row["task"] == "detection"
        ]
        self.assertTrue(detection_rows)
        for row in detection_rows:
            self.assertTrue(row["detection_claim_annotation_invalid"])
            self.assertTrue(row["diagnostic_only"])
            self.assertFalse(row["claim_eligible"])
            self.assertFalse(row["energy_claim_eligible"])
            self.assertIn(
                "detection_claim_exclusions_contract_invalid",
                row["scientific_claim_exclusion_reasons"],
            )
        self.assertTrue(all(
            row["diagnostic_only"]
            for row in unavailable_quality_binding["rows"]
            if not row["full_baseline"]
        ))
        self.assertTrue(any(
            str(row["completion_pairing_status"]).startswith(
                "downstream_endpoint_annotation_unavailable:"
            )
            for row in malformed_endpoint["rows"]
        ))

        self.assertEqual(
            {_membership(row) for row in all_pass["rows"]},
            {_membership(row) for row in mixed["rows"]},
        )
        for payload, planned, observed in (
            (all_pass, theoretical, all_pass_observed),
            (mixed, mixed_theoretical, mixed_observed),
        ):
            expected_raw = {
                _raw_technical_identity(row) for row in planned
            }
            observed_raw = {
                _raw_technical_identity(row) for row in observed
            }
            plan_raw = {
                _raw_technical_identity(row) for row in payload["rows"]
            }
            self.assertEqual(len(expected_raw), len(planned))
            self.assertEqual(len(observed_raw), len(observed))
            self.assertEqual(plan_raw, expected_raw)
            self.assertEqual(plan_raw, observed_raw)
        theoretical_split = next(
            row for row in theoretical
            if row["backend"] == "hailo8"
            and row["model"] == "resnet50"
            and row["case"] == "b024"
        )
        observed_split = next(
            row for row in all_pass_observed
            if row["backend"] == "hailo8_to_trt"
            and row["model"] == "resnet50"
            and row["case"] == "b024"
        )
        self.assertNotIn("precision", theoretical_split)
        self.assertEqual(
            observed_split["precision"], "float32_layout_fp16",
        )
        self.assertEqual(
            _raw_technical_identity(theoretical_split),
            _raw_technical_identity(observed_split),
        )
        observed_states = {
            str(row.get("quality_gate_status") or "")
            for row in mixed["rows"]
        }
        self.assertTrue(
            {"pass", "fail", "inconclusive", "unavailable"}
            .issubset(observed_states)
        )
        missing_pair_rows = [
            row for row in mixed["quality_pairing_posthoc_excluded_rows"]
            if row.get("reason") == "pair_baseline_missing"
            and row.get("setup_id") == "orin_nx_deepx_m1_01"
            and row.get("model") == "resnet50"
        ]
        self.assertTrue(missing_pair_rows)
        self.assertTrue(any(
            "vendor_full" in list(row.get("missing_baselines") or [])
            for row in missing_pair_rows
        ))

    def test_legacy_45_matrix_and_nonclaimable_measurement_regression(self) -> None:
        payload, theoretical, observed = self._run_plan(
            LEGACY_CASES, mixed_annotations=True,
        )
        self.assertEqual(payload["energy_matrix_expected_count"], 45)
        self.assertEqual(payload["energy_plan_included_count"], 45)
        self.assertEqual(payload["energy_plan_excluded_count"], 0)
        self.assertTrue(payload["energy_plan_coverage_contract_valid"])
        self.assertEqual(Counter(
            row["setup_id"] for row in payload["rows"]
        ), {
            "orin_nx_deepx_m1_01": 15,
            "orin_nx_hailo8_01": 15,
            "orin_nx_hailo10_01": 15,
        })
        self.assertEqual(sum(
            not row["full_baseline"] for row in payload["rows"]
        ), 27)
        self.assertEqual(sum(
            row["full_baseline"] for row in payload["rows"]
        ), 18)
        plan_raw = {
            _raw_technical_identity(row) for row in payload["rows"]
        }
        self.assertEqual(
            plan_raw,
            {_raw_technical_identity(row) for row in theoretical},
        )
        self.assertEqual(
            plan_raw,
            {_raw_technical_identity(row) for row in observed},
        )
        for row in payload["rows"]:
            self.assertTrue(row["diagnostic_only"])
            self.assertEqual(
                row["energy_quality_admission"]["admission_scope"],
                "native_runtime_observation",
            )
            for field in (
                "claim_ok", "semantic_claim_ok", "claim_eligible",
                "energy_claim_eligible",
                "eligible_for_energy_results_import",
                "eligible_for_scientific_claim",
            ):
                self.assertIs(row[field], False, (field, row))

    def test_part2_truth_table_and_full_baseline_bypass(self) -> None:
        from onnx_splitpoint_tool.workflow.native_energy_preflight import (
            build_native_energy_preflight,
            native_energy_preflight_blocks_streaming,
        )

        split_row = {
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b024",
            "precision": "float32_layout_fp16",
            "setup_id": "orin_nx_hailo8_01",
        }
        contract, _ = _verified_contract(
            {"contract_sha256": "a" * 64},
            expected_identity={
                **split_row,
                "comparison_backend": "hailo8",
            },
        )
        with mock.patch.object(
            PLANNER, "verify_native_split_part2_input_contract",
            _verified_part2_contract,
        ):
            self.assertEqual(
                PLANNER._split_part2_input_admission(split_row, contract),
                (
                    True,
                    "verified_test_single_static_part2_input",
                    1,
                ),
            )
            # An unsealed summary count is never an admission authority and
            # cannot veto the cryptographically bound Part-2 proof.
            explicit_conflict = {**split_row, "part2_input_count": 2}
            self.assertEqual(
                PLANNER._split_part2_input_admission(
                    explicit_conflict, contract,
                ),
                (
                    True,
                    "verified_test_single_static_part2_input",
                    1,
                ),
            )
        with mock.patch.object(
            PLANNER, "verify_native_split_part2_input_contract",
            lambda _contract: (
                None, "native_energy_part2_single_static_input_invalid"
            ),
        ):
            valid, status, count = PLANNER._split_part2_input_admission(
                split_row, contract,
            )
        self.assertFalse(valid)
        self.assertEqual(
            status, "native_energy_part2_single_static_input_invalid",
        )
        self.assertIsNone(count)

        # Full admission is represented directly by the invariant's OR branch;
        # it intentionally carries no Part-2 proof or input count.
        full_predicate = {
            "runtime_success": True,
            "energy_command_preflight_ok": True,
            "full_baseline": True,
            "split_has_valid_part2_input": False,
        }
        self.assertTrue(
            full_predicate["runtime_success"]
            and full_predicate["energy_command_preflight_ok"]
            and (
                full_predicate["full_baseline"]
                or full_predicate["split_has_valid_part2_input"]
            )
        )
        preflight = build_native_energy_preflight(
            expected_rows=[{
                "execution_mode": "native_split",
                "backend": "hailo8",
                "backend_key": "hailo8",
                "setup_id": "orin_nx_hailo8_01",
                "model": "resnet50",
                "case": "b024",
            }],
            energy_requested=True,
            energy_evidence_tier="screening",
            strict_requested=False,
            validation_requested=False,
            full_baselines_enabled=False,
            setup_ids_by_producer={
                "hailo8": "orin_nx_hailo8_01",
            },
        )
        self.assertTrue(preflight["plan_viable"])
        self.assertFalse(
            native_energy_preflight_blocks_streaming(preflight)
        )
        self.assertIn(
            "native_validation_not_requested",
            preflight["nonblocking_annotations"],
        )
        self.assertIn(
            "no_theoretical_setup_local_energy_pair",
            preflight["nonblocking_annotations"],
        )

    def test_real_part2_proof_is_quality_role_independent(self) -> None:
        from onnx_splitpoint_tool.native_command_contract import (
            canonical_json_sha256,
            verify_native_command_contract,
            verify_native_energy_command_contract,
            verify_native_split_part2_input_contract,
        )
        from onnx_splitpoint_tool.resume_artifact_contract import (
            resume_artifact_requirements,
        )

        contract_path = (
            ROOT / "tests" / "fixtures" / "v2725_hailo8_resume"
            / (
                "hailo8_to_trt__resnet50__b052__"
                "orin_nx_hailo8_01__8eb40a232207.command_contract.json"
            )
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        verified, status = verify_native_energy_command_contract(contract)
        self.assertIsNotNone(verified, status)
        metadata, status = verify_native_split_part2_input_contract(contract)
        self.assertIsNotNone(metadata, status)
        self.assertEqual(len(metadata["inputs"]), 1)

        # Change only downstream Quality roles, then reseal the already
        # technical binding and its outer command.  Technical Energy admission
        # must remain stable while the claim-capable verifier rejects it.
        mutated = json.loads(json.dumps(contract))
        binding = mutated["native_split_quality_binding"]
        binding["quality_completed"] = False
        binding["performance_claims_emitted"] = True
        binding.pop("binding_sha256", None)
        binding["binding_sha256"] = canonical_json_sha256(binding)
        mutated["native_split_quality_binding_sha256"] = binding[
            "binding_sha256"
        ]
        mutated.pop("contract_sha256", None)
        mutated["contract_sha256"] = canonical_json_sha256(mutated)
        verified, status = verify_native_energy_command_contract(mutated)
        self.assertIsNotNone(verified, status)
        metadata, status = verify_native_split_part2_input_contract(mutated)
        self.assertIsNotNone(metadata, status)
        claim_verified, claim_status = verify_native_command_contract(mutated)
        self.assertIsNone(claim_verified)
        self.assertEqual(
            claim_status,
            "native_command_contract_quality_binding_schema_or_role_invalid",
        )

        # Exercise the real Native Split preflight with the same claim-invalid
        # contract.  Localize every technical artifact, but deliberately leave
        # both semantic manifests absent: they are downstream annotations and
        # must neither be opened nor veto technical Energy admission.
        preflight = _load_script(
            "p03_native_split_energy_preflight",
            "scripts/native_split_energy_preflight.py",
        )

        def write_artifact(path: Path, content: bytes) -> dict[str, Any]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            localized = json.loads(json.dumps(mutated))
            binding = localized["native_split_quality_binding"]

            runner = write_artifact(
                root / "runner.py", b"#!/usr/bin/env python3\n",
            )
            input_image = write_artifact(
                root / "input.bin", b"p03-preflight-input",
            )
            localized["runner"] = runner["path"]
            localized["runner_sha256"] = runner["sha256"]
            localized["input_image"] = input_image["path"]
            localized["input_image_sha256"] = input_image["sha256"]

            semantic_artifacts = {
                "semantic_output_manifest",
                "semantic_boundary_manifest",
            }
            for name, artifact in localized["artifacts"].items():
                if name in semantic_artifacts:
                    artifact.update({
                        "path": str(
                            (root / f"missing-{name}.json").resolve()
                        ),
                        "sha256": "f" * 64,
                        "size_bytes": 17,
                    })
                    continue
                if name == "native_trt_meta":
                    continue
                artifact.update(write_artifact(
                    root / "artifacts" / name,
                    f"p03-{name}".encode("utf-8"),
                ))

            interpreter = localized["artifacts"]["python_executable"]
            localized["python_executable"] = interpreter["path"]
            localized["interpreter_identity"].update({
                "executable": interpreter["path"],
                "resolved_executable": interpreter["path"],
                "executable_sha256": interpreter["sha256"],
            })

            metadata = json.loads(json.dumps(
                binding["native_trt_meta_payload"]
            ))
            metadata.update({
                "onnx": localized["artifacts"]["build_part2_onnx"][
                    "path"
                ],
                "source_onnx": localized["artifacts"][
                    "build_part2_onnx"
                ]["path"],
                "engine": localized["artifacts"]["engine"]["path"],
                "engine_build_receipt_path": localized["artifacts"][
                    "engine_build_receipt"
                ]["path"],
            })
            metadata_bytes = json.dumps(
                metadata, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            localized["artifacts"]["native_trt_meta"].update(
                write_artifact(
                    root / "artifacts" / "native_trt_meta.json",
                    metadata_bytes,
                )
            )

            artifact_roles = {
                "part1_runtime": "hef",
                "boundary_metadata": "boundary_metadata",
                "source_part2_onnx": "source_part2_onnx",
                "build_part2_onnx": "build_part2_onnx",
                "engine": "engine",
                "native_trt_meta": "native_trt_meta",
                "engine_build_receipt": "engine_build_receipt",
                "trtexec": "trtexec",
            }
            technical_artifacts = {
                role: {
                    field: localized["artifacts"][name][field]
                    for field in ("path", "sha256", "size_bytes")
                }
                for role, name in artifact_roles.items()
            }
            binding["artifacts"] = json.loads(json.dumps(
                technical_artifacts
            ))
            proof = binding["local_artifact_verification"]
            proof["artifacts"] = json.loads(json.dumps(
                technical_artifacts
            ))
            proof["artifact_set_sha256"] = canonical_json_sha256(
                technical_artifacts
            )

            metadata_sha256 = canonical_json_sha256(metadata)
            metadata_file_sha256 = hashlib.sha256(
                metadata_bytes
            ).hexdigest()
            metadata_evidence = proof["embedded_json_files"][
                "native_trt_meta"
            ]
            metadata_evidence.update({
                "content_base64": base64.b64encode(
                    metadata_bytes
                ).decode("ascii"),
                "file_sha256": metadata_file_sha256,
                "file_size_bytes": len(metadata_bytes),
                "canonical_value_sha256": metadata_sha256,
            })
            proof["native_trt_meta_payload_sha256"] = metadata_sha256
            proof.pop("proof_sha256", None)
            proof["proof_sha256"] = canonical_json_sha256(proof)

            binding.update({
                "local_artifact_verification": proof,
                "native_trt_meta": json.loads(json.dumps(metadata)),
                "native_trt_meta_payload": json.loads(json.dumps(metadata)),
                "native_trt_meta_payload_sha256": metadata_sha256,
                "native_trt_meta_file_sha256": metadata_file_sha256,
                "native_trt_meta_file_size_bytes": len(metadata_bytes),
            })
            binding.pop("binding_sha256", None)
            binding["binding_sha256"] = canonical_json_sha256(binding)
            localized["native_split_quality_binding"] = binding
            localized["native_split_quality_binding_sha256"] = binding[
                "binding_sha256"
            ]
            localized["native_split_quality_local_verification"] = (
                json.loads(json.dumps(proof))
            )

            metadata_artifact = localized["artifacts"]["native_trt_meta"]
            engine_artifact = localized["artifacts"]["engine"]
            localized["boundary_contract"].update({
                "metadata_path": metadata_artifact["path"],
                "metadata_sha256": metadata_artifact["sha256"],
            })
            localized["engine"] = engine_artifact["path"]
            localized["engine_sha256"] = engine_artifact["sha256"]
            for duplicate, artifact_name in (
                ("hef", "hef"),
                ("native_executable", "native_executable"),
            ):
                localized[duplicate] = localized["artifacts"][
                    artifact_name
                ]["path"]
                localized[f"{duplicate}_sha256"] = localized["artifacts"][
                    artifact_name
                ]["sha256"]
            localized.pop("contract_sha256", None)
            localized["contract_sha256"] = canonical_json_sha256(localized)

            verified, status = verify_native_energy_command_contract(
                localized
            )
            self.assertIsNotNone(verified, status)
            metadata, status = verify_native_split_part2_input_contract(
                localized
            )
            self.assertIsNotNone(metadata, status)
            claim_verified, claim_status = verify_native_command_contract(
                localized
            )
            self.assertIsNone(claim_verified)
            self.assertEqual(
                claim_status,
                (
                    "native_command_contract_quality_binding_"
                    "schema_or_role_invalid"
                ),
            )

            preflight_script = ROOT / "scripts" / (
                "native_split_energy_preflight.py"
            )
            attestation, returncode = preflight.build_attestation(
                None,
                contract_payload=localized,
                nonce="p03-claim-invalid-technical-workload",
                expected_contract_sha256=localized["contract_sha256"],
                expected_preflight_script_sha256=hashlib.sha256(
                    preflight_script.read_bytes()
                ).hexdigest(),
                tool_root=root,
                valid_for_s=60.0,
            )
            self.assertEqual(returncode, 0, attestation)
            self.assertTrue(attestation["ok"], attestation)
            self.assertEqual(
                attestation["artifact_verification_status"], "pass",
            )
            self.assertEqual(
                attestation["semantic_payload_verification_status"],
                "downstream_annotation_not_preflighted",
            )
            self.assertEqual(attestation["verified_part2_input_count"], 1)
            self.assertTrue(
                attestation["workload_binding"]["workload_supported"]
            )
            for name in semantic_artifacts:
                self.assertFalse(Path(
                    localized["artifacts"][name]["path"]
                ).exists())
            self.assertFalse(any(
                str(row.get("label") or "").startswith(
                    "artifact:semantic_"
                )
                for row in attestation["artifact_verification"]
            ))

            # Resume consumes the admission decision frozen by this planner
            # version.  Annotation-only manifests must not become technical
            # restage prerequisites, while old archived rows without that
            # decision retain their legacy closure.
            plan_root = root / "plan"
            plan_root.mkdir()
            contract_path = plan_root / "localized.command_contract.json"
            contract_bytes = json.dumps(
                localized, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            contract_path.write_bytes(contract_bytes)
            resume_row = {
                "backend": localized["backend"],
                "model": localized["model"],
                "case": localized["case"],
                "setup_id": localized["setup_id"],
                "runtime_success": True,
                "energy_command_preflight_ok": True,
                "full_baseline": False,
                "split_has_valid_part2_input": True,
                "command_contract_file": str(contract_path),
                "command_contract_file_sha256": hashlib.sha256(
                    contract_bytes
                ).hexdigest(),
                "successful_command_contract_sha256": localized[
                    "contract_sha256"
                ],
                "native_energy_planner_admission": {
                    "schema": (
                        "onnx-splitpoint/"
                        "native-energy-planner-admission"
                    ),
                    "schema_version": 1,
                    "selected": True,
                    "runtime_success": True,
                    "energy_command_preflight_ok": True,
                    "full_baseline": False,
                    "split_has_valid_part2_input": True,
                },
            }
            frozen = resume_artifact_requirements(
                resume_row,
                plan_root=plan_root,
                frozen_remote_root=str(root.resolve()),
            )
            self.assertEqual(
                frozen["downstream_annotation_roles_omitted"],
                [
                    "semantic_boundary_manifest",
                    "semantic_output_manifest",
                ],
            )
            self.assertEqual(
                {requirement.role for requirement in frozen["requirements"]},
                {"prepared_input", "input_image"},
            )

            legacy_row = dict(resume_row)
            legacy_row.pop("native_energy_planner_admission")
            legacy = resume_artifact_requirements(
                legacy_row,
                plan_root=plan_root,
                frozen_remote_root=str(root.resolve()),
            )
            self.assertEqual(
                legacy["downstream_annotation_roles_omitted"], [],
            )
            self.assertEqual(
                {
                    requirement.role
                    for requirement in legacy["requirements"]
                },
                {
                    "prepared_input",
                    "semantic_boundary_manifest",
                    "semantic_output_manifest",
                    "input_image",
                },
            )

            # A P0.3 resume hash binds the executable measurement contract,
            # not downstream Quality/Semantic/pairing annotations.  Legacy
            # rows retain their wider historical comparison.
            resume_plan = {
                "schema": "onnx-splitpoint/native-energy-plan",
                "schema_version": 1,
                "energy_runs_per_row": 1,
            }
            p03_hash = ENERGY_RUNNER._resume_contract_sha256(
                resume_row, resume_plan,
            )
            annotation_drift = dict(resume_row)
            annotation_drift[
                "native_split_energy_quality_binding_sha256"
            ] = "b" * 64
            annotation_drift["source_request_sha256"] = "d" * 64
            self.assertEqual(
                ENERGY_RUNNER._resume_contract_sha256(
                    annotation_drift, resume_plan,
                ),
                p03_hash,
            )
            plan_annotation_drift = dict(resume_plan)
            plan_annotation_drift["pipeline_contract_sha256"] = "e" * 64
            plan_annotation_drift["model_hash_map_sha256"] = "f" * 64
            self.assertEqual(
                ENERGY_RUNNER._resume_contract_sha256(
                    resume_row, plan_annotation_drift,
                ),
                p03_hash,
            )
            technical_drift = dict(resume_row)
            technical_drift["command_contract_file_sha256"] = "c" * 64
            self.assertNotEqual(
                ENERGY_RUNNER._resume_contract_sha256(
                    technical_drift, resume_plan,
                ),
                p03_hash,
            )
            prepared_input_drift = dict(resume_row)
            prepared_input_drift[
                "prepared_feed_source_image_sha256"
            ] = "1" * 64
            self.assertNotEqual(
                ENERGY_RUNNER._resume_contract_sha256(
                    prepared_input_drift, resume_plan,
                ),
                p03_hash,
            )
            legacy_hash = ENERGY_RUNNER._resume_contract_sha256(
                legacy_row, resume_plan,
            )
            legacy_annotation_drift = dict(legacy_row)
            legacy_annotation_drift[
                "native_split_energy_quality_binding_sha256"
            ] = "b" * 64
            self.assertNotEqual(
                ENERGY_RUNNER._resume_contract_sha256(
                    legacy_annotation_drift, resume_plan,
                ),
                legacy_hash,
            )

        # A direct, unsealed metadata copy—even if it claims one input—is not
        # an authority once the portable artifact proof is absent.
        unbound = json.loads(json.dumps(contract))
        unbound["native_trt_meta_payload"] = _native_trt_meta()
        unbound.pop("native_split_quality_binding", None)
        unbound.pop("native_split_quality_binding_sha256", None)
        unbound.pop("contract_sha256", None)
        unbound["contract_sha256"] = canonical_json_sha256(unbound)
        metadata, status = verify_native_split_part2_input_contract(unbound)
        self.assertIsNone(metadata)
        self.assertEqual(status, "native_energy_part2_sealed_binding_missing")

    def test_each_technical_predicate_clause_is_enforced(self) -> None:
        def fail_one_runtime(rows: list[dict[str, Any]]) -> None:
            rows[0]["ok"] = False
            rows[0]["failure_reason"] = "fixture_runtime_failed"

        runtime_failed, _planned, _observed = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            observed_mutator=fail_one_runtime,
        )
        self.assertEqual(runtime_failed["energy_plan_included_count"], 62)
        self.assertEqual(runtime_failed["energy_plan_excluded_count"], 1)
        self.assertEqual(
            runtime_failed["excluded_rows"][0]["reason"],
            "native_performance_row_not_ok",
        )

        def runtime_builder(contract: Mapping[str, Any], **_kwargs: Any):
            if (
                contract.get("backend") == "deepx_to_trt"
                and contract.get("model") == "resnet50"
                and contract.get("case") == "b024"
            ):
                raise RuntimeError("fixture runtime argv failure")
            return ["python", "split-energy.py"]

        runtime_unconstructible, _planned, _observed = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            split_runtime_builder=runtime_builder,
        )
        self.assertEqual(
            runtime_unconstructible["energy_plan_included_count"], 62,
        )
        self.assertEqual(
            runtime_unconstructible["energy_plan_excluded_count"], 1,
        )
        self.assertEqual(
            runtime_unconstructible["excluded_rows"][0]["reason"],
            "verified_energy_command_contract_not_replayable",
        )

        def preflight_builder(
            contract: Mapping[str, Any], **_kwargs: Any,
        ):
            if (
                contract.get("backend") == "deepx_to_trt"
                and contract.get("model") == "resnet50"
                and contract.get("case") == "b024"
            ):
                raise RuntimeError("fixture preflight argv failure")
            return ["python", "split-preflight.py"]

        preflight_unconstructible, _planned, _observed = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            split_preflight_builder=preflight_builder,
        )
        self.assertEqual(
            preflight_unconstructible["energy_plan_included_count"], 62,
        )
        self.assertEqual(
            preflight_unconstructible["energy_plan_excluded_count"], 1,
        )
        self.assertEqual(
            preflight_unconstructible["excluded_rows"][0]["reason"],
            "energy_preflight_contract_not_replayable",
        )

    def test_partial_backend_failure_has_exact_45_row_ledger(self) -> None:
        """Reproduce the 18-present/27-missing production failure shape."""

        def partial_backend_fixture(
            theoretical: list[dict[str, Any]],
            observed: list[dict[str, Any]],
        ) -> tuple[
            list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
        ]:
            legacy_expected: list[dict[str, Any]] = []
            for raw in theoretical:
                row = dict(raw)
                if row.get("execution_mode") == "native_split":
                    row["backend"] = {
                        "deepx": "deepx_to_trt",
                        "hailo8": "hailo8_to_trt",
                        "hailo10h": "hailo10h_to_trt",
                    }[str(row["backend_key"])]
                    row.pop("setup_id", None)
                    row.pop("comparison_backend", None)
                legacy_expected.append(row)

            failed_rows = [
                dict(row)
                for row in observed
                if row.get("backend") in {
                    "hailo10h_to_trt", "deepx_to_trt",
                }
                and not str(row.get("backend") or "").startswith(
                    "native_full_"
                )
            ]
            self.assertEqual(len(failed_rows), 18)
            for row in failed_rows:
                row.update({
                    "ok": False,
                    "status": "partial_repetitions",
                    "failure_reason": "fixture_remote_runner_import_failed",
                    "error": "ModuleNotFoundError: fixture remote module",
                })

            failed_keys = {
                (
                    str(row.get("backend") or ""),
                    str(row.get("model") or ""),
                    str(row.get("case") or ""),
                )
                for row in failed_rows
            }
            present: list[dict[str, Any]] = []
            missing: list[dict[str, Any]] = []
            for expected in legacy_expected:
                key = (
                    {
                        "deepx": "deepx_to_trt",
                        "hailo8": "hailo8_to_trt",
                        "hailo10h": "hailo10h_to_trt",
                    }.get(
                        str(expected.get("backend") or ""),
                        str(expected.get("backend") or ""),
                    ),
                    str(expected.get("model") or ""),
                    str(expected.get("case") or ""),
                )
                if key in failed_keys:
                    failed = next(
                        row for row in failed_rows
                        if (
                            str(row.get("backend") or ""),
                            str(row.get("model") or ""),
                            str(row.get("case") or ""),
                        ) == key
                    )
                    present.append({
                        **expected,
                        "actual_ok": False,
                        "actual_status": "partial_repetitions",
                        "failure_reason": failed["failure_reason"],
                        "status_detail": failed["error"],
                        "error": failed["error"],
                    })
                else:
                    missing.append({
                        **expected,
                        "ok": False,
                        "result_ok": False,
                        "status": "missing_expected_native_row",
                        "runtime_status": "missing",
                        "failure_reason": "native_transfer_failed",
                        "status_detail": "fixture backend/full transfer failed",
                        "error": "fixture backend/full transfer failed",
                    })

            self.assertEqual(len(present), 18)
            self.assertEqual(len(missing), 27)
            matrix = {
                "schema": "onnx-splitpoint/native-expected-matrix",
                "schema_version": 2,
                "expected_row_count": 45,
                "present_expected_row_count": 18,
                "successful_expected_row_count": 0,
                "failed_expected_row_count": 18,
                "missing_expected_row_count": 27,
                "row_presence_complete": False,
                "execution_success_complete": False,
                "matrix_complete": False,
                "present_expected_rows": present,
                "successful_expected_rows": [],
                "failed_expected_rows": list(present),
                "missing_expected_rows": missing,
            }
            return legacy_expected, failed_rows, matrix

        payload, _planned, _observed = self._run_plan(
            LEGACY_CASES,
            mixed_annotations=False,
            require_matrix_presence_complete=False,
            fixture_builder=partial_backend_fixture,
        )
        self.assertEqual(payload["energy_matrix_expected_count"], 45)
        self.assertEqual(payload["energy_plan_included_count"], 0)
        self.assertEqual(payload["energy_plan_excluded_count"], 45)
        self.assertEqual(len(payload["excluded_rows"]), 45)
        self.assertEqual(
            payload["energy_plan_invalid_membership_identity_count"], 0,
        )
        self.assertEqual(
            payload["energy_plan_ledger_unexpected_identities"], [],
        )
        self.assertEqual(
            payload["energy_plan_ledger_missing_identities"], [],
        )
        self.assertEqual(
            payload["energy_plan_ledger_overlap_identities"], [],
        )
        identities = [_membership(row) for row in payload["excluded_rows"]]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(payload["preflight"]["planned_rows"], 0)
        self.assertFalse(payload["preflight"]["measurement_start_allowed"])

    def test_duplicate_runtime_identity_blocks_measurement_start(self) -> None:
        def append_duplicate(rows: list[dict[str, Any]]) -> None:
            rows.append(json.loads(json.dumps(rows[0])))

        payload, _planned, _observed = self._run_plan(
            CURRENT_CASES,
            mixed_annotations=False,
            observed_mutator=append_duplicate,
            require_matrix_presence_complete=False,
        )
        self.assertGreater(payload["deduplicated_count"], 0)
        self.assertFalse(payload["energy_plan_coverage_contract_valid"])
        self.assertFalse(payload["preflight"]["measurement_start_allowed"])
        self.assertEqual(
            payload["preflight_status"],
            "blocked_technical_measurement_contract_invalid",
        )

    def test_generic_energy_remains_empty_and_measurement_does_not_promote(self) -> None:
        from onnx_splitpoint_tool.energy.config import (
            resolve_effective_energy_config,
        )
        from onnx_splitpoint_tool.execution_plan import (
            build_effective_execution_plan,
        )
        from onnx_splitpoint_tool.run_modes import (
            apply_run_mode,
            default_run_modes_config,
        )
        from onnx_splitpoint_tool.workflow.execution_binding import (
            _energy_enabled_for_profile,
        )

        config = default_run_modes_config()
        profile = {
            "name": "p03-native-only-energy",
            "model_suite": {"primary": [{
                "id": "resnet50", "task": "classification",
                "enabled": True,
            }]},
            "run_profiles": [{
                "id": "hailo8_to_trt",
                "stage1": "hailo8",
                "stage2": "tensorrt",
            }],
            "execution_preset": {
                "id": "final",
                "follow_tool_config": False,
                "snapshot": config["modes"]["final"],
                "overrides": {
                    "native_enabled": True,
                    "energy_enabled": True,
                },
            },
        }
        resolved, _audit = apply_run_mode(profile, config=config)
        self.assertFalse(resolved["energy"]["enabled"])
        self.assertFalse(resolved["energy"]["generic_enabled"])
        self.assertEqual(
            resolved["energy"]["measurement_path"], "native_only",
        )
        plan = build_effective_execution_plan(resolved)
        self.assertFalse(plan["generic_energy_enabled"])
        self.assertTrue(plan["native_energy_enabled"])
        effective = resolve_effective_energy_config(resolved)
        self.assertFalse(effective["generic_energy_enabled"])
        self.assertTrue(effective["native_energy_enabled"])
        self.assertEqual(effective["measurement_path"], "native_only")
        self.assertFalse(
            _energy_enabled_for_profile(mock.Mock(), resolved)
        )

        payload, _theoretical, _observed = self._run_plan(
            LEGACY_CASES, mixed_annotations=True,
        )
        real_plan_row = payload["rows"][0]
        validation = ENERGY_RUNNER._prepare_measurement_execution(
            real_plan_row,
            payload,
            allowed_root=Path(
                real_plan_row["measurement_output_base_dir"]
            ).parent,
            validate_only=True,
        )
        self.assertTrue(validation["validated"])
        claim_fields = {
            "claim_ok", "semantic_claim_ok", "claim_eligible",
            "energy_claim_eligible",
            "eligible_for_energy_results_import",
            "eligible_for_scientific_claim",
        }
        promoted = {field: True for field in claim_fields}
        result = {
            **promoted,
            "row": {**real_plan_row, **promoted},
            "run": {
                **promoted,
                "energy_aggregate": {**promoted},
            },
            "energy_aggregate": {**promoted},
        }
        clamped = ENERGY_RUNNER._clamp_screening_result(result)
        layers = (
            clamped,
            clamped["row"],
            clamped["run"],
            clamped["run"]["energy_aggregate"],
            clamped["energy_aggregate"],
        )
        for layer in layers:
            for field in claim_fields:
                self.assertIs(layer[field], False, (field, layer))


if __name__ == "__main__":
    unittest.main()
