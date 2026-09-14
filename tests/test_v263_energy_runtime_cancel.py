from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from onnx_splitpoint_tool.benchmark.remote_run import (
    RemoteBenchmarkArgs,
    _bind_energy_ab_runtime_args,
)
from onnx_splitpoint_tool.energy.collector import (
    _energy_ab_runtime_contract,
    _power_command,
    _run_legacy_window_comparison,
    _run_one,
    run_duration_probe,
)
from onnx_splitpoint_tool.energy.config import (
    EnergyDefaults,
    energy_ab_cli_args,
)
from onnx_splitpoint_tool.native_energy_reporting import collect_native_energy

import yaml


ROOT = Path(__file__).resolve().parents[1]


class EnergyRuntimeContractV263Tests(unittest.TestCase):
    def _ab_defaults(self) -> EnergyDefaults:
        defaults = EnergyDefaults()
        defaults.window_ab_enabled = True
        defaults.window_ab_primary_method = "command_marker_window"
        defaults.window_ab_shadow_method = "chapter4_legacy_window"
        defaults.window_ab_baseline_method = "chapter4_baseline"
        defaults.window_ab_candidate_method = "candidate_v263"
        defaults.window_ab_same_raw_capture = True
        defaults.window_ab_mode = "shadow"
        defaults.window_ab_auto_switch = False
        defaults.window_ab_smoke_repeats = 3
        defaults.window_ab_requires_picoscope = False
        return defaults

    def test_ab_contract_enforces_three_repeats_and_never_auto_switches(self) -> None:
        contract = _energy_ab_runtime_contract(
            self._ab_defaults(),
            requested_run_count=1,
            compare_legacy_window=True,
        )
        self.assertTrue(contract["enabled"])
        self.assertTrue(contract["valid"])
        self.assertEqual(contract["effective_run_count"], 3)
        self.assertEqual(contract["scientific_primary_method"], "command_marker_window")
        self.assertEqual(contract["scientific_shadow_method"], "chapter4_legacy_window")
        self.assertEqual(contract["candidate_method"], "candidate_v263")
        self.assertEqual(contract["candidate_role"], "deprecated_alias_for_command_marker_primary")
        self.assertFalse(contract["auto_switch"])
        self.assertFalse(contract["candidate_eligible_for_auto_switch"])

    def test_caller_managed_probe_keeps_exact_single_capture(self) -> None:
        contract = _energy_ab_runtime_contract(
            self._ab_defaults(),
            requested_run_count=1,
            compare_legacy_window=True,
            exact_run_count=True,
        )
        self.assertTrue(contract["enabled"])
        self.assertEqual(contract["requested_run_count"], 1)
        self.assertEqual(contract["effective_run_count"], 1)
        self.assertEqual(contract["repeat_control"], "caller_managed_exact")
        self.assertTrue(contract["exact_run_count_requested"])
        self.assertFalse(contract["minimum_repeat_expansion_applied"])

    def test_remote_args_bind_exact_ab_contract_to_collector_defaults(self) -> None:
        args = RemoteBenchmarkArgs(
            energy_window_ab_enabled=True,
            energy_window_ab_smoke_repeats=3,
        )
        defaults = _bind_energy_ab_runtime_args(EnergyDefaults(), args)
        self.assertTrue(defaults.window_ab_enabled)
        self.assertTrue(defaults.compare_legacy_window)
        self.assertEqual(defaults.window_ab_primary_method, "command_marker_window")
        self.assertEqual(defaults.window_ab_shadow_method, "chapter4_legacy_window")
        self.assertEqual(defaults.window_ab_baseline_method, "chapter4_baseline")
        self.assertEqual(defaults.window_ab_candidate_method, "candidate_v263")
        self.assertEqual(defaults.window_ab_mode, "shadow")
        self.assertFalse(defaults.window_ab_auto_switch)
        self.assertTrue(defaults.keep_raw_parquet)

    def test_final_profile_native_path_reaches_plan_cli_and_collector(self) -> None:
        profile = yaml.safe_load(
            (ROOT / "onnx_splitpoint_tool" / "resources" / "evaluation_profiles"
             / "thesis_final_campaign_v1.yaml").read_text(encoding="utf-8")
        )
        cli_args = energy_ab_cli_args(profile["energy"]["window_method_ab"])
        self.assertEqual(cli_args[0], "--window-method-ab-json")
        propagated = json.loads(cli_args[1])
        self.assertEqual(propagated["scientific_primary_method"], "command_marker_window")
        self.assertEqual(propagated["scientific_shadow_method"], "chapter4_legacy_window")
        self.assertEqual(propagated["candidate_method"], "candidate_v263")
        self.assertFalse(propagated["candidate_eligible_for_auto_switch"])
        repeats = int(profile["energy"]["repeats"])
        self.assertEqual(repeats, 5)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.json"
            validation = root / "validation.json"
            out = root / "plan"
            summary.write_text(json.dumps({"rows": [
                {
                    "ok": True, "backend": "hailo8_to_trt", "model": "m",
                    "case": "b1", "precision": "p", "setup_id": "h8",
                    "fps_makespan": 10,
                },
                {
                    "ok": True, "backend": "native_full_hailo8", "model": "m",
                    "case": "full", "precision": "p", "setup_id": "h8",
                    "comparison_backend": "hailo8", "fps_makespan": 8,
                },
                {
                    "ok": True, "backend": "native_full_tensorrt", "model": "m",
                    "case": "full", "precision": "p", "setup_id": "h8",
                    "comparison_backend": "hailo8", "fps_makespan": 12,
                },
            ]}), encoding="utf-8")
            validation.write_text(json.dumps({"rows": [{
                "backend": "hailo8_to_trt", "model": "m", "case": "b1",
                "precision": "p", "task": "classification", "top1_match": True,
                "contract_consistent": True, "claim_ok": True, "semantic_ok": True,
            }]}), encoding="utf-8")
            command = [
                sys.executable, str(ROOT / "scripts" / "native_producer_energy_plan.py"),
                "--summary", str(summary), "--validation-summary", str(validation),
                "--out-dir", str(out), "--hailo8-ssh", "host",
                "--runs", str(repeats), *cli_args,
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads(
                (out / "native_producer_energy_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["energy_runs_per_row"], 5)
            self.assertEqual(plan["window_method_ab"]["scientific_primary_method"], "command_marker_window")
            self.assertEqual(plan["window_method_ab"]["scientific_shadow_method"], "chapter4_legacy_window")
            self.assertEqual(plan["window_method_ab"]["candidate_method"], "candidate_v263")
            # The synthetic legacy rows intentionally have no sealed Native
            # command contract.  Since 2.62.1 this must fail closed instead of
            # reconstructing a workload from defaults.  The frozen A/B
            # contract is nevertheless propagated by the plan CLI and can be
            # bound to the collector independently of row admission.
            self.assertFalse(plan["rows"])
            self.assertTrue(plan["excluded_rows"])
            self.assertTrue(all(
                row["reason"] == "successful_command_contract_missing_or_invalid"
                for row in plan["excluded_rows"]
            ))
            cli_payload = plan["window_method_ab"]

        collector_defaults = EnergyDefaults()
        native_cli = runpy.run_path(str(ROOT / "scripts" / "energy_measurement_cli.py"))
        cli_resolved = native_cli["_bind_window_method_ab"](
            collector_defaults, json.dumps(cli_payload)
        )
        self.assertEqual(cli_resolved["scientific_primary_method"], "command_marker_window")
        self.assertEqual(cli_resolved["scientific_shadow_method"], "chapter4_legacy_window")
        runtime = _energy_ab_runtime_contract(
            collector_defaults,
            requested_run_count=1,
            compare_legacy_window=collector_defaults.compare_legacy_window,
        )
        self.assertTrue(runtime["valid"])
        self.assertEqual(runtime["scientific_primary_method"], "command_marker_window")
        self.assertEqual(runtime["scientific_shadow_method"], "chapter4_legacy_window")
        self.assertEqual(runtime["candidate_method"], "candidate_v263")
        self.assertEqual(runtime["candidate_role"], "deprecated_alias_for_command_marker_primary")
        self.assertFalse(runtime["candidate_eligible_for_auto_switch"])

    def test_ab_comparison_names_marker_primary_and_uses_same_trace(self) -> None:
        defaults = self._ab_defaults()
        defaults.power_calculations_binary = "/bin/true"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run_000"
            storage = run_dir / "collector_storage"
            storage.mkdir(parents=True)
            trace = storage / "capture.parquet"
            trace.write_bytes(b"same-raw-capture")
            trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
            request = run_dir / "command_window_request.json"
            request.write_text(
                json.dumps(
                    {
                        "trace_path": str(trace),
                        "trace_sha256": trace_sha,
                        "first_sample_index": 10,
                        "last_sample_index": 30,
                        "sample_rate_hz": 2000,
                    }
                ),
                encoding="utf-8",
            )
            primary_result = run_dir / "candidate_results.yaml"
            primary_result.write_text("candidate: true\n", encoding="utf-8")
            legacy_result = run_dir / "legacy_results.yaml"
            legacy_result.write_text("legacy: true\n", encoding="utf-8")
            primary_cmd = _power_command(defaults, storage, run_dir / "processed", 5)

            def summary(payload):
                if payload.get("kind") == "baseline":
                    return {
                        "energy_total_j": 10.0,
                        "active_duration_s": 2.0,
                        "avg_power_w": 5.0,
                        "start_stop_idx": [12, 32],
                        "source_key": "firmware_results",
                    }
                return {
                    "energy_total_j": 9.5,
                    "active_duration_s": 1.9,
                    "avg_power_w": 5.0,
                    "start_stop_idx": [10, 30],
                    "source_key": "firmware_results",
                }

            with mock.patch(
                "onnx_splitpoint_tool.energy.collector._run_powercalc_limited",
                return_value={"rc": 0},
            ), mock.patch(
                "onnx_splitpoint_tool.energy.collector._find_results_yaml",
                return_value=legacy_result,
            ), mock.patch(
                "onnx_splitpoint_tool.energy.collector._read_yaml",
                return_value={"kind": "baseline"},
            ), mock.patch(
                "onnx_splitpoint_tool.energy.collector.extract_power_calculation_summary",
                side_effect=summary,
            ):
                result = _run_legacy_window_comparison(
                    run_dir,
                    defaults=defaults,
                    storage_dir=storage,
                    estimated_duration_s=5,
                    request_path=request,
                    primary_result_path=primary_result,
                    primary_result_data={"kind": "candidate"},
                    primary_command_argv=primary_cmd,
                )

            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["same_raw_trace_verified"])
            self.assertEqual(result["scientific_primary_method"], "command_marker_window")
            self.assertEqual(result["scientific_shadow_method"], "chapter4_legacy_window")
            self.assertEqual(result["command_marker_window"]["scientific_role"], "primary")
            self.assertTrue(result["command_marker_window"]["eligible_for_scientific_primary"])
            self.assertEqual(result["command_marker_window"]["energy_j"], 9.5)
            self.assertEqual(result["chapter4_legacy_window"]["scientific_role"], "shadow_only")
            self.assertEqual(result["chapter4_legacy_window"]["energy_j"], 10.0)
            self.assertEqual(result["candidate_v263"]["canonical_method"], "command_marker_window")
            self.assertFalse(result["candidate_v263"]["eligible_for_auto_switch"])

    def test_native_scientific_reporting_uses_chapter4_not_shadow_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            result_dir = run_dir / "reports" / "native_energy_measurements"
            result_dir.mkdir(parents=True)
            aggregate = {
                "scientific_primary_method_frozen": True,
                "scientific_primary_method": "chapter4_baseline",
                "scientific_primary_energy_status": "available",
                "scientific_primary_claim_eligible": True,
                "scientific_primary_energy_total_j": 10.0,
                "scientific_primary_energy_per_work_unit_j": 0.1,
                "scientific_primary_active_duration_s": 2.0,
                "scientific_primary_avg_power_w": 5.0,
                "candidate_method": "candidate_v263",
                "candidate_role": "shadow_only",
                "candidate_eligible_for_auto_switch": False,
                "candidate_v263_shadow_energy_total_j": 9.5,
                "candidate_v263_shadow_energy_per_work_unit_j": 0.095,
                "avg_energy_total_j": 9.5,
                "avg_energy_per_work_unit_j": 0.095,
                "avg_energy_work_units_used": 100,
                "energy_work_units_source": "runtime_completed_work_units",
                "runtime_completed_work_unit_run_count": 3,
                "valid_postprocessed_runs": 3,
                "energy_efficiency_claim_eligible": True,
                "final_energy_gate_status": "pass",
                "postprocess_status": "ok",
                "energy_window_effective_values": ["command_window"],
                "energy_window_requested": "command",
                "energy_physical_scope": "MB",
                "energy_primary_metric": "calibrated_input_energy_unsubtracted",
                "energy_calibrated_input_unsubtracted": True,
            }
            report = {
                "rows": [{
                    "ok": True,
                    "row": {
                        "backend": "hailo8_to_trt", "model": "m", "case": "b1",
                        "precision": "p", "setup_id": "h8", "claim_ok": True,
                        "contract_consistent": True,
                        "task": "classification",
                        "prepared_feed_task": "classification",
                        "prepared_feed_preprocess_mode": "resize",
                        "prepared_feed_letterbox_pad_value": 0,
                        "prepared_feed_source_image_sha256": "a" * 64,
                    },
                    "run": {"rc": 0, "stdout": json.dumps(aggregate)},
                }]
            }
            (result_dir / "native_producer_energy_results.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            rows = collect_native_energy(run_dir)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["scientific_primary_method"], "chapter4_baseline")
        self.assertEqual(row["energy_total_j"], 10.0)
        self.assertEqual(row["energy_per_work_j"], 0.1)
        self.assertEqual(row["candidate_v263_shadow_energy_total_j"], 9.5)
        self.assertEqual(row["candidate_v263_shadow_energy_per_work_j"], 0.095)
        self.assertFalse(row["candidate_eligible_for_auto_switch"])
        self.assertTrue(row["claim_eligible"], row["claim_exclusion_reasons"])


class EnergyCancellationV263Tests(unittest.TestCase):
    def test_run_one_cancels_process_group_promptly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event = threading.Event()
            timer = threading.Timer(0.25, event.set)
            timer.start()
            started = time.monotonic()
            try:
                result = _run_one(
                    ["/bin/bash", "-lc", "sleep 30 & wait"],
                    cwd=None,
                    stdout_path=root / "stdout.log",
                    stderr_path=root / "stderr.log",
                    heartbeat_path=root / "status.json",
                    heartbeat_interval_s=0.05,
                    cancel_event=event,
                    timeout=60,
                )
            finally:
                timer.cancel()
            self.assertTrue(result["cancelled"])
            self.assertEqual(result["rc"], 130)
            self.assertLess(time.monotonic() - started, 4.0)
            status = json.loads((root / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "cancelled")

    def test_duration_probe_propagates_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event = threading.Event()
            timer = threading.Timer(0.2, event.set)
            timer.start()
            try:
                result = run_duration_probe(
                    "sleep 30",
                    temporary,
                    timeout_s=60,
                    cancel_event=event,
                )
            finally:
                timer.cancel()
            self.assertFalse(result["ok"])
            self.assertTrue(result["cancelled"])
            self.assertEqual(result["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
