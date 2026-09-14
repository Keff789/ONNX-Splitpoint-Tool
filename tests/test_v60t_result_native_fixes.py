from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path

from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.cross_runner_reporting import compute_cross_runner_report
from onnx_splitpoint_tool.workflow.execution_binding import _copy_remote_result_files


def _profile(*, energy: bool) -> dict:
    cfg = default_run_modes_config()
    return {
        "name": "v60t-test",
        "model_suite": {"primary": [{"id": "m", "task": "detection", "evaluation_role": "development"}]},
        "selection_policy": {"max_accepted_cases_per_model": 3, "selection_strategy": "stratified_windows"},
        "run_profiles": [{"id": "deepx_m1_to_tensorrt", "stage1": "deepx_m1", "stage2": "tensorrt"}],
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": False,
            "snapshot": cfg["modes"]["standard"],
            "overrides": {"native_enabled": True, "energy_enabled": energy},
        },
    }


def test_energy_toggle_means_native_measurement_not_plan() -> None:
    enabled, _ = apply_run_mode(_profile(energy=True))
    assert enabled["energy"]["enabled"] is False
    assert enabled["energy"]["generic_enabled"] is False
    assert enabled["native_producers"]["energy"]["enabled"] is True
    assert enabled["native_producers"]["energy"]["mode"] == "measure"
    disabled, _ = apply_run_mode(_profile(energy=False))
    assert disabled["native_producers"]["energy"]["enabled"] is False
    assert disabled["native_producers"]["energy"]["mode"] == "plan"


def test_large_canonical_result_is_not_limited_by_diagnostic_cap(monkeypatch) -> None:
    monkeypatch.delenv("ONNX_SPLITPOINT_CANONICAL_RESULT_MAX_BYTES", raising=False)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        remote = root / "remote"; results = remote / "results"; results.mkdir(parents=True)
        payload = [{"id": "row", "blob": "x" * (3 * 1024 * 1024)}]
        src = results / "benchmark_results_large.json"
        src.write_text(json.dumps(payload), encoding="utf-8")
        logs = remote / "logs"; logs.mkdir(); (logs / "huge.log").write_text("z" * (3 * 1024 * 1024), encoding="utf-8")
        dst = root / "dst"
        copied = _copy_remote_result_files(remote, dst, flat_prefix="target")
        assert (dst / src.name).is_file()
        assert (dst / src.name).stat().st_size > 2 * 1024 * 1024
        manifest = json.loads((dst / "remote_diagnostics" / "target" / "result_copy_manifest.json").read_text())
        assert manifest["canonical_result_count"] >= 1
        assert manifest["status"] in {"ok", "partial"}
        assert not (dst / "remote_diagnostics" / "target" / "logs" / "huge.log").exists()
        assert any(row.get("canonical") for row in copied)


def test_canonical_result_falls_back_from_lean_bundle() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        remote = root / "remote"; remote.mkdir()
        bundle = remote / "results_bundle_lean.tar.gz"
        data = json.dumps([{"id": "fallback", "fps": 42.0}]).encode()
        info = tarfile.TarInfo("nested/benchmark_results_fallback.json"); info.size = len(data)
        with tarfile.open(bundle, "w:gz") as tf:
            tf.addfile(info, io.BytesIO(data))
        dst = root / "dst"
        _copy_remote_result_files(remote, dst, flat_prefix="target")
        recovered = dst / "benchmark_results_fallback.json"
        assert recovered.is_file()
        assert json.loads(recovered.read_text())[0]["id"] == "fallback"
        manifest = json.loads((dst / "remote_diagnostics" / "target" / "result_copy_manifest.json").read_text())
        assert manifest["fallback_extraction_count"] == 1


def test_cross_runner_report_computes_rank_transfer() -> None:
    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        reports = run / "reports"; (reports / "native_validation").mkdir(parents=True)
        native_rows = []
        validation_rows = []
        generic_rows = []
        endpoint_hash = "a" * 64
        endpoint = {
            "task": "classification", "stage": "logits",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": endpoint_hash,
            "output_endpoint_attestation": {
                "attested": True, "status": "passed",
                "endpoint_contract_hash": endpoint_hash,
            },
        }
        for idx, (g, n) in enumerate(((10.0, 5.0), (20.0, 8.0), (30.0, 12.0)), start=1):
            case = f"b{idx:03d}"
            generic_rows.append({
                "model_id": "m", "case_id": case, "direction": "deepx_m1_to_tensorrt",
                "backend": "deepx_m1_to_tensorrt", "variant": "split", "runner_regime": "generic",
                "precision": "fp16", "setup_id": "deepx-host", "comparison_backend": "deepx",
                "cycle_ms": g, "task_quality_status": "pass", "contract_consistent": True,
                "eligible_for_ranking": True, **endpoint,
            })
            native_rows.append({
                "model": "m", "case": case, "backend": "deepx_to_trt", "ok": True,
                "fps_makespan": 1000.0 / n, "precision": "fp16", "status": "ok",
                "setup_id": "deepx-host", "comparison_backend": "deepx",
                "performance_claim_eligible": True, "output_endpoint_match": True,
                "precision_quality_verified": True, "quality_evidence_verified": True,
                "repeat_claim_gate_pass": True, "comparison_stratum_explicit": True,
                **endpoint,
            })
            validation_rows.append({
                "model": "m", "case": case, "backend": "deepx_to_trt",
                "precision": "fp16", "setup_id": "deepx-host", "comparison_backend": "deepx",
                "contract_consistent": True, "semantic_ok": True, "claim_ok": True,
                "task_valid": True, "accuracy_gate_pass": True, "eligible_for_ranking": True,
                "status": "claim_ok", "gate_status": "eligible", **endpoint,
            })
        (reports / "native_producer_combined_summary.json").write_text(json.dumps({"rows": native_rows}), encoding="utf-8")
        (reports / "native_validation" / "native_producer_validation_summary.json").write_text(json.dumps({"rows": validation_rows}), encoding="utf-8")
        result = compute_cross_runner_report(run, generic_rows, minimum_candidates=3)
        assert result["status"] == "ok"
        assert result["eligible_pair_count"] == 3
        group = result["groups"][0]
        assert group["spearman_rho"] == 1.0
        assert group["kendall_tau_b"] == 1.0
        assert group["native_best_hit_at_3"] is True
        assert group["native_regret_at_3"] == 0.0
        assert group["technical_native_best_hit_at_1"] is True
        assert group["technical_native_regret_at_1"] == 0.0
        assert group["quality_native_best_hit_at_1"] is True
        assert group["quality_native_regret_at_1"] == 0.0
        assert group["technical_pairwise_concordance"] == 1.0
        assert group["quality_pairwise_concordance"] == 1.0
        assert group["native_best_hit_at_5"] is None
        assert group["native_regret_at_5"] is None

        generic_rows[0].update({
            "measurement_valid": False,
            "terminal_failure": True,
            "runtime_executable": False,
        })
        invalid = compute_cross_runner_report(
            run, generic_rows, minimum_candidates=3,
        )
        assert invalid["technical_pair_count"] == 2
        invalid_group = invalid["groups"][0]
        assert invalid_group["technical_candidate_count"] == 2
        assert invalid_group["technical_pairwise_concordance"] is None
        assert invalid_group["technical_native_best_hit_at_1"] is None
        assert invalid_group["technical_native_regret_at_1"] is None


def test_normal_benchmarkset_path_registers_artifacts() -> None:
    source = Path("onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py").read_text(encoding="utf-8")
    assert "register_benchmark_set_artifacts(" in source
    assert "post-generation registration" in source


def test_native_full_runner_and_final_report_preserve_failure_diagnostics() -> None:
    runner = Path("scripts/native_full_baseline_eval_runner.py").read_text(encoding="utf-8")
    final = Path("scripts/native_producer_final_report.py").read_text(encoding="utf-8")
    for token in ("failure_reason", "status_detail", "timed_out", "stdout_tail", "stderr_tail"):
        assert token in runner
        assert token in final
