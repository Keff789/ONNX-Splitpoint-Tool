from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool import __version__
from onnx_splitpoint_tool.benchmark import services as benchmark_services
from onnx_splitpoint_tool.workflow.contracts import StageResult
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    WORKFLOW_VERSION,
    benchmark_set_postcondition_v60v,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"v60v_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_suite(base: Path) -> Path:
    suite = base / "legacy_suite"
    case = suite / "b001"
    case.mkdir(parents=True)
    (suite / "benchmark_suite.py").write_text("print('ok')\n", encoding="utf-8")
    (suite / "benchmark_set.json").write_text(json.dumps({"cases": [{"case_id": "b001"}]}), encoding="utf-8")
    (suite / "benchmark_plan.json").write_text(json.dumps({"runs": [{"id": "cpu_ort", "case_id": "b001"}]}), encoding="utf-8")
    return suite


def test_versions_and_mutable_mapping_import() -> None:
    assert __version__ in {"0.14.21+v60v.generationnativefix", "0.14.22+v60w.smokedeferralpackfix", "0.14.23+v60x.nativeevidencefix", "0.14.25+v60z.nativefullquality", "0.14.26+v61a.nativefullenergyprogress", "0.14.27+v61b.nativeintegrationfix", "0.14.28+v61c.nativefullpairedenergyfix", "0.14.29+v61d.nativefullsemanticfix", "0.14.30+v61e.standardguifix", "2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47", "2.79.19", "2.79.20"}
    assert WORKFLOW_VERSION in {"v60v-generation-native-runtime-fixes", "v60w-smoke-deferral-pack-selection-fixes", "v60x-native-evidence-contract-fixes", "v60z-native-full-energy-quality-evidence", "v61a-native-full-energy-progress-fixes", "v61b-native-integration-live-energy-fixes", "v61c-native-full-paired-energy-fixes", "v61d-native-full-semantic-hailo8-fixes", "v61e-standard-run-gui-diagnostics-fixes", "v2.61e-campaign-contract-hardening", "v2.62-window-validation-native-binding", "v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent", "v2.79.19-calibration-warning-evalrun-closure", "v2.79.20-artifact-reuse-closure"}
    # Regression for the fatal v60u NameError in benchmark/services.py.
    assert benchmark_services.MutableMapping is not None


def test_benchmark_set_postcondition_accepts_atomic_suite(tmp_path: Path) -> None:
    suite = _valid_suite(tmp_path / "benchmark_set")
    report = benchmark_set_postcondition_v60v(tmp_path / "benchmark_set")
    assert report["valid"] is True
    assert Path(report["selected_suite_dir"]) == suite
    assert report["case_count"] == 1
    assert report["run_count"] == 1


def test_benchmark_set_postcondition_accepts_declared_generated_suite(
    tmp_path: Path,
) -> None:
    base = tmp_path / "benchmark_set"
    suite = base / "generated_suite"
    case = suite / "b052"
    case.mkdir(parents=True)
    (suite / "benchmark_suite.py").write_text(
        "print('ok')\n", encoding="utf-8",
    )
    (suite / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"case_id": "b052"}]}),
        encoding="utf-8",
    )
    (suite / "benchmark_plan.json").write_text(
        json.dumps({"runs": [{"id": "hailo8_to_trt"}]}),
        encoding="utf-8",
    )

    report = benchmark_set_postcondition_v60v(base)

    assert report["valid"] is True
    assert Path(report["selected_suite_dir"]) == suite




def test_benchmark_set_postcondition_accepts_legacy_case_dir_contract(tmp_path: Path) -> None:
    suite = tmp_path / "benchmark_set" / "legacy_suite"
    (suite / "b052").mkdir(parents=True)
    (suite / "benchmark_suite.py").write_text("print('ok')\n", encoding="utf-8")
    (suite / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"boundary": 52, "case_dir": "b052", "folder": "b052"}]}),
        encoding="utf-8",
    )
    (suite / "benchmark_plan.json").write_text(
        json.dumps({"runs": [{"id": "hailo8_to_trt"}]}), encoding="utf-8"
    )
    report = benchmark_set_postcondition_v60v(tmp_path / "benchmark_set")
    assert report["valid"] is True
    assert report["case_count"] == 1
    assert report["candidates"][0]["case_ids"] == ["b052"]


def test_benchmark_set_postcondition_rejects_empty_cases(tmp_path: Path) -> None:
    suite = tmp_path / "benchmark_set" / "legacy_suite"
    suite.mkdir(parents=True)
    (suite / "benchmark_suite.py").write_text("print('bad')\n", encoding="utf-8")
    (suite / "benchmark_set.json").write_text('{"cases": []}', encoding="utf-8")
    (suite / "benchmark_plan.json").write_text('{"runs": [{"id": "cpu"}]}', encoding="utf-8")
    report = benchmark_set_postcondition_v60v(tmp_path / "benchmark_set")
    assert report["valid"] is False
    assert "benchmark_set_has_no_cases" in report["reasons"]


def test_model_pipeline_skips_dependents_after_generation_failure(tmp_path: Path) -> None:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner._stop_requested = False
    runner.run_dir = tmp_path / "run"
    runner.manifest = {}
    runner.jobs = None
    runner.options = SimpleNamespace(skip_analysis=False, skip_benchmarks=False, no_remote=False)
    called: list[tuple[str, bool, str]] = []

    def fake_run_stage(model_id, stage, fn):
        if stage == "generate_benchmark_set":
            result = StageResult(stage=stage, model_id=model_id, status="failed", started_at="x", finished_at="x", error_detail="boom")
            called.append((stage, False, result.status))
            return result
        payload = fn()
        status = str(payload[3])
        called.append((stage, True, status))
        return StageResult(stage=stage, model_id=model_id, status=status, started_at="x", finished_at="x", skip_reason=str((payload[1] or {}).get("skip_reason") or ""))

    runner._run_stage = fake_run_stage
    # Avoid executing real stage implementations before generation.
    for name in [
        "_stage_resolve_model", "_stage_check_validation_assets", "_stage_prepare_model",
        "_stage_analyze_model", "_stage_select_split_candidates", "_stage_prepare_full_baselines",
        "_stage_generate_benchmark_set", "_stage_build_backend_artifacts", "_stage_run_benchmarks",
        "_stage_validate_outputs", "_stage_hardware_smoke",
    ]:
        setattr(runner, name, lambda *a, **k: ({}, {}, "ok", "ok"))

    EvaluationWorkflowRunner._run_model(runner, {"id": "m"})
    statuses = {stage: status for stage, _executed, status in called}
    assert statuses["generate_benchmark_set"] == "failed"
    for stage in ("build_backend_artifacts", "run_benchmarks", "validate_outputs", "hardware_smoke"):
        assert statuses[stage] == "skipped"


def test_hailo8_engine_python_resolver_skips_missing_onnx_and_uses_valid_python(tmp_path: Path, monkeypatch) -> None:
    mod = _load_script("native_fifo_smoke_matrix.py")
    good = tmp_path / "good-python"
    bad = tmp_path / "bad-python"
    good.write_text("", encoding="utf-8"); bad.write_text("", encoding="utf-8")
    ns = SimpleNamespace(
        precision="float32_layout_fp16", dequant_scale=0.0, dequant_zero_point=0.0,
        boundary_layout="memory_nhwc_to_nchw", timeout=30.0, engine_build_python="auto",
    )
    monkeypatch.setattr(mod, "_candidate_engine_pythons", lambda requested: [(str(bad), {}, "bad"), (str(good), {}, "good")])
    monkeypatch.setattr(mod, "_python_can_import", lambda py, module, env=None: (py == str(good), "probe"))
    monkeypatch.setattr(mod, "_run", lambda cmd, timeout=None, env=None: {"cmd": cmd, "rc": 0, "elapsed_s": 0.01, "stdout_tail": "ok", "stderr_tail": "", "timed_out": False})
    steps = mod._build_native_trt_part2(tmp_path, "b001", ns)
    assert steps[0]["name"] == "build_native_trt_part2_skip_python_missing_onnx"
    assert steps[-1]["name"] == "build_native_trt_part2"
    assert steps[-1]["engine_build_python"] == str(good)
    assert steps[-1]["rc"] == 0


def test_native_producer_csv_serializes_steps_and_deepx_preflight(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    bs = root / "model" / "benchmark_set"
    (bs / "b001").mkdir(parents=True)
    (bs / "benchmark_set.json").write_text(json.dumps({"cases": [{"case_id": "b001"}]}), encoding="utf-8")
    (bs / "benchmark_suite.py").write_text("print('suite')\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "native_producer_e2e_eval_runner.py"),
         "--root", str(root), "--backend", "deepx", "--models", "model",
         "--case-map", '{"model":["b001"]}', "--analysis-tag", "v60v"],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert proc.returncode == 3, proc.stdout
    terminal = json.loads(proc.stdout)
    assert terminal["ok"] is False
    assert terminal["orchestration_status"] == "failed"
    out = root / "analysis_tables" / "native_deepx_producer_e2e_eval__v60v.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rows"][0]["status"] == "missing_deepx_part1_artifact"
    csv_path = out.with_suffix(".csv")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows and json.loads(rows[0]["steps_json"])[0]["name"] == "deepx_part1_preflight"


def test_remote_resource_copies_include_engine_python_and_steps_fix() -> None:
    resource = ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
    assert "--engine-build-python" in (resource / "native_fifo_eval_runner.py").read_text(encoding="utf-8")
    assert "_candidate_engine_pythons" in (resource / "native_fifo_smoke_matrix.py").read_text(encoding="utf-8")
    e2e = (resource / "native_producer_e2e_eval_runner.py").read_text(encoding="utf-8")
    assert "steps_json" in e2e
    assert "missing_deepx_part1_artifact" in e2e
    assert (resource / "native_trt_from_benchmarkset.py").is_file()
