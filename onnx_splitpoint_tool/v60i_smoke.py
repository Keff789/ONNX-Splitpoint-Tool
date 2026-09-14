"""Fast, hardware-independent smoke tests for the v60i evidence pipeline.

The command is intentionally small and deterministic.  It checks the defects that
were exposed by the v60g development run before a user invests hours in a new
compiler/remote benchmark campaign:

* explicit per-model task routing;
* exact task-quality policy hashing and nested-gate ingestion;
* classification self-reference Top-1 enforcement;
* separation of screening observations and claim-eligible reports;
* runtime work-unit marker parsing;
* availability/capability markers of remotely synchronised helper scripts.

It does not claim to replace a short hardware smoke run.  In particular it cannot
exercise SSH, vendor runtimes, u.RECS acquisition or the physical accelerators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from . import __version__
from .benchmark.remote_run import _infer_suite_benchmark_task
from .validation.accuracy_gates import AccuracyGatePolicy, apply_accuracy_gate_to_row
from .workflow.results import normalize_benchmark_files, build_normalized_results_payload
from .workflow.scientific_reporting import build_scientific_reports
from .energy.collector import _extract_work_units_from_text_v60i
from .workflow.runner import WORKFLOW_VERSION


Check = dict[str, Any]


def _result(name: str, ok: bool, detail: str = "", **extra: Any) -> Check:
    return {"name": name, "ok": bool(ok), "detail": detail, **extra}


def _check_task_routing() -> Check:
    with tempfile.TemporaryDirectory(prefix="v60i-task-") as td:
        root = Path(td) / "resnet_coco_stale_hint"
        root.mkdir(parents=True)
        plan = {
            "model_id": "resnet50",
            "task": "classification",
            "benchmark_task": "classification",
            "runs": [{"task": "classification", "validation_dir": "/stale/coco/path"}],
        }
        p = root / "benchmark_plan.json"
        p.write_text(json.dumps(plan), encoding="utf-8")
        task = _infer_suite_benchmark_task(root, p)
        return _result("explicit_model_task_routing", task == "classification", f"resolved={task}")


def _policy() -> AccuracyGatePolicy:
    return AccuracyGatePolicy(
        name="v60i_smoke_policy",
        profile_id="v60i_smoke_policy",
        dataset_tier="screening",
        confidence_level=0.95,
        bootstrap_repetitions=5000,
        bootstrap_seed=20260710,
        screening_eligible_for_ranking=False,
    )


def _check_nested_gate_and_policy_hash() -> Check:
    policy = _policy()
    row = {
        "model_id": "detector",
        "case_id": "b001",
        "backend": "hailo8_to_tensorrt",
        "variant": "split",
        "task": "detection",
        "compile_ok": True,
        "runtime_ok": True,
        "contract_consistent": True,
        "task_quality_gate": {
            "decision": "fail",
            "tier": "screening",
            "policy": policy.as_dict(),
            "primary": {
                "metric": "coco_ap_50_95",
                "candidate": 0.40,
                "reference": 0.43,
                "delta": -0.03,
                "ci_low": -0.04,
                "ci_high": -0.02,
                "margin": 0.01,
                "n": 50,
            },
        },
    }
    apply_accuracy_gate_to_row(row, policy)
    ok = (
        row.get("accuracy_gate_decision") == "fail"
        and row.get("accuracy_gate_policy_match") is True
        and row.get("accuracy_gate_policy_sha256") == policy.sha256()
        and row.get("runtime_quality_gate_policy_sha256") == policy.sha256()
    )
    return _result(
        "nested_quality_gate_and_policy_hash",
        ok,
        f"decision={row.get('accuracy_gate_decision')} match={row.get('accuracy_gate_policy_match')}",
    )


def _check_classification_top1_contract() -> Check:
    row = {
        "model": "resnet50",
        "task": "classification",
        "ok": True,
        "self_reference_available": True,
        "self_reference_ok": True,
        "semantic_ok": True,
        "top1_match": False,
        "top5_overlap": 5,
        "buildable": True,
        "runtime_executable": True,
        "structural_contract_pass": True,
    }
    apply_accuracy_gate_to_row(row, _policy())
    ok = (
        row.get("contract_consistent") is True
        and row.get("numerical_similarity_pass") is False
        and row.get("performance_eligible") is False
    )
    return _result(
        "classification_top1_contract",
        ok,
        (
            f"contract={row.get('contract_consistent')} "
            f"numeric={row.get('numerical_similarity_pass')} "
            f"reason={row.get('numerical_similarity_reason')}"
        ),
    )


def _check_scientific_report_split() -> Check:
    with tempfile.TemporaryDirectory(prefix="v60i-report-") as td:
        run = Path(td)
        (run / "models" / "m" / "benchmark_results").mkdir(parents=True)
        profile = {
            "name": "v60i_smoke",
            "campaign": {"mode": "development"},
            "quality_gate": _policy().as_dict(),
            "models": [{"id": "m", "task": "detection", "evaluation_role": "development"}],
        }
        try:
            import yaml
            (run / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
        except Exception:
            # The full package requires PyYAML, but JSON is also valid YAML.
            (run / "profile.yaml").write_text(json.dumps(profile), encoding="utf-8")
        raw = {
            "model_id": "m",
            "case_id": "b001",
            "backend": "ort_tensorrt",
            "variant": "split",
            "task": "detection",
            "compile_ok": True,
            "runtime_ok": True,
            "validation_ok": True,
            "contract_consistent": True,
            "total_latency_ms": 10.0,
            "fps": 100.0,
            "task_quality_gate": {
                "decision": "pass",
                "tier": "screening",
                "policy": _policy().as_dict(),
                "primary": {
                    "metric": "coco_ap_50_95",
                    "candidate": 0.40,
                    "reference": 0.40,
                    "delta": 0.0,
                    "ci_low": -0.001,
                    "ci_high": 0.001,
                    "margin": 0.01,
                    "n": 50,
                },
            },
        }
        raw_path = run / "models" / "m" / "benchmark_results" / "benchmark_results_smoke.json"
        raw_path.write_text(json.dumps([raw]), encoding="utf-8")
        rows, sources = normalize_benchmark_files(model_id="m", source_paths=[raw_path.parent])
        payload = build_normalized_results_payload(model_id="m", results=rows, sources=sources, run_root=run)
        (raw_path.parent / "normalized_results.json").write_text(json.dumps(payload), encoding="utf-8")
        build_scientific_reports(
            run,
            profile_id="v60i_smoke",
            tool_version=__version__,
            workflow_version=WORKFLOW_VERSION,
        )
        root = run / "reports" / "scientific"
        claim = root / "claim_eligible_performance.csv"
        screening = root / "screening_performance_observations.csv"
        ok = claim.is_file() and screening.is_file()
        if ok:
            claim_rows = max(0, sum(1 for _ in claim.open(encoding="utf-8")) - 1)
            screening_rows = max(0, sum(1 for _ in screening.open(encoding="utf-8")) - 1)
            ok = claim_rows == 0 and screening_rows >= 1
        else:
            claim_rows = screening_rows = -1
        return _result(
            "screening_vs_claim_reporting",
            ok,
            f"claim_rows={claim_rows} screening_rows={screening_rows}",
        )


def _check_work_unit_marker() -> Check:
    text = "noise\n__SPLITPOINT_WORK_UNITS__=1234\n__SPLITPOINT_WORK_UNITS_SOURCE__=frames\n"
    count, source = _extract_work_units_from_text_v60i(text)
    return _result("runtime_work_unit_marker", count == 1234 and source == "frames", f"count={count} source={source}")


def _check_script_capabilities() -> Check:
    root = Path(__file__).resolve().parents[1]
    requirements: Mapping[str, tuple[str, ...]] = {
        "scripts/native_full_baseline_eval_runner.py": ("--setup-id", "--comparison-backend", "frames"),
        "scripts/native_producer_energy_plan.py": ("--validation-summary", "runtime_marker_preferred"),
        "scripts/run_and_report_work_units.py": ("__SPLITPOINT_WORK_UNITS__",),
        "scripts/native_producer_validate_visualize.py": ("--quality-gate-json", "top1_match"),
    }
    missing: list[str] = []
    for rel, tokens in requirements.items():
        p = root / rel
        if not p.is_file():
            p = Path(__file__).resolve().parent / "resources" / "remote_scripts" / Path(rel).name
        if not p.is_file():
            missing.append(f"{rel}:missing_file")
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for token in tokens:
            if token not in text:
                missing.append(f"{rel}:{token}")
    return _result("remote_script_capability_markers", not missing, "ok" if not missing else "; ".join(missing))


def run_smoke() -> dict[str, Any]:
    checks: list[Check] = []
    funcs: list[Callable[[], Check]] = [
        _check_task_routing,
        _check_nested_gate_and_policy_hash,
        _check_classification_top1_contract,
        _check_scientific_report_split,
        _check_work_unit_marker,
        _check_script_capabilities,
    ]
    for fn in funcs:
        try:
            checks.append(fn())
        except Exception as exc:  # expose failure without hiding the remaining checks
            checks.append(_result(fn.__name__.lstrip("_"), False, f"{type(exc).__name__}: {exc}"))
    passed = sum(1 for c in checks if c.get("ok"))
    return {
        "schema": "onnx-splitpoint/v60i-smoke-report",
        "schema_version": 1,
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "status": "ok" if passed == len(checks) else "failed",
        "passed": passed,
        "failed": len(checks) - passed,
        "checks": checks,
        "hardware_coverage": False,
        "note": "Run a short targeted hardware EvalRun after this local smoke passes.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fast v60i evidence-pipeline smoke tests")
    parser.add_argument("--json", dest="json_path", default="", help="Optional path for the JSON report")
    parser.add_argument("--quiet", action="store_true", help="Print only the final status line")
    ns = parser.parse_args(argv)
    report = run_smoke()
    if ns.json_path:
        out = Path(ns.json_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not ns.quiet:
        for check in report["checks"]:
            marker = "PASS" if check.get("ok") else "FAIL"
            print(f"[{marker}] {check.get('name')}: {check.get('detail')}")
    print(
        f"v60i smoke: {report['status']} "
        f"({report['passed']} passed, {report['failed']} failed) "
        f"tool={report['tool_version']} workflow={report['workflow_version']}"
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
