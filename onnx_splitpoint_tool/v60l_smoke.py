"""Fast hardware-independent smoke checks for v60l evidence binding fixes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .campaign import create_dataset_manifest
from .dataset_provisioning import load_registry, save_registry
from .v60k_smoke import run_smoke as run_v60k_smoke
from .validation.accuracy_gates import AccuracyGatePolicy
from .workflow.dataset_binding import bind_profile_dataset_registry, calibration_manifest_for_task
from .workflow.runner import WORKFLOW_VERSION, _bare_sha256_v60l

ROOT = Path(__file__).resolve().parents[1]


def _result(name: str, ok: bool, detail: str = "", **extra: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail, **extra}


def _runtime_loader():
    text = (ROOT / "onnx_splitpoint_tool" / "resources" / "templates" / "run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    tree = ast.parse(text)
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_task_quality_policy")
    ns = {"Any": object, "Dict": dict, "Path": Path, "json": json, "hashlib": hashlib}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "<v60l-policy>", "exec"), ns)
    return ns["_load_task_quality_policy"]


def _check_quality_policy_inline_json() -> dict[str, Any]:
    policy = {
        "name": "task_quality_v1",
        "profile_id": "task_quality_v1",
        "dataset_tier": "screening",
        "canonical_reference": "canonical_full_onnx",
        "classification": {"primary_metric": "top1_accuracy", "non_inferiority_margin": 0.01, "guardrails": {"top5_accuracy_margin": 0.01}},
        "detection": {"primary_metric": "coco_ap_50_95", "non_inferiority_margin": 0.01, "guardrails": {"ap50_margin": 0.01, "ap75_margin": 0.01}},
        "statistics": {"method": "paired_bootstrap", "confidence_level": 0.95, "bootstrap_repetitions": 5000, "seed": 20260710, "decision": "lower_one_sided_bound"},
        "native_contract": {"detection_self_reference_min_match_ratio": 0.9, "self_reference_counts_as_task_accuracy": False},
        "screening_eligible_for_ranking": False,
        "legacy_point_estimate_eligible_for_ranking": False,
    }
    parsed = _runtime_loader()(json.dumps(policy, separators=(",", ":")))
    expected = AccuracyGatePolicy.from_mapping(policy).sha256()
    return _result(
        "inline_quality_policy_exact_hash",
        parsed.get("policy_sha256") == expected and parsed.get("statistics", {}).get("bootstrap_repetitions") == 5000,
        f"runtime={parsed.get('policy_sha256')} expected={expected}",
    )


def _check_hash_normalisation() -> dict[str, Any]:
    ok = _bare_sha256_v60l("sha256:ABC123") == _bare_sha256_v60l("abc123") == "abc123"
    return _result("remote_script_hash_normalisation", ok, "prefixed local and bare remote digest compare equally")


def _check_dataset_registry_binding() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v60l-bind-") as temp:
        root = Path(temp)
        data = root / "cls" / "n00000001"
        data.mkdir(parents=True)
        (data / "a.jpg").write_bytes(b"a")
        manifest = create_dataset_manifest(
            task="classification", role="calibration", dataset_id="cls", split="train",
            root=data.parent, output=root / "cls_cal.json", max_items=0,
        )
        registry_path = root / "dataset_registry.json"
        registry = load_registry(registry_path)
        registry["manifests"] = {"classification_calibration": str(manifest)}
        save_registry(registry, registry_path)
        profile = {"campaign": {"dataset_registry": str(registry_path), "auto_bind_dataset_registry": True, "dataset_manifests": {"classification": {"calibration": ""}}}}
        bound, report = bind_profile_dataset_registry(profile, verify_manifests=True)
        ok = calibration_manifest_for_task(bound, "classification") == str(manifest.resolve()) and report.get("bound_count") == 1
        return _result("dataset_registry_auto_binding", ok, f"status={report.get('status')} bound={report.get('bound_count')}")


def _check_phase_probe() -> dict[str, Any]:
    text = (ROOT / "onnx_splitpoint_tool" / "resources" / "templates" / "benchmark_suite.py.txt").read_text(encoding="utf-8")
    ok = 'runner_supports_phase_runs = "--phase-runs" in _runner_probe' in text and '[:50000]' not in text
    return _result("full_runner_capability_probe", ok, "--phase-runs probe reads the complete generated runner")


def run_smoke() -> dict[str, Any]:
    base = run_v60k_smoke()
    checks = list(base.get("checks") or [])
    for fn in (_check_hash_normalisation, _check_quality_policy_inline_json, _check_dataset_registry_binding, _check_phase_probe):
        try:
            checks.append(fn())
        except Exception as exc:
            checks.append(_result(fn.__name__.lstrip("_"), False, f"{type(exc).__name__}: {exc}"))
    passed = sum(1 for row in checks if row.get("ok"))
    return {
        "schema": "onnx-splitpoint/v60l-smoke-report",
        "schema_version": 1,
        "tool_version": __version__,
        "workflow_version": WORKFLOW_VERSION,
        "status": "ok" if passed == len(checks) else "failed",
        "passed": passed,
        "failed": len(checks) - passed,
        "checks": checks,
        "hardware_coverage": False,
        "network_coverage": False,
        "note": "The smoke checks the v60l policy, registry, hash and generated-runner fixes without contacting accelerators.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fast v60l evidence-binding smoke tests")
    parser.add_argument("--json", dest="json_path", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = run_smoke()
    if args.json_path:
        out = Path(args.json_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if not args.quiet:
        for check in report["checks"]:
            print(f"[{'PASS' if check.get('ok') else 'FAIL'}] {check.get('name')}: {check.get('detail')}")
    print(f"v60l smoke: {report['status']} ({report['passed']} passed, {report['failed']} failed) tool={report['tool_version']} workflow={report['workflow_version']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

# v60m: development profiles keep screening validation unless final validation
# was explicitly selected; calibration manifests may still auto-bind.
from onnx_splitpoint_tool.v60m_policy import install_dataset_binding_guards as _v60m_install_dataset_binding_guards
_v60m_install_dataset_binding_guards(globals())
