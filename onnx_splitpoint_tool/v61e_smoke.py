from __future__ import annotations

import ast
import json
from pathlib import Path

from . import __development_lineage__, __next_major_version__, __release__, __version__
from .execution_plan import build_effective_execution_plan
from .workflow.runner import WORKFLOW_VERSION

EXPECTED_VERSION = ("2.61.0+v61e", "2.62.0", "2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.7", "2.75.8", "2.75.10", "2.75.11", "2.75.12", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26")
EXPECTED_RELEASE = ("2.61e", "2.62", "2.63", "2.64", "2.65", "2.66", "2.67", "2.68", "2.69f", "2.70f", "2.70g", "2.70h", "2.70i", "2.70j", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.7", "2.75.8", "2.75.10", "2.75.11", "2.75.12", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.26")
EXPECTED_LINEAGE = ("v61e", "v2.62", "v2.63", "v2.64", "v2.65", "v2.66", "v2.67", "v2.68", "v2.69f", "v2.70f", "v2.70g", "v2.70h", "v2.70i", "v2.70j", "v2.71.1", "v2.71.2", "v2.71.3", "v2.71.4", "v2.72.0", "v2.72.1", "v2.72.2", "v2.72.3", "v2.72.4", "v2.72.5", "v2.72.6", "v2.73.0", "v2.73.6", "v2.73.7", "v2.75.2", "v2.75.7", "v2.75.8", "v2.75.10", "v2.75.11", "v2.75.12", "v2.75.14", "v2.75.16", "v2.75.17", "v2.75.20", "v2.75.21", "v2.75.22", "v2.75.24", "v2.75.26")
EXPECTED_NEXT_MAJOR = "3.0.0"
EXPECTED_WORKFLOW = (
    "v2.61e-campaign-contract-hardening",
    "v2.62-window-validation-native-binding",
    "v2.63-campaign-ready",
    "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair",
)


def _panel_has_no_free_self(root: Path) -> bool:
    tree = ast.parse((root / "gui/panels/panel_hardware.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "build_panel":
            return not any(isinstance(child, ast.Name) and child.id == "self" for child in ast.walk(node))
    return False


def _ranking_warning_ok() -> bool:
    profile = {
        "model_suite": {"primary": [{"id": "resnet50", "enabled": True, "task": "classification"}]},
        "selection_policy": {"max_accepted_cases_per_model": 1},
        "run_profiles": [{"id": "hailo8_to_tensorrt", "enabled": True}],
        "execution_preset": {
            "id": "standard",
            "label": "Standard",
            "snapshot": {
                "defaults": {"native_enabled": True, "energy_enabled": False},
                "quality": {"bootstrap_repetitions": 500},
                "ranking": {"enabled": True, "minimum_candidates_for_correlation": 3},
                "runtime": {"native": {}, "benchmark": {}},
                "data": {"validation_items": {"classification": 500, "detection": 500}},
                "build": {"hailo": {}},
            },
            "overrides": {},
        },
    }
    plan = build_effective_execution_plan(profile)
    return (
        plan.get("ranking_candidate_shortfall") == 2
        and any(row.get("id") == "ranking_candidate_shortfall" for row in plan.get("warnings", []))
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    app_text = (root / "gui/app.py").read_text(encoding="utf-8")
    template = (root / "resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    checks = {
        "version": (
            __version__ in EXPECTED_VERSION
            and __release__ in EXPECTED_RELEASE
            and __development_lineage__ in EXPECTED_LINEAGE
            and __next_major_version__ == EXPECTED_NEXT_MAJOR
        ),
        "workflow": WORKFLOW_VERSION in EXPECTED_WORKFLOW,
        "tool_config_no_free_self": _panel_has_no_free_self(root),
        "lazy_tab_survives_builder_error": (
            "Der Tab bleibt verfügbar" in app_text
            and "Erneut laden" in app_text
            and "Only now replace the placeholder" in app_text
        ),
        "ranking_shortfall_warning": _ranking_warning_ok(),
        "standard_bootstrap_progress": all(
            marker in template
            for marker in (
                "[quality-gate][bootstrap] START",
                "[quality-gate][bootstrap] CACHE_READY",
                "[quality-gate][bootstrap] PROGRESS",
                "eta=",
                "[quality-gate][bootstrap] END",
            )
        ),
    }
    result = {
        "ok": all(checks.values()),
        "passed": sum(bool(v) for v in checks.values()),
        "failed": sum(not bool(v) for v in checks.values()),
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
