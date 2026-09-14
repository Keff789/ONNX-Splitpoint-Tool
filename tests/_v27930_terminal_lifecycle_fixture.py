"""Synthetic stages; production run(), jobs, cleanup, lock and index lifecycle.

No hardware/quality claims are made by these small management-only runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from onnx_splitpoint_tool.workflow.artifacts import write_json
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


class TerminalLifecycleRunner(EvaluationWorkflowRunner):
    measurement_status = "failed"

    def _load_profile(self) -> None:
        self.profile_id = "terminal-management-fixture"
        self.profile_path = str(self.options.profile)
        self.profile_source = "synthetic offline stages"
        self.profile_payload = {"profile_id": self.profile_id, "models": []}

    def _resolve_model_rows(self) -> list[dict[str, Any]]:
        return []

    def _run_model_pipeline_with_cache_preflight(self, rows: object) -> None:
        return None

    def _stage_resolve_profile(self, *_args: Any) -> tuple:
        return {}, {}, "synthetic profile stage", "ok"

    _stage_campaign_preflight = _stage_resolve_profile
    _stage_evaluate_quality = _stage_resolve_profile
    _stage_aggregate_results = _stage_resolve_profile
    _stage_run_native_producers = _stage_resolve_profile

    def _stage_generate_report(self, *_args: Any) -> tuple:
        path = self.run_dir / "reports/synthetic_measurements.json"
        write_json(path, {"diagnostic_only": True, "status": self.measurement_status})
        return {"synthetic_report": path}, {}, "synthetic report", "ok"

    def _derive_final_status(self) -> tuple[str, dict[str, Any]]:
        return self.measurement_status, {"blocking_reasons": []}


def options_for(root: Path, run_id: str = "terminal-fixture") -> WorkflowOptions:
    root.mkdir(parents=True, exist_ok=True)
    profile = root / "synthetic_profile.yaml"
    profile.write_text("profile_id: terminal-management-fixture\nmodels: []\n", encoding="utf-8")
    return WorkflowOptions(
        profile=str(profile), out=str(root / "runs"), run_id=run_id,
        no_remote=True, skip_benchmarks=True, dry_run=True,
    )


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }


def closure(runner: EvaluationWorkflowRunner) -> dict[str, Any]:
    return json.loads((runner.run_dir / "reports/artifact_index_closure.json").read_text())
