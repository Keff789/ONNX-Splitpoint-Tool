from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from onnx_splitpoint_tool.workflow.debug_pack_policy import (
    OFFLINE_REPLAY_CORE,
    is_offline_replay_core,
)
from scripts import native_producer_final_report as final_report


ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _load_debug_pack_cli():
    path = ROOT / "scripts" / "create_evaluation_debug_pack.py"
    spec = importlib.util.spec_from_file_location(
        "v27518_create_evaluation_debug_pack", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_report_projects_missing_runtime_policy_diagnostic_alias() -> None:
    """Replay the policy shape emitted by the real v2.75.17 producer."""

    policy_sha256 = _digest("smoke-task-quality-policy")
    projected = final_report._claim_contract_fields({
        "task_quality_policy_sha256": policy_sha256,
        "runtime_quality_gate_policy_sha256": "",
    })

    assert projected["task_quality_policy_sha256"] == policy_sha256
    assert projected["runtime_quality_gate_policy_sha256"] == policy_sha256


def test_native_full_collector_preserves_projected_runtime_policy_alias(
    tmp_path: Path,
) -> None:
    """Exercise the real row assembly, including later dict overrides."""

    policy_sha256 = _digest("smoke-task-quality-policy")
    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    (analysis / "native_full_baseline_eval.json").write_text(
        json.dumps({
            "rows": [{
                "backend": "native_full_deepx",
                "model": "resnet50.onnx",
                "ok": True,
                "outer_makespan_verified": True,
                "task_quality_policy_sha256": policy_sha256,
                "runtime_quality_gate_policy_sha256": "",
            }],
        }),
        encoding="utf-8",
    )

    rows = final_report._rows_from_native_full(tmp_path)

    assert len(rows) == 1
    assert rows[0]["task_quality_policy_sha256"] == policy_sha256
    assert rows[0]["runtime_quality_gate_policy_sha256"] == policy_sha256


def test_final_report_preserves_conflicting_explicit_runtime_policy() -> None:
    """An explicit conflict must remain visible to the fail-closed validator."""

    task_sha256 = _digest("task-policy")
    runtime_sha256 = _digest("different-runtime-policy")
    projected = final_report._claim_contract_fields({
        "task_quality_policy_sha256": task_sha256,
        "runtime_quality_gate_policy_sha256": runtime_sha256,
    })

    assert projected["task_quality_policy_sha256"] == task_sha256
    assert projected["runtime_quality_gate_policy_sha256"] == runtime_sha256


def test_gui_and_cli_share_the_exact_bounded_replay_core_policy(
    tmp_path: Path,
) -> None:
    cli = _load_debug_pack_cli()
    assert cli.REPLAY_CORE is OFFLINE_REPLAY_CORE
    assert len(OFFLINE_REPLAY_CORE) == 4
    assert not is_offline_replay_core(
        "reports/native_energy_measurements/attempt/raw.parquet"
    )
    assert not is_offline_replay_core(
        "../reports/native_energy_measurements/native_producer_energy_results.json"
    )

    for relative in OFFLINE_REPLAY_CORE:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * 4096, encoding="utf-8")
        included, reason = cli._should_include(
            tmp_path,
            path,
            16,
            probe_include_raw=False,
        )
        assert included is True, (relative, reason)

    gui_source = (ROOT / "onnx_splitpoint_tool/gui/app.py").read_text(
        encoding="utf-8"
    )
    builder_source = (
        ROOT / "onnx_splitpoint_tool/workflow/debug_pack.py"
    ).read_text(encoding="utf-8")
    assert "from ..workflow.debug_pack import create_evaluation_debug_pack" in gui_source
    assert "result = create_evaluation_debug_pack(" in gui_source
    assert 'return "offline_replay_core"' in builder_source
    assert '"offline_replay_core": {' in builder_source
