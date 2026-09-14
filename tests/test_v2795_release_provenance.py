from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool as pkg
from onnx_splitpoint_tool import v2794_smoke, v2795_smoke

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.79.5"
BUILD_ID = "v2.79.5-release-launcher-evidence-closure"


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_historical_v2795_identity_remains_available() -> None:
    assert v2795_smoke.VERSION == VERSION
    assert v2795_smoke.LINEAGE == "v2.79"
    assert v2795_smoke.BUILD_ID == BUILD_ID
    assert v2794_smoke.VERSION == "2.79.4"


def test_v2795_declares_complete_closure_contract() -> None:
    required = {
        "release_line_smoke_alias_tracks_current_maintenance_release",
        "native_three_stage_productized_single_or_multi_image_corpus",
        "native_three_stage_explicit_corpus_reference_and_out_root",
        "native_three_stage_nonempty_stage_timing_projection",
        "native_three_stage_nonempty_oracle_parity_projection",
        "current_release_identity_and_acceptance_alias_closure",
        "v2795_seven_model_launcher_release_closure",
        "version_dynamic_evidence_source_snapshot_prefix",
        "current_release_concurrent_runner_evidence_labels",
        "native_three_stage_oracle_parity_status_fail_closed",
    }
    assert required <= v2795_smoke.NEW_FEATURES
    assert required <= v2795_smoke.REQUIRED_FEATURES
    assert required <= set(pkg.__build_features__)


def test_historical_v2795_entrypoints_remain_available() -> None:
    pyproject = _text("pyproject.toml")
    for marker in (
        'onnx-splitpoint-smoke-v2795 = "onnx_splitpoint_tool.v2795_smoke:main"',
        'onnx-splitpoint-smoke-v2-79-5 = "onnx_splitpoint_tool.v2795_smoke:main"',
        'onnx-splitpoint-smoke-v2794 = "onnx_splitpoint_tool.v2794_smoke:main"',
    ):
        assert marker in pyproject


def test_historical_v2795_acceptance_remains_self_identifying() -> None:
    release = _text("scripts/run_v2795_small_acceptance.sh")
    assert "onnx_splitpoint_tool.v2795_smoke" in release
    assert "tests/test_v2795_release_provenance.py" in release
    assert "PASS v2.79.5 small acceptance" in release
