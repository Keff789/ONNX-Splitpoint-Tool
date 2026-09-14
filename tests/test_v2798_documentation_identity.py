from __future__ import annotations

import re
from pathlib import Path

from onnx_splitpoint_tool import __build_features__
from onnx_splitpoint_tool.release_identity import (
    BUILD_ID as CURRENT_BUILD_ID,
    VERSION as CURRENT_VERSION,
)
from onnx_splitpoint_tool.v2798_smoke import (
    BUILD_ID as V2798_BUILD_ID,
    NEW_FEATURES,
    REQUIRED_FEATURES,
    VERSION as V2798_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_root_readme_declares_current_v27910_and_retains_v2798_loader_closure() -> None:
    text = _read("README.md")
    assert text.splitlines()[0] == (
        f"# ONNX Split-Point Tool — release {CURRENT_VERSION}"
    )
    match = re.search(
        r"^Current identity:\n\n(?P<items>(?:- [^\n]+\n)+)",
        text,
        flags=re.MULTILINE,
    )
    assert match is not None
    assert match.group("items").splitlines() == [
        f"- package and GUI release: `{CURRENT_VERSION}` (public release line `v2.79`)",
        f"- workflow/build: `{CURRENT_BUILD_ID}`",
    ]
    historical_heading = "Historical Release 2.79.8"
    next_heading = "Historical release 2.79.7"
    assert historical_heading in text and next_heading in text
    historical = text[text.index(historical_heading):text.index(next_heading)]
    assert V2798_BUILD_ID in historical
    assert "validated evaluation-profile loader" in historical
    assert "exact runtime snapshot loader" in historical


def test_documentation_index_declares_current_and_historical_identity() -> None:
    text = _read("docs/README.md")
    assert text.splitlines()[:4] == [
        f"# Documentation index — current release {CURRENT_VERSION}",
        "",
        "Current build/workflow:",
        f"`{CURRENT_BUILD_ID}`.",
    ]
    historical_heading = "Historical Release 2.79.8"
    next_heading = "Historical release 2.79.7"
    assert historical_heading in text and next_heading in text
    historical = text[text.index(historical_heading):text.index(next_heading)]
    assert "strict profile/schema admission mismatch" in historical
    assert "both real\npre-hardware loader paths" in historical


def test_versioning_current_table_and_v2798_history_are_exact() -> None:
    text = _read("docs/VERSIONING.md")
    current_heading = "## Current release: 2.79.13 — platform-power calibration operational repair"
    v27912_heading = (
        "## Historical release: 2.79.12 — platform-power calibration provenance closure"
    )
    v27911_heading = (
        "## Historical release: 2.79.11 — platform-power, energy and evidence closure"
    )
    v27910_heading = (
        "## Historical release: 2.79.10 — platform-power release closure"
    )
    v2799_heading = (
        "## Historical release: 2.79.9 — u.RECS platform power and M.2 idle calibration"
    )
    v2798_heading = (
        "## Historical release: 2.79.8 — YOLO11 gate-profile schema closure"
    )
    v2797_heading = (
        "## Historical release: 2.79.7 — YOLO11 six-path runtime-identity closure"
    )
    for heading in (current_heading, v27912_heading, v27911_heading, v27910_heading, v2799_heading, v2798_heading, v2797_heading):
        assert heading in text
    assert text.index(current_heading) < text.index(v27912_heading)
    assert text.index(v27912_heading) < text.index(v27911_heading)
    assert text.index(v27911_heading) < text.index(v27910_heading)
    assert text.index(v27910_heading) < text.index(v2799_heading)
    assert text.index(v2799_heading) < text.index(v2798_heading)
    assert text.index(v2798_heading) < text.index(v2797_heading)

    current = text[text.index(current_heading):text.index(v27912_heading)]
    assert f"Build/workflow: `{CURRENT_BUILD_ID}`." in current
    assert (
        f"| GUI and GitHub release label | `{CURRENT_VERSION}` / `v2.79` |"
        in current
    )
    assert f"| Python package version | `{CURRENT_VERSION}` |" in current

    historical = text[text.index(v2798_heading):text.index(v2797_heading)]
    assert V2798_VERSION == "2.79.8"
    assert f"Build/workflow: `{V2798_BUILD_ID}`." in historical
    assert "validated evaluation-profile loader" in historical
    assert "runtime snapshot\nloader" in historical


def test_native_three_stage_document_marks_v27910_current_and_v2798_historical() -> None:
    text = _read("docs/NATIVE_THREE_STAGE.md")
    assert text.splitlines()[0] == (
        f"# Native three-stage execution (current release {CURRENT_VERSION})"
    )
    current_heading = "## v2.79.13 current release binding"
    v27912_heading = "## v2.79.12 historical release binding"
    v27911_heading = "## v2.79.11 historical release binding"
    v27910_heading = "## v2.79.10 historical release binding"
    v2799_heading = "## v2.79.9 historical release binding"
    v2798_heading = "## v2.79.8 historical release binding"
    v2797_heading = "## v2.79.7 historical release binding"
    for heading in (current_heading, v27912_heading, v27911_heading, v27910_heading, v2799_heading, v2798_heading, v2797_heading):
        assert heading in text
    assert text.index(current_heading) < text.index(v27912_heading)
    assert text.index(v27912_heading) < text.index(v27911_heading)
    current = text[text.index(current_heading):text.index(v27912_heading)]
    historical = text[text.index(v2798_heading):text.index(v2797_heading)]
    assert CURRENT_BUILD_ID in current
    assert V2798_BUILD_ID in historical
    assert "does not change Three-Stage\nor Native Full execution" in historical


def test_v2798_feature_contract_remains_frozen_and_cumulative() -> None:
    required = {
        "yolo11_gate_profile_schema_closure",
        "yolo11_gate_profile_real_loader_preflight",
    }
    assert required == set(NEW_FEATURES)
    assert required <= set(REQUIRED_FEATURES)
    assert required <= set(__build_features__)
