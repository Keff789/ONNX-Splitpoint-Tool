from __future__ import annotations

import hashlib
import json
from pathlib import Path

from onnx_splitpoint_tool import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v27547_smoke import (
    BUILD_ID,
    CRITICAL_MODULES,
    NEW_FEATURES,
    REQUIRED_FEATURES,
    REQUIRED_FILES,
    RETAINED_V27546_FEATURES,
    VALIDATOR_MIRROR_SHA256,
    VERSION,
    main as smoke_main,
)
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v27547_exact_release_identity_and_features() -> None:
    assert VERSION == "2.75.47"
    assert BUILD_ID == (
        "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    )
    assert __version__ == VERSION
    assert __release__ == VERSION
    assert __development_lineage__ == f"v{VERSION}"
    assert __build_id__ == BUILD_ID
    assert WORKFLOW_VERSION == BUILD_ID
    assert __build_contract_version__ == 2
    assert NEW_FEATURES == {
        "model_bound_yolov7_anchor_contract",
        "official_coco_yolov7_decoder_ab_probe",
        "generic_native_yolov7_decoder_parity",
        "debug_pack_ranking_audit_intent_fix",
        "debug_pack_non_audit_completeness",
        "early_declared_model_sha256_admission",
    }
    assert REQUIRED_FEATURES <= set(__build_features__)
    assert RETAINED_V27546_FEATURES <= set(__build_features__)


def test_v27547_packaging_documents_profile_and_harness_are_pinned() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    root_block = lock.split('name = "onnx-splitpoint-tool"', 1)[1]
    assert root_block.lstrip().startswith('version = "2.75.47"')
    for marker in (
        'onnx-splitpoint-smoke-v27547 = '
        '"onnx_splitpoint_tool.v27547_smoke:main"',
        'onnx-splitpoint-smoke-v2-75-47 = '
        '"onnx_splitpoint_tool.v27547_smoke:main"',
        'onnx-splitpoint-smoke-v27546 = '
        '"onnx_splitpoint_tool.v27546_smoke:main"',
        'onnx-splitpoint-smoke-v2-75-46 = '
        '"onnx_splitpoint_tool.v27546_smoke:main"',
    ):
        assert marker in pyproject
    for name in REQUIRED_FILES:
        assert (ROOT / name).is_file(), name
    harness = ROOT / "scripts/run_v27547_small_acceptance.sh"
    assert harness.stat().st_mode & 0o100
    assert (
        ROOT / "scripts/run_yolov7_decoder_ab_probe.sh"
    ).stat().st_mode & 0o100
    harness_text = harness.read_text(encoding="utf-8")
    assert "--scope installed" in harness_text
    assert "test_v27547_yolov7_decoder_contract.py" in harness_text
    assert "test_v27547_model_hash_gate.py" in harness_text
    assert "test_compact_debug_pack.py" in harness_text
    assert "test_v27547_release_integration.py" in harness_text
    probe_text = (ROOT / "scripts/probe_yolov7_decoder_ab.py").read_text(
        encoding="utf-8"
    )
    assert "OFFICIAL_COCO_MAX_DETS = (1, 10, 100)" in probe_text
    for name in (
        "README.md",
        "docs/README.md",
        "docs/VERSIONING.md",
        "TESTANLEITUNG_2.75.47.md",
        "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert VERSION in text
        assert BUILD_ID in text


def test_v27547_claim_critical_inventory_covers_both_repairs() -> None:
    build = package_build_snapshot()
    critical = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert CRITICAL_MODULES <= critical
    assert "validation/official_coco.py" in critical
    assert (
        "resources/remote_scripts/native_producer_validate_visualize.py"
        in critical
    )
    source = ROOT / "scripts/native_producer_validate_visualize.py"
    mirror = (
        ROOT
        / "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_producer_validate_visualize.py"
    )
    assert source.read_bytes() == mirror.read_bytes()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        VALIDATOR_MIRROR_SHA256
    )


def test_v27547_source_allowlist_retains_v27546_release_documents() -> None:
    helper = (ROOT / "scripts/build_source_manifest.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        '"TESTANLEITUNG_2.75.46.md"',
        '"VERSION_2.75.46_BUILD_AND_TEST_REPORT.md"',
        'f"TESTANLEITUNG_{package_version}.md"',
        'f"VERSION_{package_version}_BUILD_AND_TEST_REPORT.md"',
    ):
        assert marker in helper


def test_v27547_release_scripts_target_current_and_previous_smokes() -> None:
    updater = (ROOT / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    local_acceptance = (ROOT / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8"
    )
    native_console = (ROOT / "scripts/native_console_smoke.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "--expected-version 2.75.47",
        "onnx-splitpoint-smoke-v27547",
        "onnx-splitpoint-smoke-v2-75-47",
        "onnx-splitpoint-smoke-v27546",
        "onnx-splitpoint-smoke-v2-75-46",
        "--run-entrypoint onnx-splitpoint-smoke-v27547",
        "onnx-splitpoint-tool==2.75.47",
        "pycocotools",
        "dependency_bootstrap --groups yolov7_probe",
        "pip install --upgrade 'pycocotools>=2.0.7'",
        ">= (2, 0, 7)",
        'required=("onnxruntime","numpy","PIL","pycocotools")',
        "printf '  cd -- %q\\n' \"$TOOL_DIR\"",
    ):
        assert marker in updater
    for marker in (
        'EXPECTED_VERSION="2.75.47"',
        f'EXPECTED_BUILD="{BUILD_ID}"',
        "scripts/run_v27547_small_acceptance.sh",
        "tests/test_v27547_release_provenance.py",
        "tests/test_v27547_release_integration.py",
        "tests/test_v27547_yolov7_decoder_contract.py",
        "tests/test_v27547_model_hash_gate.py",
        "onnx_splitpoint_tool.v27547_smoke",
    ):
        assert marker in local_acceptance
    for marker in (
        "tests/test_v27547_release_provenance.py",
        "tests/test_v27547_release_integration.py",
        "tests/test_v27547_yolov7_decoder_contract.py",
        "tests/test_v27547_model_hash_gate.py",
        "tests/test_compact_debug_pack.py",
    ):
        assert marker in native_console


def test_v27547_source_manifest_metadata_parser_reads_current_identity() -> None:
    from scripts.build_source_manifest import _metadata

    assert _metadata(ROOT) == (VERSION, BUILD_ID)


def test_v27547_guide_keeps_cpu_ab_a_hard_pre_gui_gate() -> None:
    guide = (ROOT / "TESTANLEITUNG_2.75.47.md").read_text(encoding="utf-8")
    guide_lines = guide.splitlines()
    assert guide_lines.count("## Frischer YOLOv7-only GUI-Anker") == 1
    fence_open = False
    for line in guide_lines:
        if not line.startswith("```"):
            continue
        assert line == ("```" if fence_open else "```bash")
        fence_open = not fence_open
    assert fence_open is False
    for marker in (
        "dependency_bootstrap",
        "--groups yolov7_probe",
        "pip install --upgrade 'pycocotools>=2.0.7'",
        "pycocotools>=2.0.7",
        "pycocotools",
        "--preflight-only",
        "--validation-manifest",
        "--selection-request",
        "--images-root",
        "--output-dir",
        "official_coco",
        "status: completed",
        "acceptance.status: accepted",
        "status: ok",
        "max_detections=300",
        "maxDets=[1,10,100]",
        "standard_production_ap_50_95_at_least_020",
        "standard_production_ap50_at_least_035",
        "standard_production_ap75_at_least_020",
        "standard_production_ap75_ap50_ratio_at_least_045",
        "upstream_sanity_ap_50_95_at_least_025",
        "upstream_sanity_ap50_at_least_040",
        "upstream_sanity_ap75_at_least_025",
        "upstream_sanity_ap75_ap50_ratio_at_least_050",
        "canonical_accuracy_claim",
        "kein kanonischer",
        "Returncode 2",
        "kein** Evidence-Archiv",
        "Full-run-Output-Beanspruchung",
        "Evidence-Ordner",
        "Evaluation Results Bundle",
        "3d51d22511681cc28b99c2d927fe08003b92a234a4ec101683a0875d0ae72c32",
        "b8ac329f5d3e7e201a3938d36c63bfe96a13c8a2be6e4fe6ddfab89cbc135e45",
        "yolov7_paper_v27547_standard_anchor_b500.yaml",
        "Start",
        "nicht Resume",
        "B500",
        "kein B1000",
        "Native-first",
        "Energy aus",
        'ARCHIVE="${OUT}.tar.gz"',
        'SIDECAR="${ARCHIVE}.manifest.json"',
        'test -s "$ARCHIVE"',
        'test -s "$SIDECAR"',
        'payload["archive_sha256"] == actual_sha256',
        'payload["archive_member_count"] == actual_members',
        'test -d "$IMAGES"',
        'test ! -e "${OUT}.tar.gz.manifest.json"',
    ):
        assert marker in guide

    report = (ROOT / "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md").read_text(
        encoding="utf-8"
    )
    for marker in (
        "AP50:95 >= 0,25",
        "AP50:95 >= 0,20",
        "AP50 >= 0,35",
        "AP75 >= 0,20",
        "AP75/AP50 >= 0,45",
        "AP50 >= 0,40",
        "AP75 >= 0,25",
        "AP75/AP50 >= 0,50",
        "keinen kanonischen",
        "canonical_accuracy_claim=false",
    ):
        assert marker in report


def test_v27547_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v27547-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0
