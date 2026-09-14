from __future__ import annotations

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
from onnx_splitpoint_tool.v27546_smoke import (
    BUILD_ID,
    CURRENT_BUILD_ID,
    CURRENT_VERSION,
    CRITICAL_MODULES,
    NEW_FEATURES,
    REQUIRED_FEATURES,
    REQUIRED_FILES,
    RETAINED_V27545_FEATURES,
    VERSION,
    main as smoke_main,
)
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v27546_exact_release_identity_and_features() -> None:
    assert VERSION == "2.75.46"
    assert BUILD_ID == (
        "v2.75.46-native-full-onnx-attestation-"
        "standard-quality-projection"
    )
    assert CURRENT_VERSION == "2.75.47"
    assert CURRENT_BUILD_ID == (
        "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    )
    assert __version__ == CURRENT_VERSION
    assert __release__ == CURRENT_VERSION
    assert __development_lineage__ == f"v{CURRENT_VERSION}"
    assert __build_id__ == CURRENT_BUILD_ID
    assert WORKFLOW_VERSION == CURRENT_BUILD_ID
    assert __build_contract_version__ == 2
    assert NEW_FEATURES == {
        "native_full_selected_onnx_interpreter_attestation",
        "native_full_onnx_attestation_exception_diagnostics",
        "standard_setup_local_quality_contract_projection",
        "generic_quality_diagnostic_row_separation",
        "deepx_calibration_size_output_contract_hash_normalization",
        "deepx_calibration_size_semantic_endpoint_invariant",
    }
    assert REQUIRED_FEATURES <= set(__build_features__)


def test_v27546_packaging_documents_and_harness_are_pinned() -> None:
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
        'onnx-splitpoint-smoke-v27545 = '
        '"onnx_splitpoint_tool.v27545_smoke:main"',
        'onnx-splitpoint-smoke-v2-75-45 = '
        '"onnx_splitpoint_tool.v27545_smoke:main"',
    ):
        assert marker in pyproject
    for name in REQUIRED_FILES:
        assert (ROOT / name).is_file(), name
    harness = ROOT / "scripts/run_v27546_small_acceptance.sh"
    assert harness.stat().st_mode & 0o100
    harness_text = harness.read_text(encoding="utf-8")
    assert "--scope installed" in harness_text
    assert "test_v27546_hailo_onnx_interpreter.py" in harness_text
    assert (
        "test_v27546_calibration_size_canary_output_contract.py"
        in harness_text
    )
    assert "test_v27546_standard_quality_projection.py" in harness_text
    for name in (
        "TESTANLEITUNG_2.75.46.md",
        "VERSION_2.75.46_BUILD_AND_TEST_REPORT.md",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert VERSION in text
        assert BUILD_ID in text


def test_v27546_claim_critical_inventory_covers_all_repairs() -> None:
    build = package_build_snapshot()
    critical = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert CRITICAL_MODULES <= critical


def test_v27546_preserves_v27545_historical_release_assets_and_features() -> None:
    release_marked = (
        "onnx_splitpoint_tool/v27545_smoke.py",
        "scripts/run_v27545_small_acceptance.sh",
        "tests/test_v27545_release_provenance.py",
        "TESTANLEITUNG_2.75.45.md",
        "VERSION_2.75.45_BUILD_AND_TEST_REPORT.md",
    )
    for name in release_marked:
        path = ROOT / name
        assert path.is_file(), name
        assert "2.75.45" in path.read_text(encoding="utf-8")
    assert (
        ROOT / "tests/test_v27545_large_audit_working_set_admission.py"
    ).is_file()
    assert RETAINED_V27545_FEATURES <= set(__build_features__)


def test_v27546_source_allowlist_retains_v27545_and_v27546_release_docs() -> None:
    helper = (ROOT / "scripts/build_source_manifest.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        '"TESTANLEITUNG_2.75.45.md"',
        '"VERSION_2.75.45_BUILD_AND_TEST_REPORT.md"',
        'f"TESTANLEITUNG_{package_version}.md"',
        'f"VERSION_{package_version}_BUILD_AND_TEST_REPORT.md"',
    ):
        assert marker in helper


def test_v27546_release_scripts_target_current_and_previous_smokes() -> None:
    updater = (ROOT / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    local_acceptance = (
        ROOT / "scripts/run_local_acceptance.sh"
    ).read_text(encoding="utf-8")
    native_console = (
        ROOT / "scripts/native_console_smoke.py"
    ).read_text(encoding="utf-8")
    for marker in (
        "--expected-version 2.75.47",
        "onnx-splitpoint-smoke-v27547",
        "onnx-splitpoint-smoke-v2-75-47",
        "onnx-splitpoint-smoke-v27546",
        "onnx-splitpoint-smoke-v2-75-46",
        "onnx-splitpoint-smoke-v27545",
        "onnx-splitpoint-smoke-v2-75-45",
    ):
        assert marker in updater
    for marker in (
        'EXPECTED_VERSION="2.75.47"',
        f'EXPECTED_BUILD="{CURRENT_BUILD_ID}"',
        "scripts/run_v27547_small_acceptance.sh",
        "tests/test_v27547_release_provenance.py",
        "onnx_splitpoint_tool.v27547_smoke",
    ):
        assert marker in local_acceptance
    for marker in (
        "tests/test_v27546_release_provenance.py",
        "tests/test_v27546_hailo_onnx_interpreter.py",
        "tests/test_v27546_standard_quality_projection.py",
        "tests/test_v27546_calibration_size_canary_output_contract.py",
    ):
        assert marker in native_console


def test_v27546_source_manifest_metadata_parser_reads_current_identity() -> None:
    from scripts.build_source_manifest import _metadata

    assert _metadata(ROOT) == (CURRENT_VERSION, CURRENT_BUILD_ID)


def test_v27546_remote_runner_copy_is_exact() -> None:
    source = ROOT / "scripts/native_full_baseline_eval_runner.py"
    packaged = (
        ROOT
        / "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_full_baseline_eval_runner.py"
    )
    assert source.read_bytes() == packaged.read_bytes()


def test_v27546_anchor_profile_loads_with_exact_frozen_axes(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "run_modes.yaml"),
    )
    profile = (
        ROOT
        / "profiles/resnet_yolo26s_yolov7_v27546_standard_anchor_b500.yaml"
    )
    loaded = load_evaluation_profile(
        str(profile), base_dir=profile.parent, validate=True,
    )
    plan = build_effective_execution_plan(loaded.raw_profile)
    assert loaded.profile_id == (
        "resnet_yolo26s_yolov7_v27546_standard_anchor_b500"
    )
    assert loaded.start_snapshot["consistency"] == {
        "status": "ok",
        "mismatches": [],
    }
    assert loaded.raw_profile["selection_policy"]["forced_cases"] == {
        "resnet50": ["b052"],
        "yolo26s": ["b038"],
        "yolov7_paper": ["b044"],
    }
    assert plan["candidate_counts_by_model"] == {
        "resnet50": 1,
        "yolo26s": 1,
        "yolov7_paper": 1,
    }
    assert plan["native_enabled"] is True
    assert plan["native_full_baselines"] is True
    assert plan["generic_energy_enabled"] is False
    assert plan["native_energy_enabled"] is False
    assert plan["ranking_enabled"] is False
    assert plan["score_independent_audit_enabled"] is False
    assert plan["calibration_items"] == {
        "classification": 500,
        "detection": 500,
    }
    assert plan["validation_items"] == {
        "classification": 500,
        "detection": 500,
    }
    assert plan["bootstrap_repetitions"] == 500
    assert plan["deepx_classification_preprocessing"] == (
        "imagenet_mean_std"
    )
    assert plan["deepx_classification_preprocessing_explicit"] is True
    assert plan["deepx_full_cache_contract"] == "v2_exact_explicit"
    assert plan["deepx_cache_dir"] == (
        "~/Models/BackendArtifacts/deepx/v2.75.44/"
        "thesis_standard_b500_imagenet_mean_std"
    )


def test_v27546_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v27546-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0
