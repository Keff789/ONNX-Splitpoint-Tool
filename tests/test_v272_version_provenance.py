from __future__ import annotations

import importlib.util
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from onnx_splitpoint_tool import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v272_smoke import main as smoke_main
from onnx_splitpoint_tool.v272_smoke import (
    CURRENT_BUILD_ID,
    CURRENT_VERSION,
)
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def _load_release_builder():
    path = ROOT / "scripts" / "build_source_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_source_release_v272",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_fixture(root: Path) -> None:
    (root / "onnx_splitpoint_tool" / "workflow").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "9.8.7"\n',
        encoding="utf-8",
    )
    (root / "onnx_splitpoint_tool" / "workflow" / "runner.py").write_text(
        'WORKFLOW_VERSION = "fixture-workflow"\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text("payload\n", encoding="utf-8")
    executable = root / "start_gui.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)


def test_v272_identity_is_exact() -> None:
    assert __version__ == CURRENT_VERSION
    assert __release__ == CURRENT_VERSION
    assert __development_lineage__ == f"v{CURRENT_VERSION}"
    assert __build_id__ == CURRENT_BUILD_ID
    assert WORKFLOW_VERSION == CURRENT_BUILD_ID
    assert __build_contract_version__ == 2
    assert {
        "completed_detection_execution_contract",
        "separate_detection_hash_domains",
        "exact_quality_evidence_binding_axes",
        "planned_vs_matrix_energy_coverage",
        "canonical_completed_result_artifact_transport_hash",
        "contract_bound_yolov7_activation_strategy",
        "detection_completion_preflight_identity",
        "prospective_hailo10_yolo26_claim_exclusion",
        "four_dimensional_claim_exclusion_reporting",
        "hailo8_system_trt_process_local_hailo_sites",
        "mixed_runtime_import_preflight",
        "primary_child_error_precedence",
        "completed_v2_structural_consumer_precedence",
        "orthogonal_quality_observation_and_accuracy",
        "quality_first_energy_collector_admission",
        "contract_frozen_energy_repeat_counts",
        "complete_energy_exclusion_ledger",
        "canonical_energy_coverage_projection",
        "post_native_expected_energy_matrix",
        "shared_sealed_energy_quality_admission",
        "fail_closed_energy_plan_result_ledger",
        "completed_v2_prelegacy_consumer_normalization",
        "legacy_positive_energy_evidence_projection",
        "canonical_profile_selection_snapshot_identity",
        "setup_scoped_central_quality_request_identity",
        "semantic_central_quality_mirror_deduplication",
        "portable_native_split_semantic_dumps",
        "hailo_full_raw_endpoint_quality_binding",
        "timed_frozen_decoder_quality_invariant",
        "smoke_bound_negative_quality_observation_energy",
        "canonical_detection_quality_contract_projection",
        "manifest_bound_central_quality_staging_deduplication",
        "backend_bound_resume_artifact_roles",
        "hailo8_exact_row_resume_contract",
        "hailo8_python_detection_preflight_source_recovery",
        "native_energy_runtime_success_admission",
        "deepx_exact_row_resume_artifact_rehydration",
        "campaign_preflight_success_exit_contract",
        "exact_yolov7_paper_final_template",
        "runtime_exact_campaign_contract_templates",
        "yolov7_paper_final_canary",
        "inherited_energy_method_runtime_binding_gate",
    }.issubset(set(__build_features__))


def test_v272_packaging_metadata_and_docs_are_current() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    versioning = (ROOT / "docs" / "VERSIONING.md").read_text(
        encoding="utf-8",
    )
    assert 'version = "2.75.47"' in pyproject
    assert (
        'onnx-splitpoint-smoke-v272 = '
        '"onnx_splitpoint_tool.v272_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-72 = '
        '"onnx_splitpoint_tool.v272_smoke:main"'
    ) in pyproject
    root_block = lock.split('name = "onnx-splitpoint-tool"', 1)[1]
    assert root_block.lstrip().startswith('version = "2.75.47"')
    for text in (readme, versioning):
        assert CURRENT_VERSION in text
        assert CURRENT_BUILD_ID in text


def test_v272_release_smoke_and_critical_inventory(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["passed"] == 12
    assert result["failed"] == 0
    build = package_build_snapshot()
    assert build["critical_module_set_complete"] is True
    assert {
        "v272_smoke.py",
        "native_detection_diagnostics.py",
        "native_energy_quality_admission.py",
        "resume_hailo8_source_recovery.py",
    }.issubset(set(build["critical_module_sha256"]))


def test_v272_source_console_smoke_selects_current_regressions() -> None:
    console = (ROOT / "scripts" / "native_console_smoke.py").read_text(
        encoding="utf-8",
    )
    for test_path in (
        "tests/test_v2727_native_energy_runtime_admission.py",
        "tests/test_v2727_native_energy_runtime_admission_edges.py",
        "tests/test_v2727_native_energy_runtime_admission_contract.py",
        "tests/test_v2726_hailo8_resume_preflight_closure.py",
        "tests/test_v2725_hailo8_resume_contract.py",
        "tests/test_v2723_campaign_gate_repairs.py",
        "tests/test_v2722_offline_contract_repair.py",
        "tests/test_v2721_hailo8_mixed_runtime_launcher.py",
        "tests/test_v2721_completed_v2_host_consumer.py",
        "tests/test_v2721_energy_contract_repairs.py",
        "tests/test_v272_version_provenance.py",
        "tests/test_v272_detection_completion_runtime.py",
        "tests/test_v272_detection_hotloop_integration.py",
        "tests/test_v272_hailo8_completed_detection_hotloop.py",
        "tests/test_v272_evidence_status_contract.py",
        "tests/test_v272_evidence_status_reporting.py",
        "tests/test_v272_claim_exclusion_reporting.py",
        "tests/test_v272_yolov7_head_mapping.py",
        "tests/test_hailo10_yolo26_full_semantic_fixture.py",
        "tests/test_v2621_split_energy_preflight.py",
        "tests/test_v269f_hailo_trt_interface_contract.py",
        "tests/test_v269f_vendor_full_detection.py",
        "tests/test_v269e_quality_export_vendored.py",
        "tests/test_v269a_central_quality_request_join.py",
    ):
        assert test_path in console
    fixture = (
        ROOT
        / "tests"
        / "fixtures"
        / "v272"
        / "hailo10_yolo26_full_semantic.json"
    )
    assert fixture.is_file()
    fixture_payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert fixture_payload["confidence_threshold"] == 0.25
    assert "Run the narrow 2.75.47 source regression block." in console


def test_source_release_builder_is_byte_deterministic(tmp_path: Path) -> None:
    builder = _load_release_builder()
    source = tmp_path / "source"
    source.mkdir()
    _source_fixture(source)
    builder.source_manifest.build(source)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = builder.build_release(source, first)
    second_result = builder.build_release(source, second)

    assert first_result["ok"] is True
    assert second_result["ok"] is True
    assert first.read_bytes() == second.read_bytes()
    assert first_result["archive_sha256"] == second_result["archive_sha256"]
    assert first_result["member_count"] == (
        first_result["manifest_file_count"] + 2
    )
    with zipfile.ZipFile(first) as archive:
        infos = archive.infolist()
        assert archive.testzip() is None
        assert [info.filename for info in infos] == sorted(
            info.filename for info in infos
        )
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0)
            and info.compress_type == zipfile.ZIP_DEFLATED
            and info.create_system == 3
            for info in infos
        )
        executable = archive.getinfo(
            "ONNX-Splitpoint-Tool_v9.8.7/start_gui.sh",
        )
        assert stat.S_IMODE(executable.external_attr >> 16) == 0o755


def test_source_manifest_excludes_nested_temp_editor_coverage_and_os_junk(
    tmp_path: Path,
) -> None:
    builder = _load_release_builder()
    source = tmp_path / "source"
    source.mkdir()
    _source_fixture(source)
    source_config = source / ".coveragerc"
    source_config.write_text("[run]\nbranch = true\n", encoding="utf-8")
    junk_paths = (
        ".venv/bin/python",
        "nested/__pycache__/module.pyc",
        "nested/.pytest_cache/state",
        "nested/tmp/result.json",
        "nested/temp/result.json",
        "nested/.tmp/result.json",
        "nested/.temp/result.json",
        "nested/.cache/result.json",
        "nested/artifacts/.coverage/chunk",
        "nested/.hypothesis/example",
        "nested/.nox/session/result.json",
        "nested/htmlcov/index.html",
        "nested/.idea/workspace.xml",
        "nested/.vscode/settings.json",
        "nested/__MACOSX/._payload.py",
        "nested/result.tmp",
        "nested/RESULT.TMP",
        "nested/result.temp",
        "nested/result.bak",
        "nested/result.backup",
        "nested/result.rej",
        "nested/result.swp",
        "nested/result.swo",
        "nested/result.part",
        "nested/result.partial",
        "nested/result.crdownload",
        "nested/result.download",
        "nested/draft.py~",
        "nested/.#draft.py",
        "nested/.coverage",
        "nested/.coverage.worker-1",
        "nested/.COVERAGE.worker-2",
        "nested/coverage.xml",
        "nested/coverage.json",
        "nested/.DS_Store",
        "nested/.directory",
        "nested/._payload.py",
        "nested/Thumbs.db",
        "nested/desktop.ini",
    )
    for relative in junk_paths:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("junk\n", encoding="utf-8")

    manifest = builder.source_manifest.build(source)
    included = {row["path"] for row in manifest["files"]}

    assert ".coveragerc" not in included
    assert included.isdisjoint(junk_paths)
    assert included == {
        "README.md",
        "onnx_splitpoint_tool/workflow/runner.py",
        "pyproject.toml",
        "start_gui.sh",
    }


def test_source_release_builder_rejects_in_tree_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_release_builder()
    source = tmp_path / "source"
    source.mkdir()
    _source_fixture(source)
    builder.source_manifest.build(source)
    output = source / "release-artifacts" / "source.zip"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_source_release.py",
            "--root",
            str(source),
            "--out",
            str(output),
        ],
    )

    with pytest.raises(
        ValueError,
        match="source-release output must be outside the source root",
    ):
        builder.main()

    assert not output.parent.exists()


def test_source_release_builder_fails_closed_on_stale_manifest(
    tmp_path: Path,
) -> None:
    builder = _load_release_builder()
    source = tmp_path / "source"
    source.mkdir()
    _source_fixture(source)
    builder.source_manifest.build(source)
    (source / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source manifest is stale"):
        builder.build_release(source, tmp_path / "stale.zip")


def test_source_release_builder_rejects_symlinks(tmp_path: Path) -> None:
    builder = _load_release_builder()
    source = tmp_path / "source"
    source.mkdir()
    _source_fixture(source)
    target = source / "README.md"
    link = source / "start_gui.sh"
    link.unlink()
    try:
        os.symlink(target.name, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    builder.source_manifest.build(source)

    with pytest.raises(ValueError, match="may not be symlinks"):
        builder.build_release(source, tmp_path / "symlink.zip")
