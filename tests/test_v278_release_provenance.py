from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.ranking_methods import (
    RANKING_METHOD_IMPLEMENTATION,
    WORKFLOW_RANKING_METHOD,
)
from onnx_splitpoint_tool.v278_smoke import (
    BUILD_ID,
    LINEAGE,
    REQUIRED_FEATURES,
    VERSION,
    main as smoke_main,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v278_exact_release_identity() -> None:
    assert VERSION == "2.78.4"
    assert LINEAGE == "v2.78"
    assert BUILD_ID == "v2.78.4-gate-a-planned-stop-audit-launch"
    assert __version__ == VERSION
    assert __release__ == VERSION
    assert __development_lineage__ == LINEAGE
    assert __build_id__ == BUILD_ID
    assert WORKFLOW_VERSION == BUILD_ID
    assert __build_contract_version__ == 2
    assert REQUIRED_FEATURES <= set(__build_features__)
    assert "gate_a_planned_stop_audit_launch" in __build_features__
    assert "forced_audit_deployment_anchor_first" in __build_features__
    assert "frozen_seven_model_b500_audit20_launch" in __build_features__
    assert "frozen_hardware_registry_projection" in __build_features__
    assert "read_only_existing_evidence_verification" in __build_features__
    assert "real_request_reference_descriptor_compatibility" in __build_features__
    assert "immutable_management_cpu_reference_snapshots" in __build_features__
    assert "exact_hailo_build_evidence_recovery_index" in __build_features__
    assert "hailo8_first_common_anchor_feasibility" in __build_features__
    assert "generic_hailo_native_quality_selection_gate" in __build_features__
    assert "deepx_model_sidecar_identity" in __build_features__
    assert "resnet_quantized_interface_semantic_aggregation" in __build_features__


def test_v278_ranker_freeze_identity() -> None:
    assert WORKFLOW_RANKING_METHOD == "cut_bytes_only"
    assert RANKING_METHOD_IMPLEMENTATION == (
        "v277-cut-bytes-only-workflow-freeze-1"
    )


def test_v278_documentation_uses_exact_current_build_identity() -> None:
    obsolete = "v2.78.3-native-dual-endpoint-completion-tail-instrumentation"
    for relative in ("README.md", "docs/README.md", "docs/VERSIONING.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert BUILD_ID in text, relative
        assert obsolete not in text, relative


def test_v278_packaging_and_entrypoints() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.78.4"' in pyproject
    root_block = lock.split('name = "onnx-splitpoint-tool"', 1)[1]
    assert root_block.lstrip().startswith('version = "2.78.4"')
    assert (
        'onnx-splitpoint-smoke-v278 = '
        '"onnx_splitpoint_tool.v278_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-78 = '
        '"onnx_splitpoint_tool.v278_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v277 = '
        '"onnx_splitpoint_tool.v277_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-77 = '
        '"onnx_splitpoint_tool.v277_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-backend-semantic-smoke = '
        '"onnx_splitpoint_tool.backend_semantic_smoke:main"'
    ) in pyproject
    assert (ROOT / "onnx_splitpoint_tool/existing_evidence_verifier.py").is_file()
    assert (ROOT / "onnx_splitpoint_tool/management_reference.py").is_file()
    assert (ROOT / "onnx_splitpoint_tool/quality_replay.py").is_file()
    assert (ROOT / "tests/test_v2782_management_reference_resume.py").is_file()
    assert (ROOT / "tests/test_v2782_quality_replay_reference_path.py").is_file()
    assert (ROOT / "scripts/verify_existing_adapter_evidence.py").is_file()
    assert (ROOT / "scripts/recover_b5_build_evidence.py").is_file()
    assert (
        ROOT / "scripts/run_v2783_yolo11_hailo8_first_gate.sh"
    ).is_file()
    assert (ROOT / "onnx_splitpoint_tool/build_evidence.py").is_file()
    assert (ROOT / "onnx_splitpoint_tool/runtime_evidence.py").is_file()
    assert (
        ROOT / "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"
    ).is_file()
    assert (
        ROOT / "profiles/complete_set_7models_v2784_b500_audit20.yaml"
    ).is_file()
    assert (
        ROOT / "scripts/run_v2784_seven_model_long_overnight.sh"
    ).is_file()
    assert (
        ROOT / "scripts/verify_v2783_yolo11_gate_a_output.py"
    ).is_file()


def test_v278_offline_updater_targets_current_release() -> None:
    updater = (ROOT / "scripts/update_source_release.sh").read_text(
        encoding="utf-8",
    )
    local_acceptance = (ROOT / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8",
    )
    assert "--expected-version 2.78.4" in updater
    assert (
        "onnx-splitpoint-backend-semantic-smoke="
        "onnx_splitpoint_tool.backend_semantic_smoke:main"
    ) in updater
    assert (
        "onnx-splitpoint-smoke-v277="
        "onnx_splitpoint_tool.v277_smoke:main"
    ) in updater
    assert (
        "onnx-splitpoint-smoke-v2-77="
        "onnx_splitpoint_tool.v277_smoke:main"
    ) in updater
    assert (
        "onnx-splitpoint-smoke-v278="
        "onnx_splitpoint_tool.v278_smoke:main"
    ) in updater
    assert (
        "onnx-splitpoint-smoke-v2-78="
        "onnx_splitpoint_tool.v278_smoke:main"
    ) in updater
    assert "--run-entrypoint onnx-splitpoint-smoke-v278" in updater
    assert "onnx-splitpoint-tool==2.78.4" in updater
    assert "bash scripts/run_v278_small_acceptance.sh" in local_acceptance


def test_v278_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    assert "PASS v2.78 smoke" in capsys.readouterr().out
