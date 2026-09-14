"""Hardware-independent operational repair smoke for version 2.79.13."""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .ranking_methods import RANKING_METHOD_IMPLEMENTATION, WORKFLOW_RANKING_METHOD
from .release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, VERSION
from .v27912_smoke import REQUIRED_FEATURES as V27912_REQUIRED_FEATURES
from .workflow.runner import WORKFLOW_VERSION

LINEAGE = DEVELOPMENT_LINEAGE
NEW_FEATURES = {
    "platform_power_gui_missing_method_preparation",
    "installed_operational_root_scope",
    "v27911_editable_launcher_upgrade",
}
REQUIRED_FEATURES = set(V27912_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    updater = (root / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    generic_acceptance = (
        root / "scripts/run_v279_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    local_acceptance = (root / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8"
    )
    acceptance = (root / "scripts/run_v27913_small_acceptance.sh").read_text(
        encoding="utf-8"
    )
    generic_launcher = (
        root / "scripts/run_v279_seven_model_long_overnight.sh"
    ).read_text(encoding="utf-8")
    checks = {
        "version": __version__ == __release__ == VERSION == "2.79.13",
        "lineage": __development_lineage__ == LINEAGE == "v2.79",
        "build": (
            __build_id__
            == WORKFLOW_VERSION
            == BUILD_ID
            == "v2.79.13-platform-power-calibration-operational-repair"
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "ranking_freeze": (
            WORKFLOW_RANKING_METHOD == "cut_bytes_only"
            and RANKING_METHOD_IMPLEMENTATION
            == "v277-cut-bytes-only-workflow-freeze-1"
        ),
        "current_files": all(
            (root / relative).is_file()
            for relative in (
                "onnx_splitpoint_tool/v27913_smoke.py",
                "onnx_splitpoint_tool/gui/panels/panel_hardware.py",
                "onnx_splitpoint_tool/source_integrity.py",
                "scripts/build_source_manifest.py",
                "scripts/refresh_editable_install.py",
                "scripts/update_source_release.sh",
                "scripts/run_v27913_small_acceptance.sh",
                "scripts/run_v27913_seven_model_long_overnight.sh",
                "scripts/launcher_status_v27913.py",
                "profiles/complete_set_7models_v27913_b500_audit20.yaml",
                "tests/test_v27913_platform_power_operational_repair.py",
                "tests/test_v27913_release_provenance.py",
            )
        ),
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'version = "2.79.13"',
                'onnx-splitpoint-smoke-v27913 = "onnx_splitpoint_tool.v27913_smoke:main"',
                'onnx-splitpoint-smoke-v2-79-13 = "onnx_splitpoint_tool.v27913_smoke:main"',
            )
        ),
        "updater": all(
            marker in updater
            for marker in (
                "--expected-version 2.79.13",
                "onnx-splitpoint-smoke-v27913=onnx_splitpoint_tool.v27913_smoke:main",
                "--run-entrypoint onnx-splitpoint-smoke-v27913",
                "onnx-splitpoint-tool==2.79.13",
            )
        ),
        "acceptance_aliases": (
            'run_v27913_small_acceptance.sh" "$@"' in generic_acceptance
            and "run_v27913_small_acceptance.sh" in local_acceptance
        ),
        "installed_manifest_scope": (
            "build_source_manifest.py" in acceptance
            and '--root "$SOURCE_ROOT" --verify --scope installed'
            in acceptance
        ),
        "longrun_alias": (
            'CURRENT_LAUNCHER="$SCRIPT_DIR/run_v27913_seven_model_long_overnight.sh"'
            in generic_launcher
            and 'exec bash "$CURRENT_LAUNCHER" "$@"' in generic_launcher
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(
            json.dumps(
                {"status": "FAIL", "failed": failed, "checks": checks},
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {"status": "PASS", "version": VERSION, "checks": checks},
            indent=2,
        )
    )
    print("PASS v2.79.13 smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
