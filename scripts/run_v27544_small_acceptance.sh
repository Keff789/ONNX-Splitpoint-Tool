#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.44 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27544_small_acceptance.sh was sourced.' \
    'Run: bash scripts/run_v27544_small_acceptance.sh' >&2
  return 0
fi

set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  PYTHON="$SOURCE_ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf '1/4 identity, syntax and v2.75.44 cohort-projection contract\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.44-runtime-validation-cohort-projection"
assert tool.__version__ == "2.75.44", tool.__version__
assert tool.__release__ == "2.75.44", tool.__release__
assert tool.__development_lineage__ == "v2.75.44"
assert tool.__build_id__ == expected
assert WORKFLOW_VERSION == expected
assert {
    "legacy_source_to_runtime_validation_cohort_projection",
    "legacy_portable_dataset_identity_compatibility",
    "installed_workspace_manifest_scope",
    "real_b500_provisioning_authority",
    "setup_local_tensorrt_quality_companion_identity_gate",
} <= set(tool.__build_features__)
for path in (
    Path("profiles/resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml"),
    Path("scripts/preflight_v27541_deepx_calibration_1000.py"),
    Path("scripts/run_v27543_small_acceptance.sh"),
    Path("scripts/refresh_editable_install.py"),
    Path("onnx_splitpoint_tool/deepx/calibration_size_canary.py"),
    Path("onnx_splitpoint_tool/v27543_smoke.py"),
    Path("onnx_splitpoint_tool/v27544_smoke.py"),
):
    assert path.is_file(), path
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27544_small_acceptance.sh \
  scripts/run_v27543_small_acceptance.sh \
  scripts/run_v27542_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh

printf '2/4 installed-source integrity (retained user profiles permitted)\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed

printf '3/4 focused projection, canary and retained-path tests\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27544_release_provenance.py \
  tests/test_v27543_release_provenance.py \
  tests/test_v27542_release_provenance.py \
  tests/test_v27542_installed_source_manifest_scope.py \
  tests/test_v27542_stdlib_editable_refresh.py \
  tests/test_v27541_release_provenance.py \
  tests/test_v27541_calibration_baseline_pin.py \
  tests/test_v27541_deepx_calibration_1000_profile_preflight.py \
  tests/test_v27541_deepx_calibration_size_canary.py \
  tests/test_v269a_deepx_full_central_quality.py \
  tests/test_v60j_complete_dataset_provisioning.py \
  tests/test_v269e_quality_export_vendored.py \
  tests/test_deepx_full_preprocessing_ab_contract.py \
  tests/test_v27540_deepx_preprocessing_probe.py \
  tests/test_v27540_deepx_preprocessing_ab_pairing.py \
  tests/test_v27540_ready_deepx_preprocessing_ab_profiles.py \
  tests/test_deepx_full_only_quality_dispatch.py \
  tests/test_v27521_setup_local_trt_dispatch.py \
  tests/test_v27522_full_only_debug_pack.py \
  tests/test_v269d_full_dedupe_endpoint_contracts.py \
  tests/test_v269d_management_trt_quality_summary.py \
  tests/test_v269d_trt_quality_only_transport.py \
  tests/test_v27538_full_only_performance_matrix.py \
  tests/test_v27539_remote_premutation_dispatch.py \
  tests/test_v27539_remote_dispatch_propagation.py

printf '4/4 current release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27544_smoke

printf '%s\n' \
  'PASS v2.75.44 small acceptance' \
  'SKIP hardware, SSH, DX-COM, TensorRT build, Energy and inference'
