#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.41 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27541_small_acceptance.sh was sourced.' \
    'Run: bash scripts/run_v27541_small_acceptance.sh' >&2
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

printf '1/4 identity, syntax and v2.75.41 feature contract\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.41-deepx-calibration-500-vs-1000-canary"
assert tool.__version__ == "2.75.41", tool.__version__
assert tool.__release__ == "2.75.41", tool.__release__
assert tool.__development_lineage__ == "v2.75.41"
assert tool.__build_id__ == expected
assert WORKFLOW_VERSION == expected
assert {
    "deepx_calibration_size_canary",
    "deepx_calibration_500_1000_subset_lock",
    "deepx_calibration_size_pairing_verifier",
    "deepx_calibration_1000_readiness_preflight",
    "packaged_resnet_v27541_deepx_calibration_1000_profile",
} <= set(tool.__build_features__)
for path in (
    Path("profiles/resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml"),
    Path("scripts/pin_v27541_deepx_calibration_baseline.py"),
    Path("scripts/preflight_v27541_deepx_calibration_1000.py"),
    Path("scripts/verify_v27541_deepx_calibration_size_canary.py"),
    Path("onnx_splitpoint_tool/deepx/calibration_size_canary.py"),
):
    assert path.is_file(), path
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27541_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh

printf '2/4 read-only source manifest\n'
"$PYTHON" -B scripts/build_source_manifest.py --root "$SOURCE_ROOT" --verify

printf '3/4 focused calibration-canary and retained-path tests\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27541_release_provenance.py \
  tests/test_v27541_calibration_baseline_pin.py \
  tests/test_v27541_deepx_calibration_1000_profile_preflight.py \
  tests/test_v27541_deepx_calibration_size_canary.py \
  tests/test_v60j_complete_dataset_provisioning.py \
  tests/test_v269e_quality_export_vendored.py \
  tests/test_deepx_full_preprocessing_ab_contract.py \
  tests/test_v27540_deepx_preprocessing_probe.py \
  tests/test_v27540_deepx_preprocessing_ab_pairing.py \
  tests/test_v27540_ready_deepx_preprocessing_ab_profiles.py \
  tests/test_deepx_full_only_quality_dispatch.py \
  tests/test_v27538_full_only_performance_matrix.py \
  tests/test_v27539_remote_premutation_dispatch.py \
  tests/test_v27539_remote_dispatch_propagation.py

printf '4/4 current release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27541_smoke

printf '%s\n' \
  'PASS v2.75.41 small acceptance' \
  'SKIP hardware, SSH, DX-COM, TensorRT build, Energy and inference'
