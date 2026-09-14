#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.40 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27540_small_acceptance.sh was sourced.' \
    'Run: bash scripts/run_v27540_small_acceptance.sh' >&2
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

printf '1/4 identity, syntax and v2.75.40 feature contract\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = (
    "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair"
)
assert tool.__version__ == "2.75.40", tool.__version__
assert tool.__release__ == "2.75.40", tool.__release__
assert tool.__development_lineage__ == "v2.75.40"
assert tool.__build_id__ == expected
assert WORKFLOW_VERSION == expected
assert {
    "deepx_classification_preprocessing_ab",
    "deepx_imagenet_mean_std_build_adapter",
    "deepx_full_v2_cache_contract",
    "deepx_preprocessing_ab_paired_cohort_lock",
    "deepx_preprocessing_ab_isolated_cache",
    "deepx_preprocessing_float_ort_probe",
    "deepx_preprocessing_ab_pairing_verifier",
    "full_only_tensorrt_physical_alias_precedence",
    "packaged_resnet_v27540_deepx_preprocessing_ab_profiles",
} <= set(tool.__build_features__)
for path in (
    Path("profiles/resnet50_v27540_deepx_preprocess_a_current_scale_only.yaml"),
    Path("profiles/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std.yaml"),
    Path("scripts/deepx_classification_preprocessing_probe.py"),
    Path("scripts/verify_v27540_deepx_preprocessing_ab.py"),
):
    assert path.is_file(), path
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27540_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh

printf '2/4 read-only source manifest\n'
"$PYTHON" -B scripts/build_source_manifest.py --root "$SOURCE_ROOT" --verify

printf '3/4 focused alias, DeepX A/B, profile and provenance tests\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v269e_quality_export_vendored.py \
  tests/test_deepx_full_preprocessing_ab_contract.py \
  tests/test_v27540_deepx_preprocessing_probe.py \
  tests/test_v27540_deepx_preprocessing_ab_pairing.py \
  tests/test_v27540_ready_deepx_preprocessing_ab_profiles.py \
  tests/test_v27540_release_provenance.py \
  tests/test_deepx_full_only_quality_dispatch.py \
  tests/test_v27538_full_only_performance_matrix.py \
  tests/test_v27539_remote_premutation_dispatch.py \
  tests/test_v27539_remote_dispatch_propagation.py

printf '4/4 current release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27540_smoke

printf '%s\n' \
  'PASS v2.75.40 small acceptance' \
  'SKIP hardware, SSH, DX-COM, TensorRT build, Energy and inference'
