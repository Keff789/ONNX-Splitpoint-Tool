#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.46 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27546_small_acceptance.sh was sourced.' \
    'Run: bash scripts/run_v27546_small_acceptance.sh' >&2
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

printf '1/4 identity, syntax and v2.75.46 integrated contracts\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = (
    "v2.75.46-native-full-onnx-attestation-"
    "standard-quality-projection"
)
assert tool.__version__ == "2.75.46", tool.__version__
assert tool.__release__ == "2.75.46", tool.__release__
assert tool.__development_lineage__ == "v2.75.46"
assert tool.__build_id__ == expected
assert WORKFLOW_VERSION == expected
assert {
    "native_full_selected_onnx_interpreter_attestation",
    "native_full_onnx_attestation_exception_diagnostics",
    "standard_setup_local_quality_contract_projection",
    "generic_quality_diagnostic_row_separation",
    "deepx_calibration_size_output_contract_hash_normalization",
    "deepx_calibration_size_semantic_endpoint_invariant",
    "gui_large_audit_start_confirmation",
    "large_audit_active_trt_working_set_admission",
    "retained_trt_cache_budget_separation",
    "large_audit_resume_cache_reuse",
} <= set(tool.__build_features__)
for path in (
    Path("profiles/resnet_yolo26s_yolov7_v27546_standard_anchor_b500.yaml"),
    Path("scripts/run_v27545_small_acceptance.sh"),
    Path("scripts/run_v27546_small_acceptance.sh"),
    Path("onnx_splitpoint_tool/v27545_smoke.py"),
    Path("onnx_splitpoint_tool/v27546_smoke.py"),
    Path("tests/test_v27545_large_audit_working_set_admission.py"),
    Path("tests/test_v27546_hailo_onnx_interpreter.py"),
    Path("tests/test_v27546_calibration_size_canary_output_contract.py"),
    Path("tests/test_v27546_standard_quality_projection.py"),
    Path("tests/test_v27546_release_provenance.py"),
    Path("TESTANLEITUNG_2.75.46.md"),
    Path("VERSION_2.75.46_BUILD_AND_TEST_REPORT.md"),
):
    assert path.is_file(), path
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27546_small_acceptance.sh \
  scripts/run_v27545_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh

printf '2/4 installed-source integrity (retained user profiles permitted)\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed

printf '3/4 focused ONNX, quality projection, canary and v2.75.45 retention tests\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27546_hailo_onnx_interpreter.py \
  tests/test_v27546_calibration_size_canary_output_contract.py \
  tests/test_v27546_standard_quality_projection.py \
  tests/test_v27546_release_provenance.py \
  tests/test_v27545_large_audit_working_set_admission.py \
  tests/test_v27522_quality_reporting.py \
  tests/test_v269d_management_trt_quality_summary.py

printf '4/4 current release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27546_smoke

printf '%s\n' \
  'PASS v2.75.46 small acceptance' \
  'SKIP hardware, SSH, DX-COM, TensorRT build, Energy and inference'
