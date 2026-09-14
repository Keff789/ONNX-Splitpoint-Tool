#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.45 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27545_small_acceptance.sh was sourced.' \
    'Run: bash scripts/run_v27545_small_acceptance.sh' >&2
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

printf '1/4 identity, syntax and v2.75.45 large-audit admission contract\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.45-gui-large-audit-trt-working-set-admission"
assert tool.__version__ == "2.75.45", tool.__version__
assert tool.__release__ == "2.75.45", tool.__release__
assert tool.__development_lineage__ == "v2.75.45"
assert tool.__build_id__ == expected
assert WORKFLOW_VERSION == expected
assert {
    "gui_large_audit_start_confirmation",
    "audit_minimum_valid_bound",
    "gui_explicit_deepx_classification_preprocessing",
    "large_audit_active_trt_working_set_admission",
    "retained_trt_cache_budget_separation",
    "large_audit_resume_cache_reuse",
} <= set(tool.__build_features__)
for path in (
    Path("scripts/run_v27544_small_acceptance.sh"),
    Path("scripts/run_v27545_small_acceptance.sh"),
    Path("onnx_splitpoint_tool/v27544_smoke.py"),
    Path("onnx_splitpoint_tool/v27545_smoke.py"),
    Path("tests/test_v27545_large_audit_working_set_admission.py"),
    Path("tests/test_v27545_release_provenance.py"),
    Path("TESTANLEITUNG_2.75.45.md"),
    Path("VERSION_2.75.45_BUILD_AND_TEST_REPORT.md"),
):
    assert path.is_file(), path
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27545_small_acceptance.sh \
  scripts/run_v27544_small_acceptance.sh \
  scripts/run_v27543_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh

printf '2/4 installed-source integrity (retained user profiles permitted)\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed

printf '3/4 focused GUI, audit, cache, resume and release tests\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27545_large_audit_working_set_admission.py \
  tests/test_v27545_release_provenance.py \
  tests/test_v27544_release_provenance.py \
  tests/test_v27543_release_provenance.py \
  tests/test_v27542_release_provenance.py \
  tests/test_profile_editor_campaign_roundtrip_v261e.py \
  tests/test_v60n_run_modes.py \
  tests/test_v264_execution_plan.py \
  tests/test_v27530_score_independent_native_ranking_audit.py \
  tests/test_v27516_primary_remote_error_cache_policy.py \
  tests/test_v27516_read_only_run_admission.py \
  tests/test_v27539_remote_premutation_dispatch.py \
  tests/test_v2713_resume_artifact_contract.py \
  tests/test_v2713_resume_artifact_rehydration.py \
  tests/test_v2713_resume_preparation.py \
  tests/test_v2713_resume_remote_rehydration.py \
  tests/test_v2757_cache_canary_handoff.py \
  tests/test_v2757_cache_verify_only.py \
  tests/test_deepx_full_preprocessing_ab_contract.py \
  tests/test_v27540_ready_deepx_preprocessing_ab_profiles.py

printf '4/4 current release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27545_smoke

printf '%s\n' \
  'PASS v2.75.45 small acceptance' \
  'SKIP hardware, SSH, DX-COM, TensorRT build, Energy and inference'
