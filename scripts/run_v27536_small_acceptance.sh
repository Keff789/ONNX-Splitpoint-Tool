#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.36 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27536_small_acceptance.sh was sourced.' \
    'Run: bash scripts/run_v27536_small_acceptance.sh' >&2
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

printf '1/4 identity, syntax and 2.75.36 release features\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.36-singleton-validator-trt-cache-energy-na-repair-r2"
assert tool.__version__ == "2.75.36", tool.__version__
assert tool.__release__ == "2.75.36", tool.__release__
assert tool.__development_lineage__ == "v2.75.36"
assert tool.__build_id__ == expected
assert WORKFLOW_VERSION == expected
assert {
    "native_singleton_shape_validator_equivalence",
    "stable_trt_engine_cache_namespace",
    "verified_legacy_trt_engine_cache_migration",
    "jetson_cuda_driver_trt_abi_attestation",
    "selected_gpu_trt_engine_cache_identity",
    "warm_trt_retention_metadata_fast_path",
    "quality_first_no_build_receipt_preserving_reuse",
    "disabled_energy_not_applicable_evidence",
    "packaged_resnet_v27536_acceptance_profile",
    "frozen_audit_rejection_continuation",
    "profile_selection_budget_roundtrip",
    "packaged_resnet_small_acceptance_profile",
    "hailo10_singleton_boundary_layout_resolution",
    "native_split_quality_case_failure_isolation",
    "frozen_audit_execution_union_generation",
    "frozen_audit_union_identity_preflight",
    "audit_scoped_standard_development_ranking",
    "model_validation_summary_debug_pack",
    "score_independent_audit_execution_plan",
    "native_completed_task_ranking_strata",
    "truthful_native_ranking_audit_status",
    "known_alias_debug_pack_deduplication",
} <= set(tool.__build_features__)
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27536_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/run_fresh_standard_workflow.sh \
  scripts/update_source_release.sh

printf '2/4 read-only source manifest\n'
"$PYTHON" -B scripts/build_source_manifest.py --root "$SOURCE_ROOT" --verify

printf '3/4 frozen audit union, ranking and debug-pack contracts\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_frozen_audit_generation_scope.py \
  tests/test_v264_execution_plan.py \
  tests/test_evaluation_role_alias_normalization.py \
  tests/test_compact_debug_pack.py \
  tests/test_v27534_hailo10_singleton_boundary_isolation.py \
  tests/test_v269f_hailo_trt_interface_contract.py \
  tests/test_v27516_primary_remote_error_cache_policy.py \
  tests/test_v269f_native_split_quality_runtime_e2e.py \
  tests/test_v27536_quality_first_real_builder_reuse.py \
  tests/test_v27536_energy_not_applicable_evidence.py \
  tests/test_v27530_score_independent_native_ranking_audit.py \
  tests/test_profile_editor_campaign_roundtrip_v261e.py \
  tests/test_v27536_ready_acceptance_profile.py \
  tests/test_v27529_final_quality_standard_path.py \
  tests/test_v275_release_contract.py \
  tests/test_v27522_release_provenance.py \
  tests/test_v273_version_provenance.py \
  tests/test_v273_local_acceptance_harness.py \
  tests/test_clean_source_allowlist.py

printf '4/4 retained release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v272_smoke

printf '%s\n' \
  'PASS v2.75.36 small acceptance' \
  'SKIP hardware, Hailo DFC, DX-COM, TensorRT build, SSH, Energy and inference'
