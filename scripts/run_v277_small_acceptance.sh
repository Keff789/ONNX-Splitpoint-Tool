#!/usr/bin/env bash
# Compact, hardware-independent v2.77.15 acceptance.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v277_small_acceptance.sh was sourced; run it with bash.' >&2
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

printf '1/4 release identity\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v277_smoke

printf '2/4 focused regressions\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v277_release_provenance.py \
  tests/test_v270l_completed_endpoint_v2.py \
  tests/test_v27713_backend_semantic_smoke.py \
  tests/test_v27713_hailo8_artifact_canary.py \
  tests/test_v277_cut_bytes_workflow_freeze.py \
  tests/test_v276_offline_scientific_replay.py \
  tests/test_v276_reporting_identity.py \
  tests/test_v276_endpoint_lifecycle.py \
  tests/test_v276_quality_projection.py \
  tests/test_v276_pipeline_repairs.py \
  tests/test_v276_native_supplement_replay.py \
  tests/test_v276_cross_runner_reporting.py \
  tests/test_v276_scientific_development_leader.py \
  tests/test_v27550_scientific_report_fixes.py \
  tests/test_v27538_aggregate_full_only_quality_status.py \
  tests/test_v27550_parallel_hailo_builds.py \
  tests/test_v27513_hailo_full_contract_overlay.py::test_sparse_yolov7_materialization_normalizes_and_stays_attested \
  tests/test_v2772_hailo_full_scope.py \
  tests/test_v2772_hailo_parallel_canary_profile.py \
  tests/test_v2772_hailo_parallel_canary_verifier.py \
  tests/test_v2774_exact_scope_and_managed_venv.py \
  tests/test_v2776_direct_parse_probe.py \
  tests/test_v2777_phase5_hailo_projection.py \
  tests/test_v2777_yolo11_native_adapter.py \
  tests/test_v2778_missing_full_quality_resume.py \
  tests/test_v2779_targeted_resume_attempt_isolation.py \
  tests/test_v27710_abandoned_resume_recovery.py \
  tests/test_v27710_reuse_cohort_attestation.py \
  tests/test_missing_full_quality_attestation.py \
  tests/test_resume_missing_full_quality_wrapper_hardening.py \
  tests/test_deepx_full_only_quality_dispatch.py \
  tests/test_v27539_remote_dispatch_propagation.py \
  tests/test_v268_endpoint_attestation.py \
  tests/test_v269e_quality_export_vendored.py \
  tests/test_v27519_trt_contract_serialization.py \
  tests/test_v27519_trt_contract_production_chain.py \
  tests/test_scientific_reporting_v60.py \
  tests/test_v60s_build_scheduler.py \
  tests/test_v60u_native_contract_debug_fixes.py \
  tests/test_clean_source_allowlist.py

printf '3/4 source integrity\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed >/dev/null
printf 'PASS source integrity\n'

printf '4/4 lightweight syntax\n'
bash -n scripts/run_v277_small_acceptance.sh \
  scripts/run_v2772_hailo_parallel_build_canary.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh
"$PYTHON" -B - <<'PY'
from pathlib import Path

for relative_path in (
    "onnx_splitpoint_tool/__init__.py",
    "onnx_splitpoint_tool/v277_smoke.py",
    "onnx_splitpoint_tool/backend_semantic_smoke.py",
    "onnx_splitpoint_tool/ranking_methods.py",
    "onnx_splitpoint_tool/workflow/runner.py",
    "onnx_splitpoint_tool/workflow/execution_binding.py",
    "onnx_splitpoint_tool/workflow/missing_full_quality_attestation.py",
    "onnx_splitpoint_tool/workflow/results.py",
    "onnx_splitpoint_tool/workflow/scientific_reporting.py",
    "onnx_splitpoint_tool/workflow/cross_runner_reporting.py",
    "scripts/replay_scientific_reports.py",
    "scripts/replay_native_supplement_reports.py",
    "scripts/preflight_v2772_hailo_parallel_build_canary.py",
    "scripts/verify_v2772_hailo_parallel_build_canary.py",
    "scripts/probe_v2776_yolo26_archived_part1.py",
    "scripts/resume_missing_full_quality.py",
    "scripts/run_backend_semantic_smokes.py",
    "scripts/native_producer_validate_visualize.py",
    "onnx_splitpoint_tool/resources/remote_scripts/native_producer_validate_visualize.py",
    "scripts/run_v27713_hailo8_artifact_canary.py",
    "scripts/verify_v27713_hailo8_canary_payload.py",
    "onnx_splitpoint_tool/native_detection_postprocess.py",
    "onnx_splitpoint_tool/native_output_endpoint.py",
    "onnx_splitpoint_tool/run_modes.py",
    "onnx_splitpoint_tool/workflow/profile_options.py",
    "onnx_splitpoint_tool/workflow/hailo_remote_binding.py",
):
    path = Path(relative_path)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("PASS lightweight syntax")
PY

printf '%s\n' \
  'PASS v2.77.15 small acceptance' \
  'SKIP accelerator compiler and runtime hardware in this local gate'
