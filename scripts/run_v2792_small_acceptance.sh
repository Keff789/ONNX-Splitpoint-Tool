#!/usr/bin/env bash
# Hardware-independent v2.79.2 acceptance and inherited regression gate.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v2792_small_acceptance.sh was sourced; run it with bash.' >&2
  return 0
fi

set -Eeuo pipefail
SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${PY:-}" ]]; then PYTHON="$PY";
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then PYTHON="$SOURCE_ROOT/.venv/bin/python";
else PYTHON=python3; fi
cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf '1/5 release identity and v2.79.2 contract smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v2792_smoke

printf '2/5 v2.79.2 B500-evidence regressions\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v2792_required_scope.py \
  tests/test_v2792_detection_records.py \
  tests/test_v2792_logical_join.py \
  tests/test_v2792_evidence_state.py \
  tests/test_v2792_scheduler_eta_receipts.py \
  tests/test_v2792_launcher_status.py \
  tests/test_v2792_launcher_profile.py \
  tests/test_v2792_reconciler.py

printf '3/5 Native concurrent runner and inherited focused regressions\n'
COMMON_TESTS=(
  tests/test_v279_release_provenance.py
  tests/test_v2791_concurrent_three_stage_normal_runner.py
  tests/test_v279_native_three_stage.py
  tests/test_v279_launcher_fixes.py
  tests/test_v2783_native_dual_endpoint.py
  tests/test_v2783_completion_tail_canary.py
  tests/test_v2783_build_evidence.py
  tests/test_v2783_runtime_evidence.py
)
ONNX_TESTS=(
  tests/test_v2783_hailo8_first_feasibility.py
  tests/test_v2783_hailo8_evidence_binding.py
  tests/test_v2783_generation_state_atomic.py
  tests/test_v2783_deepx_model_manifest.py
  tests/test_v2783_resnet_quantized_interface_aggregation.py
  tests/test_v2783_yolo11_gate_a_profile.py
  tests/test_v2783_yolo11_hailo8_first_launcher.py
  tests/test_v2783_yolo11_gate_a_output_verifier.py
  tests/test_v2784_seven_model_long_profile.py
  tests/test_v2784_seven_model_overnight_launcher.py
  tests/test_v277_cut_bytes_workflow_freeze.py
  tests/test_v276_cross_runner_reporting.py
  tests/test_v27550_parallel_hailo_builds.py
  tests/test_v2777_yolo11_native_adapter.py
  tests/test_v2778_missing_full_quality_resume.py
  tests/test_missing_full_quality_attestation.py
  tests/test_clean_source_allowlist.py
)
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short "${COMMON_TESTS[@]}"
if "$PYTHON" -B -c 'import onnx' >/dev/null 2>&1; then
  "$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short "${ONNX_TESTS[@]}"
else
  printf '%s\n' \
    'SKIP inherited ONNX-dependent regression subset: onnx is unavailable in this interpreter' \
    'The installed Smartmirror2 venv is expected to execute this subset.'
  "$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
    tests/test_v2784_seven_model_long_profile.py \
    tests/test_v2784_seven_model_overnight_launcher.py \
    tests/test_v277_cut_bytes_workflow_freeze.py \
    tests/test_clean_source_allowlist.py
fi

printf '4/5 source integrity\n'
"$PYTHON" -B scripts/build_source_manifest.py --root "$SOURCE_ROOT" --verify --scope installed >/dev/null
printf 'PASS source integrity\n'

printf '5/5 lightweight syntax\n'
bash -n \
  scripts/run_v2792_small_acceptance.sh \
  scripts/run_v279_small_acceptance.sh \
  scripts/run_v2792_seven_model_long_overnight.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh
"$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts tests
printf '%s\n' \
  'PASS v2.79.2 small acceptance' \
  'SKIP accelerator compiler and runtime hardware in this local gate'
