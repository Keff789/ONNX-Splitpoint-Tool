#!/usr/bin/env bash
# Hardware-independent v2.79.4 acceptance and inherited regression gate.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s
' 'SKIP: run_v2794_small_acceptance.sh was sourced; run it with bash.' >&2
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

printf '1/6 release identity and v2.79.4 contract smoke
'
"$PYTHON" -B -m onnx_splitpoint_tool.v2794_smoke

printf '2/6 v2.79.4 release-line and productized Three-Stage regressions
'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short   tests/test_v2794_release_provenance.py   tests/test_v2793_productized_three_stage.py

printf '3/6 inherited v2.79.2 B500-evidence regressions
'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short   tests/test_v2792_required_scope.py   tests/test_v2792_detection_records.py   tests/test_v2792_logical_join.py   tests/test_v2792_evidence_state.py   tests/test_v2792_scheduler_eta_receipts.py   tests/test_v2792_launcher_status.py   tests/test_v2792_launcher_profile.py   tests/test_v2792_reconciler.py

printf '4/6 Native concurrent runner and inherited focused regressions
'
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
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short "${COMMON_TESTS[@]}"

printf '5/6 source integrity
'
"$PYTHON" -B scripts/build_source_manifest.py --root "$SOURCE_ROOT" --verify --scope installed >/dev/null
printf 'PASS source integrity
'

printf '6/6 lightweight syntax
'
bash -n   scripts/run_v2794_small_acceptance.sh   scripts/run_v2793_small_acceptance.sh   scripts/run_v2792_small_acceptance.sh   scripts/run_v2792_seven_model_long_overnight.sh   scripts/run_local_acceptance.sh   scripts/update_source_release.sh
"$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts tests
printf '%s
'   'PASS v2.79.4 small acceptance'   'SKIP accelerator compiler and runtime hardware in this local gate'
