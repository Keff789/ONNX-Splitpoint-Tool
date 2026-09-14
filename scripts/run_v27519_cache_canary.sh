#!/usr/bin/env bash
# One-command local acceptance plus exact ResNet50/b052/Hailo-8 cache canary.

set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$SOURCE_ROOT/.venv/bin/python"
MODELS_ROOT="${MODELS_ROOT:-$HOME/Models}"
RUNS_ROOT="${RUNS_ROOT:-$MODELS_ROOT/EvaluationRuns}"
RUN_ID="cache_verify_resnet50_b052_hailo8_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="$RUNS_ROOT/$RUN_ID"

if [[ ! -x "$PYTHON" ]]; then
  printf 'FEHLER: Tool-Venv fehlt: %s\n' "$PYTHON" >&2
  exit 69
fi

cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY="$PYTHON" bash scripts/run_v27519_small_acceptance.sh

printf '%s\n' \
  'PASS Kleinabnahme; starte exakt resnet50/b052/hailo8.' \
  'Cache-Miss bleibt compilerfrei und endet sichtbar als cache_miss_blocked.'

workflow_rc=0
if "$PYTHON" -B -m onnx_splitpoint_tool.workflow.run_evaluation \
    --profile profiles/cache_verify_resnet50_b052_hailo8.yaml \
    --models-root "$MODELS_ROOT" \
    --out "$RUNS_ROOT" \
    --run-id "$RUN_ID" \
    --profile-driven \
    --require-run-mode smoke \
    --require-fresh-run \
    --execution-mode generate_and_run; then
  workflow_rc=0
else
  workflow_rc=$?
fi

verify_rc=0
if "$PYTHON" -B scripts/verify_cache_canary_result.py \
    --run-dir "$RUN_DIR"; then
  verify_rc=0
else
  verify_rc=$?
fi

# A diagnostic EvalRun may remain globally partial because Generic, Quality
# and Energy are intentionally not applicable.  The exact Native postcondition
# above is authoritative; cancellation/control errors remain fatal.
if (( verify_rc != 0 || workflow_rc > 1 )); then
  printf 'FAIL cache canary: workflow_rc=%d verify_rc=%d run=%s\n' \
    "$workflow_rc" "$verify_rc" "$RUN_DIR" >&2
  exit 2
fi

printf 'PASS semantic cache canary: %s\n' "$RUN_DIR"
