#!/usr/bin/env bash
# v2.75.28 exact-id entry point; reuse the audited warm-cache launcher.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run this script in its own Bash process.' >&2
  return 0
fi

set -Eeuo pipefail

TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export ONNX_SPLITPOINT_CANARY_PROFILE_ID="${ONNX_SPLITPOINT_CANARY_PROFILE_ID:-yolov7_paper_final_contract_canary_v27528}"
exec bash "$TOOL/scripts/run_v27527_yolov7_final_canary.sh" "$@"
