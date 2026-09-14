#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-v60i_smoke_report.json}"

"${PYTHON_BIN}" -m onnx_splitpoint_tool.v60i_smoke --json "${OUT}"
printf '\nWrote smoke report: %s\n' "${OUT}"
