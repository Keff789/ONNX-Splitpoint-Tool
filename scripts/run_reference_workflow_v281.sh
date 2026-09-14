#!/usr/bin/env bash
set -Eeuo pipefail
V281_TOOL="${ONNX_SPLITPOINT_TOOL_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
export PYTHONDONTWRITEBYTECODE=1 ORT_DISABLE_TELEMETRY=1
exec "$V281_TOOL/.venv/bin/python" -I -B "$V281_TOOL/scripts/reference_workflow_gate_v281.py" "$@"
