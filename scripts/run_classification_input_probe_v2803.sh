#!/usr/bin/env bash
set -Eeuo pipefail
export ORT_DISABLE_TELEMETRY=1
PROBE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROBE_TOOL_ROOT="${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
if [[ -d "$PROBE_SCRIPT_DIR/../onnx_splitpoint_tool" ]]; then
  PROBE_TOOL_ROOT="$(cd -- "$PROBE_SCRIPT_DIR/.." && pwd -P)"
fi
PROBE_PYTHON="$PROBE_TOOL_ROOT/.venv/bin/python"
[[ -x "$PROBE_PYTHON" ]] || { echo "STOP: vorhandene Tool-Venv fehlt: $PROBE_TOOL_ROOT" >&2; exit 2; }
exec "$PROBE_PYTHON" -I -B "$PROBE_TOOL_ROOT/scripts/classification_input_probe_v2803.py" --tool-dir "$PROBE_TOOL_ROOT" "$@"
