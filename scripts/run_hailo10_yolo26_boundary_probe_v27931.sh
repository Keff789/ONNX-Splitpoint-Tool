#!/usr/bin/env bash
set -euo pipefail
export ORT_DISABLE_TELEMETRY=1
PROBE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBE_TOOL_ROOT="${ONNX_SPLITPOINT_TOOL_DIR:-$HOME/ONNX-Splitpoint-Tool}"
if [[ -f "$PROBE_SCRIPT_DIR/hailo10_yolo26_boundary_probe_v27931.py" && -d "$PROBE_SCRIPT_DIR/../onnx_splitpoint_tool" ]]; then
  PROBE_TOOL_ROOT="$(cd -- "$PROBE_SCRIPT_DIR/.." && pwd)"
fi
PROBE_PYTHON="$PROBE_TOOL_ROOT/.venv/bin/python"
if [[ ! -x "$PROBE_PYTHON" ]]; then
  PROBE_PYTHON="$PROBE_TOOL_ROOT/venv/bin/python"
fi
if [[ ! -x "$PROBE_PYTHON" ]]; then
  echo "STOP: Tool-Python fehlt: $PROBE_TOOL_ROOT" >&2
  exit 2
fi
exec "$PROBE_PYTHON" -B "$PROBE_TOOL_ROOT/scripts/hailo10_yolo26_boundary_probe_v27931.py" --source-root "$PROBE_TOOL_ROOT" "$@"
