#!/usr/bin/env bash
set -euo pipefail
TASK_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TASK_PYTHON=${ONNX_SPLITPOINT_PYTHON:-"$TASK_SOURCE/.venv/bin/python"}
if [[ ! -x "$TASK_PYTHON" ]]; then TASK_PYTHON=python3; fi
export ORT_DISABLE_TELEMETRY=1
exec "$TASK_PYTHON" -I -B "$TASK_SOURCE/scripts/classification_probe_v27931/lib/collect_smokes.py" --tool-dir "$TASK_SOURCE" "$@"
