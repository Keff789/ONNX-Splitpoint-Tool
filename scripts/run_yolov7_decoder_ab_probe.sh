#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="${OSP_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  printf 'STOP: Python environment not found: %s\n' "$PY" >&2
  exit 66
fi

"$PY" -m onnx_splitpoint_tool.dependency_bootstrap \
  --groups yolov7_probe \
  --python "$PY"

exec "$PY" -B "$ROOT/scripts/probe_yolov7_decoder_ab.py" "$@"
