#!/usr/bin/env bash
set -Eeuo pipefail
REUSE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REUSE_TOOL_DIR="$(cd -- "$REUSE_SCRIPT_DIR/.." && pwd)"
REUSE_PY="$REUSE_TOOL_DIR/.venv/bin/python"
[[ -x "$REUSE_PY" ]] || { echo "STOP: Tool-Venv fehlt: $REUSE_PY" >&2; exit 2; }
exec "$REUSE_PY" -I -B "$REUSE_SCRIPT_DIR/hailo_reuse_probe_v280.py" "$@"
