#!/usr/bin/env bash
set -Eeuo pipefail
GATE_TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$GATE_TOOL/.venv/bin/python" -I -B "$GATE_TOOL/scripts/reference_workflow_gate_v2804.py" "$@"
