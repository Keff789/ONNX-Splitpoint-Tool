#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec "${PY:-$ROOT/.venv/bin/python}" -I -B "$ROOT/scripts/reference_workflow_gate_v282.py" "$@"
