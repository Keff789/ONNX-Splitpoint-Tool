#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Controller requires only the Python standard library. Workers use the DFC venvs.
exec python3 -I -B "$HERE/collect.py" "$@"
