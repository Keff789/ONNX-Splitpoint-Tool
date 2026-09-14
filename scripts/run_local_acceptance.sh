#!/usr/bin/env bash
# Compact local acceptance for the current maintenance release.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_local_acceptance.sh was sourced; run it with bash.' >&2
  return 0
fi

set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$SOURCE_ROOT"

if [[ -n "${PY:-}" ]]; then
  export PY
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  export PY="$SOURCE_ROOT/.venv/bin/python"
fi
bash scripts/run_v282_small_acceptance.sh "$@"
