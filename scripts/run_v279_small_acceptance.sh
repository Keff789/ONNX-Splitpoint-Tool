#!/usr/bin/env bash
# Compatibility launcher for the current release acceptance (historical alias).

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'SKIP: run_v279_small_acceptance.sh was sourced; run it with bash.' >&2
  return 0
fi

set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$ROOT/scripts/run_v282_small_acceptance.sh" "$@"
