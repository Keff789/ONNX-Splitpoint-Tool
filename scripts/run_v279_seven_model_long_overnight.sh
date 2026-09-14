#!/usr/bin/env bash
# Current v2.79 maintenance-line compatibility launcher.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v279_seven_model_long_overnight.sh was sourced.' \
    'Run it in its own Bash process.' >&2
  return 0
fi

set -Eeuo pipefail
SCRIPT_FILE="${BASH_SOURCE[0]}"
[[ -f "$SCRIPT_FILE" && ! -L "$SCRIPT_FILE" ]] || {
  printf 'ERROR: Compatibility launcher fehlt oder ist ein Symlink: %s\n' \
    "$SCRIPT_FILE" >&2
  exit 66
}
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_FILE")" && pwd -P)"
CURRENT_LAUNCHER="$SCRIPT_DIR/run_v27917_seven_model_long_overnight.sh"
[[ -f "$CURRENT_LAUNCHER" && ! -L "$CURRENT_LAUNCHER" ]] || {
  printf 'ERROR: Aktueller v2.79.17-Launcher fehlt oder ist unsicher: %s\n' \
    "$CURRENT_LAUNCHER" >&2
  exit 66
}

exec bash "$CURRENT_LAUNCHER" "$@"
