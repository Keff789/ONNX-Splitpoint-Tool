#!/usr/bin/env bash

# Source this file (do NOT execute) to activate the Hailo DFC virtualenv.
#
# Usage:
#   source ./scripts/activate_hailo_dfc_venv.sh
#
# By default the setup script creates:
#   ~/hailo_dfc_venv

VENV_DIR="${1:-${HOME}/hailo_dfc_venv}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[ERR] venv not found: ${VENV_DIR}" >&2
  echo "      Expected something like: ${HOME}/hailo_dfc_venv" >&2
  return 1
fi

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[ERR] activate script not found: ${VENV_DIR}/bin/activate" >&2
  return 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[OK] Activated Hailo DFC venv: ${VENV_DIR}"
python3 -c "import sys; print('[OK] python:', sys.version.split()[0])" 2>/dev/null || true
