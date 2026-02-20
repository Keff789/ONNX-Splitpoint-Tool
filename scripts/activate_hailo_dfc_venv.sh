#!/usr/bin/env bash

# Source this file (do NOT execute) to activate a Hailo DFC virtualenv.
#
# New (managed) venvs:
#   source ./scripts/activate_hailo_dfc_venv.sh hailo8
#   source ./scripts/activate_hailo_dfc_venv.sh hailo10
#
# Legacy (single venv):
#   source ./scripts/activate_hailo_dfc_venv.sh /path/to/venv_dir
#   (default: ~/hailo_dfc_venv)

arg="${1:-}"

if [[ "${arg}" == "hailo8" ]]; then
  VENV_DIR="${HOME}/.onnx_splitpoint_tool/hailo/venv_hailo8"
elif [[ "${arg}" == "hailo10" ]]; then
  VENV_DIR="${HOME}/.onnx_splitpoint_tool/hailo/venv_hailo10"
elif [[ -n "${arg}" ]]; then
  VENV_DIR="${arg}"
else
  VENV_DIR="${HOME}/hailo_dfc_venv"
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[ERR] venv not found: ${VENV_DIR}" >&2
  echo "      Managed venvs (recommended):" >&2
  echo "        ${HOME}/.onnx_splitpoint_tool/hailo/venv_hailo8" >&2
  echo "        ${HOME}/.onnx_splitpoint_tool/hailo/venv_hailo10" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
echo "[OK] Activated: ${VENV_DIR}"
