#!/usr/bin/env bash

# Hailo Dataflow Compiler (DFC) setup helper for WSL2 / Ubuntu.
#
# This script creates a virtualenv (default: ~/hailo_dfc_venv) and installs a
# user-provided Hailo DFC wheel.
#
# You must download the DFC wheel yourself from Hailo.
# The wheel is typically named like:
#   hailo_dataflow_compiler-3.x.y-cp38-cp38-linux_x86_64.whl
#
# Usage:
#   ./scripts/setup_hailo_dfc_wsl.sh /path/to/hailo_dataflow_compiler-*.whl [venv_dir]
#
# Tip: if the wheel is downloaded on Windows, it is usually visible in WSL under:
#   /mnt/c/Users/<YOU>/Downloads/

set -euo pipefail

WHEEL_PATH="${1:-}"
VENV_DIR="${2:-${HOME}/hailo_dfc_venv}"

if [ -z "${WHEEL_PATH}" ]; then
  echo "Usage: $0 /path/to/hailo_dataflow_compiler-*.whl [venv_dir]" >&2
  echo "Tip: if you downloaded in Windows, try:" >&2
  echo "  ls /mnt/c/Users/*/Downloads/hailo*_dataflow*_compiler*.whl" >&2
  exit 2
fi

if [ ! -f "${WHEEL_PATH}" ]; then
  echo "[ERR] Wheel not found: ${WHEEL_PATH}" >&2
  exit 2
fi

echo "[INFO] Using wheel: ${WHEEL_PATH}"
echo "[INFO] Target venv: ${VENV_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERR] python3 not found. Install it first:" >&2
  echo "  sudo apt update && sudo apt install -y python3 python3-venv python3-pip" >&2
  exit 2
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[INFO] Creating venv…"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[INFO] Upgrading pip/setuptools…"
python3 -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing Hailo DFC wheel…"
python3 -m pip install "${WHEEL_PATH}"

# Avoid common ONNX IR/version mismatches during translation.
echo "[INFO] Ensuring recent 'onnx' is installed…"
python3 -m pip install --upgrade onnx

echo "[OK] DFC installed. Activate later with:"
echo "  source \"${VENV_DIR}/bin/activate\""
