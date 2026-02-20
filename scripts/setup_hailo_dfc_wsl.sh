#!/usr/bin/env bash

# Hailo Dataflow Compiler (DFC) setup helper for Linux / WSL.
#
# Historical note:
#   This repo used to rely on a single user-managed venv (~/hailo_dfc_venv) and a
#   manual wheel path argument. With Hailo-8 vs Hailo-10 requiring different DFC
#   versions, the tool now supports *managed* per-profile venvs (hailo8/hailo10)
#   via:
#     ./scripts/provision_hailo_dfcs_wsl.sh --all
#
# New recommended usage (no args):
#   ./scripts/setup_hailo_dfc_wsl.sh
#   -> provisions managed venvs from wheels bundled under:
#        onnx_splitpoint_tool/resources/hailo/hailo8/*.whl
#        onnx_splitpoint_tool/resources/hailo/hailo10/*.whl
#
# Legacy usage (still supported):
#   ./scripts/setup_hailo_dfc_wsl.sh /path/to/hailo_dataflow_compiler-*.whl [venv_dir]
#
# Important:
#   Do NOT blindly "pip install --upgrade onnx" in the DFC venv.
#   Hailo DFC wheels pin onnx/protobuf versions (e.g. DFC 3.33 pins onnx==1.16.0
#   and protobuf==3.20.3). Upgrading ONNX often upgrades protobuf and breaks the SDK.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# New mode: no args (or option-style args) -> delegate to managed provisioning.
if [[ $# -eq 0 ]]; then
  echo "[INFO] No wheel argument given. Provisioning managed DFC venvs from repo resources…"
  exec "${SCRIPT_DIR}/provision_hailo_dfcs_wsl.sh" --all
fi

if [[ "${1}" == --* ]]; then
  echo "[INFO] Option-style invocation detected. Delegating to managed provisioning…"
  exec "${SCRIPT_DIR}/provision_hailo_dfcs_wsl.sh" "$@"
fi

# ---------------------------- Legacy mode ----------------------------

WHEEL_PATH="${1:-}"
VENV_DIR="${2:-${HOME}/hailo_dfc_venv}"

if [ -z "${WHEEL_PATH}" ]; then
  echo "Usage: $0 /path/to/hailo_dataflow_compiler-*.whl [venv_dir]" >&2
  echo "Tip: place wheels into the repo instead and run:" >&2
  echo "  ${SCRIPT_DIR}/provision_hailo_dfcs_wsl.sh --all" >&2
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

echo "[INFO] Running pip check…"
python3 -m pip check || true

echo "[INFO] Sanity import…"
python3 -c "import hailo_sdk_client, onnx, google.protobuf; print('OK', getattr(hailo_sdk_client,'__version__','?'), onnx.__version__, google.protobuf.__version__)" || true

echo "[OK] DFC installed. Activate later with:"
echo "  source "${VENV_DIR}/bin/activate""
echo
echo "[TIP] Preferred setup for Hailo-8 + Hailo-10:"
echo "  1) Put wheels into onnx_splitpoint_tool/resources/hailo/hailo8 and hailo10"
echo "  2) Run: ./scripts/provision_hailo_dfcs_wsl.sh --all"
