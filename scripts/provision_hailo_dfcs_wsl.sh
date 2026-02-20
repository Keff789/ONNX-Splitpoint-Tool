#!/usr/bin/env bash

# Provision *managed* Hailo DFC virtualenvs from wheels bundled in the repo.
#
# Put your DFC wheels here (inside the repo):
#   onnx_splitpoint_tool/resources/hailo/hailo8/*.whl
#   onnx_splitpoint_tool/resources/hailo/hailo10/*.whl
#
# Then run (inside Linux/WSL):
#   ./scripts/provision_hailo_dfcs_wsl.sh --all
#
# Python requirement:
#   The managed DFC venvs must be created with Python >= 3.10 (DFC deps like jax
#   have no wheels for Python 3.8).
#
# Convenience:
#   If you created a repo-local .venv with uv/pyenv (Python 3.10+), this script
#   will auto-use it when you do not pass --python explicitly.
#
# This calls:
#   python3 -m onnx_splitpoint_tool.hailo.dfc_provision ...

set -euo pipefail

# Ensure we're in the repo root (script lives in ./scripts)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Auto-select a Python>=3.10 for venv creation if the user did not specify one.
has_python=0
for arg in "$@"; do
  if [[ "${arg}" == "--python" || "${arg}" == --python=* ]]; then
    has_python=1
    break
  fi
done

extra_args=()
if [[ "${has_python}" -eq 0 ]]; then
  cand=""
  if [[ -n "${DFC_PYTHON:-}" ]]; then
    cand="${DFC_PYTHON}"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    # Common when using uv: uv venv --python 3.10
    cand="${REPO_ROOT}/.venv/bin/python"
  elif command -v python3.10 >/dev/null 2>&1; then
    cand="$(command -v python3.10)"
  fi

  if [[ -n "${cand}" ]]; then
    extra_args=(--python "${cand}")
  fi
fi

python3 -m onnx_splitpoint_tool.hailo.dfc_provision "${extra_args[@]}" "$@"
