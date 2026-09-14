#!/usr/bin/env bash
# Compact, hardware-independent v2.75.49 DeepX-Full closeout acceptance.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v27549_small_acceptance.sh was sourced; run it with bash.' >&2
  return 0
fi

set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  PYTHON="$SOURCE_ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf '1/4 release identity and frozen profile\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27549_smoke

printf '2/4 source integrity\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed >/dev/null
printf 'PASS source integrity\n'

printf '3/4 DeepX Full binding and clean maintenance release\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27549_deepx_full_semantic_registry_binding.py \
  tests/test_clean_source_allowlist.py

printf '4/4 lightweight syntax\n'
bash -n scripts/run_v27549_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh
"$PYTHON" -B - <<'PY'
from pathlib import Path

for relative_path in (
    "onnx_splitpoint_tool/__init__.py",
    "onnx_splitpoint_tool/v27549_smoke.py",
    "onnx_splitpoint_tool/workflow/runner.py",
):
    path = Path(relative_path)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("PASS lightweight syntax")
PY

printf '%s\n' \
  'PASS v2.75.49 small acceptance' \
  'SKIP CPU B500 inference and accelerator hardware in this local gate'
