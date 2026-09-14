#!/usr/bin/env bash
# Compact, hardware-independent v2.75.50 acceptance.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v27550_small_acceptance.sh was sourced; run it with bash.' >&2
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

printf '1/4 release identity\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27550_smoke

printf '2/4 focused reporting and scheduler regressions\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27550_scientific_report_fixes.py \
  tests/test_v27550_parallel_hailo_builds.py \
  tests/test_scientific_reporting_v60.py \
  tests/test_v60s_build_scheduler.py \
  tests/test_clean_source_allowlist.py

printf '3/4 source integrity\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed >/dev/null
printf 'PASS source integrity\n'

printf '4/4 lightweight syntax\n'
bash -n scripts/run_v27550_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh
"$PYTHON" -B - <<'PY'
from pathlib import Path

for relative_path in (
    "onnx_splitpoint_tool/__init__.py",
    "onnx_splitpoint_tool/v27550_smoke.py",
    "onnx_splitpoint_tool/workflow/runner.py",
    "onnx_splitpoint_tool/workflow/scientific_reporting.py",
    "onnx_splitpoint_tool/workflow/cross_runner_reporting.py",
    "onnx_splitpoint_tool/benchmark/services.py",
):
    path = Path(relative_path)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("PASS lightweight syntax")
PY

printf '%s\n' \
  'PASS v2.75.50 small acceptance' \
  'SKIP accelerator compiler and runtime hardware in this local gate'
