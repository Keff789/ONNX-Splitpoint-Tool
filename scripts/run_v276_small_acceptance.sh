#!/usr/bin/env bash
# Compact, hardware-independent v2.76.2 acceptance.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v276_small_acceptance.sh was sourced; run it with bash.' >&2
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
"$PYTHON" -B -m onnx_splitpoint_tool.v276_smoke

printf '2/4 focused regressions\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v276_release_provenance.py \
  tests/test_v276_offline_scientific_replay.py \
  tests/test_v276_reporting_identity.py \
  tests/test_v276_endpoint_lifecycle.py \
  tests/test_v276_quality_projection.py \
  tests/test_v276_pipeline_repairs.py \
  tests/test_v276_native_supplement_replay.py \
  tests/test_v276_cross_runner_reporting.py \
  tests/test_v276_scientific_development_leader.py \
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
bash -n scripts/run_v276_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh
"$PYTHON" -B - <<'PY'
from pathlib import Path

for relative_path in (
    "onnx_splitpoint_tool/__init__.py",
    "onnx_splitpoint_tool/v276_smoke.py",
    "onnx_splitpoint_tool/workflow/runner.py",
    "onnx_splitpoint_tool/workflow/scientific_reporting.py",
    "onnx_splitpoint_tool/workflow/cross_runner_reporting.py",
    "onnx_splitpoint_tool/benchmark/services.py",
    "onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py",
    "onnx_splitpoint_tool/workflow/endpoint_lifecycle.py",
    "onnx_splitpoint_tool/reporting_quality_decomposition.py",
    "scripts/replay_scientific_reports.py",
    "scripts/replay_native_supplement_reports.py",
):
    path = Path(relative_path)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print("PASS lightweight syntax")
PY

printf '%s\n' \
  'PASS v2.76.2 small acceptance' \
  'SKIP accelerator compiler and runtime hardware in this local gate'
