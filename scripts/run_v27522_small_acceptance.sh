#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.22 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "SKIP: scripts/run_v27522_small_acceptance.sh was sourced." \
    "Run: bash scripts/run_v27522_small_acceptance.sh" >&2
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

REPLAY_FIXTURE="$(mktemp -d -t onnx-splitpoint-v27522-replay-missing-XXXXXXXX)"
cleanup() {
  if [[ -d "$REPLAY_FIXTURE" ]]; then
    rm -r -- "$REPLAY_FIXTURE"
  fi
}
trap cleanup EXIT

cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf '1/5 identity and syntax\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.22-quality-replay-reporting-repair"
assert tool.__version__ == "2.75.22", tool.__version__
assert tool.__release__ == "2.75.22", tool.__release__
assert tool.__development_lineage__ == "v2.75.22", tool.__development_lineage__
assert tool.__build_id__ == expected, tool.__build_id__
assert WORKFLOW_VERSION == expected, WORKFLOW_VERSION
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/update_source_release.sh \
  scripts/run_local_acceptance.sh \
  scripts/run_v27522_small_acceptance.sh \
  scripts/run_v27522_yolov7_offline_replay.sh \
  scripts/run_v27521_cache_canary.sh \
  scripts/run_v27521_full_only_check.sh

printf '2/5 read-only source manifest\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" \
  --verify

printf '3/5 focused v2.75.22 regression contracts\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27522_quality_ap75_replay.py \
  tests/test_v27522_quality_reporting.py \
  tests/test_v27522_full_only_debug_pack.py \
  tests/test_v27522_release_provenance.py \
  tests/test_v275_release_contract.py \
  tests/test_clean_source_allowlist.py

printf '4/5 missing replay-input contract\n'
MISSING_RUN="$REPLAY_FIXTURE/source_run"
mkdir -p -- "$MISSING_RUN/quality_management"
"$PYTHON" -B - "$MISSING_RUN" <<'PYTHON'
import json
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
payload = {
    "schema": "onnx-splitpoint/central-quality-summary",
    "schema_version": 1,
    "status": "ok",
    "request_count": 1,
    "results": [
        {
            "model_id": "yolov7_paper",
            "task": "detection",
            "case_id": "b044",
            "variant": "full",
            "source_run_id": "hailo8",
            "source_setup_id": "orin_nx_hailo8_01",
            "source_request": "models/yolov7_paper/missing/full_request.json",
        }
    ],
}
(run_dir / "quality_management" / "central_quality_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PYTHON
MISSING_LOG="$REPLAY_FIXTURE/missing.log"
if "$PYTHON" -B scripts/replay_central_quality.py \
    --eval-run-dir "$MISSING_RUN" \
    --workers 1 >"$MISSING_LOG" 2>&1; then
  printf 'FAIL: replay unexpectedly accepted missing Prediction JSONs\n' >&2
  exit 1
else
  replay_rc=$?
fi
if [[ "$replay_rc" -ne 2 ]]; then
  printf 'FAIL: missing-input replay rc=%s, expected 2\n' "$replay_rc" >&2
  sed -n '1,120p' "$MISSING_LOG" >&2
  exit 1
fi
grep -F "original EvaluationRun" "$MISSING_LOG" >/dev/null
grep -F "missing required request/prediction paths" "$MISSING_LOG" >/dev/null
test ! -e "$REPLAY_FIXTURE/source_run_offline_quality_replay_v27522"
printf 'PASS missing replay inputs fail closed without output\n'

printf '5/5 release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v273_smoke

printf '%s\n' \
  "PASS v2.75.22 small acceptance" \
  "SKIP hardware, Hailo DFC, DX-COM, TensorRT build, SSH and inference"
