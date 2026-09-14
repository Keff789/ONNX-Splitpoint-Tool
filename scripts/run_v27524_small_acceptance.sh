#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.24 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "SKIP: scripts/run_v27524_small_acceptance.sh was sourced." \
    "Run: bash scripts/run_v27524_small_acceptance.sh" >&2
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

printf '1/4 identity and syntax\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.24-standard-hef-preupload-gate"
assert tool.__version__ == "2.75.24", tool.__version__
assert tool.__release__ == "2.75.24", tool.__release__
assert tool.__development_lineage__ == "v2.75.24", tool.__development_lineage__
assert tool.__build_id__ == expected, tool.__build_id__
assert WORKFLOW_VERSION == expected, WORKFLOW_VERSION
for feature in (
    "standard_identity_full_only_canary_cache_probe",
    "receipt_attested_hailo_full_preupload_gate",
    "failed_backend_artifact_upload_block",
    "custom_evaluation_profile_update_preservation",
    "run_mode_full_only_canary_preservation",
    "quality_canary_variant_recipe_validation",
    "yolov7_full_only_canary_launcher",
):
    assert feature in tool.__build_features__, feature
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/update_source_release.sh \
  scripts/run_local_acceptance.sh \
  scripts/run_v27524_small_acceptance.sh \
  scripts/run_v27524_yolov7_quality_canary.sh \
  scripts/run_v27524_yolov7_pack_replay.sh \
  scripts/run_v27522_yolov7_offline_replay.sh

printf '2/4 read-only source manifest\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" \
  --verify

printf '3/4 focused v2.75.24 regression contracts\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27522_quality_ap75_replay.py \
  tests/test_v27522_quality_reporting.py \
  tests/test_v27522_full_only_debug_pack.py \
  tests/test_v27522_release_provenance.py \
  tests/test_v27524_canary_launch.py \
  tests/test_v275_release_contract.py \
  tests/test_clean_source_allowlist.py

printf '4/4 release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v273_smoke

printf '%s\n' \
  "PASS v2.75.24 small acceptance" \
  "SKIP hardware, Hailo DFC, DX-COM, TensorRT build, SSH and inference"
