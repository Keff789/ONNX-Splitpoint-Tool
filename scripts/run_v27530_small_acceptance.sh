#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.30 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27530_small_acceptance.sh was sourced.' \
    'Run: bash scripts/run_v27530_small_acceptance.sh' >&2
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

printf '1/4 identity, syntax and Final Quality Standard-path feature\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.30-score-independent-native-ranking-audit"
assert tool.__version__ == "2.75.30", tool.__version__
assert tool.__release__ == "2.75.30", tool.__release__
assert tool.__development_lineage__ == "v2.75.30", tool.__development_lineage__
assert tool.__build_id__ == expected, tool.__build_id__
assert WORKFLOW_VERSION == expected, WORKFLOW_VERSION
assert "final_quality_standard_path_profile" in tool.__build_features__
assert "score_independent_development_ranking_audit" in tool.__build_features__
assert "quality_gated_native_ranking_by_execution_stratum" in tool.__build_features__
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27530_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/run_fresh_standard_workflow.sh \
  scripts/update_source_release.sh

printf '2/4 read-only source manifest\n'
"$PYTHON" -B scripts/build_source_manifest.py --root "$SOURCE_ROOT" --verify

printf '3/4 focused v2.75.30 Final Quality and retained Standard contracts\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27529_final_quality_standard_path.py \
  tests/test_v27530_score_independent_native_ranking_audit.py \
  tests/test_v60n_run_modes.py \
  tests/test_v2735_standard_execution_path.py \
  tests/test_profile_editor_campaign_roundtrip_v261e.py \
  tests/test_v2741_native_energy_quality_projection.py \
  tests/test_v275_workflow_energy_contract.py \
  tests/test_v270f_version_provenance.py \
  tests/test_v275_release_contract.py \
  tests/test_v273_local_acceptance_harness.py \
  tests/test_clean_source_allowlist.py

printf '4/4 retained release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v272_smoke

printf '%s\n' \
  'PASS v2.75.30 small acceptance' \
  'SKIP hardware, Hailo DFC, DX-COM, TensorRT build, SSH and inference'
