#!/usr/bin/env bash
# Fast, local and hardware-free v2.75.28 acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27528_small_acceptance.sh was sourced.' \
    'Run: bash scripts/run_v27528_small_acceptance.sh' >&2
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

printf '1/4 identity, syntax and Final bootstrap features\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path
import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.28-final-campaign-bootstrap-energy-provenance-repair"
assert tool.__version__ == "2.75.28", tool.__version__
assert tool.__release__ == "2.75.28", tool.__release__
assert tool.__development_lineage__ == "v2.75.28", tool.__development_lineage__
assert tool.__build_id__ == expected, tool.__build_id__
assert WORKFLOW_VERSION == expected, WORKFLOW_VERSION
for feature in (
    "campaign_preflight_success_exit_contract",
    "exact_yolov7_paper_final_template",
    "runtime_exact_campaign_contract_templates",
    "yolov7_paper_final_canary",
    "inherited_energy_method_runtime_binding_gate",
):
    assert feature in tool.__build_features__, feature
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27528_small_acceptance.sh \
  scripts/run_v27528_yolov7_final_canary.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh

printf '2/4 read-only source manifest\n'
"$PYTHON" -B scripts/build_source_manifest.py --root "$SOURCE_ROOT" --verify

printf '3/4 focused v2.75.28 and retained Final contracts\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27528_campaign_preparation.py \
  tests/test_v27528_energy_inherited_runtime_binding.py \
  tests/test_v27528_template_canary.py \
  tests/test_v27527_claim_scope.py \
  tests/test_evaluation_role_alias_normalization.py \
  tests/test_v27527_matrix_canary.py \
  tests/test_v27516_primary_remote_error_cache_policy.py \
  tests/test_v60n_run_modes.py \
  tests/test_profile_editor_campaign_roundtrip_v261e.py \
  tests/test_v270f_version_provenance.py \
  tests/test_v275_release_contract.py \
  tests/test_v273_local_acceptance_harness.py \
  tests/test_clean_source_allowlist.py

printf '4/4 retained release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v272_smoke

printf '%s\n' \
  'PASS v2.75.28 small acceptance' \
  'SKIP hardware, Hailo DFC, DX-COM, TensorRT build, SSH and inference'
