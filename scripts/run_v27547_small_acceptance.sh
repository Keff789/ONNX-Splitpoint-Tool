#!/usr/bin/env bash
# Hardware-independent v2.75.47 decoder/debug-pack acceptance. Run; do not source.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: scripts/run_v27547_small_acceptance.sh was sourced.' \
    'Run it with: bash scripts/run_v27547_small_acceptance.sh' >&2
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

printf '1/5 identity, syntax and v2.75.47 integrated contracts\n'
"$PYTHON" -B - <<'PYTHON'
import ast
from pathlib import Path

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

expected = "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
assert tool.__version__ == "2.75.47", tool.__version__
assert tool.__release__ == "2.75.47", tool.__release__
assert tool.__development_lineage__ == "v2.75.47"
assert tool.__build_id__ == expected
assert WORKFLOW_VERSION == expected
assert {
    "model_bound_yolov7_anchor_contract",
    "official_coco_yolov7_decoder_ab_probe",
    "generic_native_yolov7_decoder_parity",
    "debug_pack_ranking_audit_intent_fix",
    "debug_pack_non_audit_completeness",
    "early_declared_model_sha256_admission",
} <= set(tool.__build_features__)
for path in (
    Path("profiles/yolov7_paper_v27547_standard_anchor_b500.yaml"),
    Path("scripts/probe_yolov7_decoder_ab.py"),
    Path("scripts/run_yolov7_decoder_ab_probe.sh"),
    Path("scripts/run_v27547_small_acceptance.sh"),
    Path("onnx_splitpoint_tool/v27547_smoke.py"),
    Path("tests/test_v27547_yolov7_decoder_contract.py"),
    Path("tests/test_v27547_model_hash_gate.py"),
    Path("tests/test_v27547_release_integration.py"),
    Path("tests/test_v27547_release_provenance.py"),
    Path("TESTANLEITUNG_2.75.47.md"),
    Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
):
    assert path.is_file(), path
for root in (Path("onnx_splitpoint_tool"), Path("scripts"), Path("tests")):
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("PASS identity/syntax")
PYTHON
bash -n \
  scripts/run_v27547_small_acceptance.sh \
  scripts/run_yolov7_decoder_ab_probe.sh \
  scripts/run_v27546_small_acceptance.sh \
  scripts/run_local_acceptance.sh \
  scripts/update_source_release.sh

printf '2/5 installed-source integrity (retained user profiles permitted)\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed

printf '3/5 decoder, parity, debug-pack and release integration tests\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27547_yolov7_decoder_contract.py \
  tests/test_v27547_model_hash_gate.py \
  tests/test_compact_debug_pack.py \
  tests/test_v27547_release_integration.py \
  tests/test_v27547_release_provenance.py \
  tests/test_v272_yolov7_head_mapping.py \
  tests/test_v270l_completed_endpoint_v2.py \
  tests/test_v272_detection_completion_runtime.py

printf '4/5 retained v2.75.46 evidence-closure regressions\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27546_hailo_onnx_interpreter.py \
  tests/test_v27546_calibration_size_canary_output_contract.py \
  tests/test_v27546_standard_quality_projection.py \
  tests/test_v27546_release_provenance.py

printf '5/5 current release smoke\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27547_smoke

printf '%s\n' \
  'PASS v2.75.47 small acceptance' \
  'SKIP CPU B500 inference, hardware, SSH, DX-COM, TensorRT build and Energy' \
  'NEXT mandatory gate: official-COCO YOLOv7 decoder A/B probe; only then GUI Start'
