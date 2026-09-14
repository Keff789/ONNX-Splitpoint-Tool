#!/usr/bin/env bash
# Compact, hardware-independent v2.75.48 anchor-closeout acceptance.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    'SKIP: run_v27548_small_acceptance.sh was sourced; run it with bash.' >&2
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

printf '1/4 release identity and syntax\n'
"$PYTHON" -B -m onnx_splitpoint_tool.v27548_smoke
"$PYTHON" -B -m compileall -q onnx_splitpoint_tool scripts tests
bash -n scripts/run_v27548_small_acceptance.sh scripts/update_source_release.sh

printf '2/4 source integrity\n'
"$PYTHON" -B scripts/build_source_manifest.py \
  --root "$SOURCE_ROOT" --verify --scope installed >/dev/null
printf 'PASS source integrity\n'

printf '3/4 CPU provenance and YOLOv7 completion contracts\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v27548_official_coco_sha.py \
  tests/test_v27547_yolov7_decoder_contract.py \
  tests/test_v272_detection_completion_runtime.py

printf '4/4 DeepX prepared-input completion contract\n'
"$PYTHON" -B -m pytest -q -p no:cacheprovider --tb=short \
  tests/test_v2711_direct_bn6_producer_binding.py

printf '%s\n' \
  'PASS v2.75.48 small acceptance' \
  'SKIP CPU B500 inference and accelerator hardware in this local gate'
