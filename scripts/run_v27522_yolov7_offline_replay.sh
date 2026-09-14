#!/usr/bin/env bash
# Hardware-free replay of the preserved v2.75.21 YOLOv7 Quality predictions.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "SKIP: scripts/run_v27522_yolov7_offline_replay.sh was sourced." \
    "Run: bash scripts/run_v27522_yolov7_offline_replay.sh" >&2
  return 0
fi

set -Eeuo pipefail

SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEFAULT_RUN_DIR="/home/kmika/Models/EvaluationRuns/yolov7_quality_canary_v27521_20260808_111835_20260808_111900"

if (( $# > 1 )); then
  printf 'Usage: %s [EVALUATION_RUN_DIR]\n' "$0" >&2
  exit 64
fi

RUN_DIR="${1:-$DEFAULT_RUN_DIR}"
if [[ -n "${PY:-}" ]]; then
  PYTHON="$PY"
elif [[ -x "$SOURCE_ROOT/.venv/bin/python" ]]; then
  PYTHON="$SOURCE_ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  printf 'FEHLER: EvaluationRun fehlt: %s\n' "$RUN_DIR" >&2
  exit 66
fi

RUN_DIR="$(readlink -f -- "$RUN_DIR")"
RUN_PARENT="$(dirname -- "$RUN_DIR")"
RUN_NAME="$(basename -- "$RUN_DIR")"
OUT_DIR="$RUN_PARENT/${RUN_NAME}_offline_quality_replay_v27522"

cd -- "$SOURCE_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf '%s\n' \
  "Hardwarefreie Quality-Auswertung aller sechs zentralen Ergebnisse" \
  "RUN_DIR=$RUN_DIR" \
  "OUT_DIR=$OUT_DIR"

exec "$PYTHON" -B scripts/replay_central_quality.py \
  --eval-run-dir "$RUN_DIR" \
  --out-dir "$OUT_DIR" \
  --workers 4
