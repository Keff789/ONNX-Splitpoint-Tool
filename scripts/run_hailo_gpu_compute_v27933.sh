#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/hailo_gpu_diagnostics/run_hailo_gpu_smoke.sh" "$@"
