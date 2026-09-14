#!/usr/bin/env bash
set -Eeuo pipefail

# v2.79.34: the old acceptance contract required forced, uncached compilation
# and measured overlap. Under the reuse policy it cannot establish that claim.
# Stop before creating any run, selecting a compiler or changing cache policy.
printf '%s\n' \
  'CANARY_RESULT=NOT_RUN' \
  'REASON=legacy_force_canary_retired' \
  'The v2772 force-build acceptance launcher is retired in v2.79.34.' \
  'Use the normal GUI/CLI with profiles/resnet50_v2772_hailo_parallel_build_canary.yaml.' \
  'That profile reuses matching artifacts and builds missing artifacts.' \
  'A cache hit does not prove fresh compilation or scheduler overlap.' \
  'The historical verifier remains available to inspect original v2772 evidence.'
exit 2
