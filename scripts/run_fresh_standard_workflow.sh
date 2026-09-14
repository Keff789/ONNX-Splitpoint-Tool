#!/usr/bin/env bash
# Start one fresh Standard run through the normal Evaluation Workflow.

# Keep the guard before shell options and assignments so sourcing is a no-op.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' \
    "SKIP: scripts/run_fresh_standard_workflow.sh was sourced." \
    "Run it in its own process with: bash scripts/run_fresh_standard_workflow.sh --profile PROFILE" >&2
  return 0
fi

set -Eeuo pipefail

usage() {
  printf '%s\n' \
    "Usage: bash scripts/run_fresh_standard_workflow.sh --profile PATH [--out PATH] [--only-model ID] [--plan]" \
    "" \
    "Starts the existing EvaluationWorkflowRunner once with the resolved Standard" \
    "profile snapshot. No resume, result import, partial stage, direct SSH, suite" \
    "copy, native runner, or separate energy runner is implemented here."
}

main() {
  local source_root
  local python_bin
  local profile_path=""
  local out_root="$HOME/Models/EvaluationRuns"
  local only_model=""
  local plan_only=0
  local -a command

  source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
  python_bin="${PY:-python3}"

  while (( $# > 0 )); do
    case "$1" in
      --profile)
        (( $# >= 2 )) || { printf 'ERROR: --profile needs a path.\n' >&2; return 2; }
        profile_path="$2"
        shift 2
        ;;
      --out)
        (( $# >= 2 )) || { printf 'ERROR: --out needs a path.\n' >&2; return 2; }
        out_root="$2"
        shift 2
        ;;
      --only-model)
        (( $# >= 2 )) || { printf 'ERROR: --only-model needs an id.\n' >&2; return 2; }
        only_model="$2"
        shift 2
        ;;
      --plan)
        plan_only=1
        shift
        ;;
      --help|-h)
        usage
        return 0
        ;;
      *)
        printf 'ERROR: unsupported option: %s\n' "$1" >&2
        usage >&2
        return 2
        ;;
    esac
  done

  if [[ -z "$profile_path" ]]; then
    printf 'ERROR: --profile is required.\n' >&2
    return 2
  fi
  if [[ ! -f "$profile_path" ]]; then
    printf 'ERROR: Evaluation profile not found: %s\n' "$profile_path" >&2
    return 2
  fi
  if [[ "$python_bin" == */* ]]; then
    [[ -x "$python_bin" ]] || {
      printf 'ERROR: Python is not executable: %s\n' "$python_bin" >&2
      return 2
    }
  else
    command -v -- "$python_bin" >/dev/null || {
      printf 'ERROR: Python command not found: %s\n' "$python_bin" >&2
      return 2
    }
  fi

  command=(
    "$python_bin" -B -m onnx_splitpoint_tool.workflow.run_evaluation
    --profile-driven
    --require-run-mode standard
    --require-fresh-run
    --profile "$profile_path"
    --out "$out_root"
    --execution-mode generate_and_run
  )
  if [[ -n "$only_model" ]]; then
    command+=(--only-model "$only_model")
  fi

  if (( plan_only )); then
    printf 'Standard workflow command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi

  cd -- "$source_root"
  exec "${command[@]}"
}

main "$@"
