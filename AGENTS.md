# ONNX Splitpoint Tool – Codex Working Rules

## Project role

This repository contains the ONNX Splitpoint Tool.

Smartmirror2 is the x86 Linux controller.

The controller:
- contains the main repository and Python environment;
- performs central analysis and orchestration;
- connects to accelerator systems via SSH.

Remote accelerator systems include:
- Jetson Orin NX + Hailo8;
- Jetson Orin NX + Hailo10H;
- Jetson Orin NX + DeepX M1;
- TensorRT execution on the appropriate Jetson.

Do not assume that Smartmirror2 itself is a Jetson.

## Source of truth

Before making changes:

1. Read the implementation plan supplied by the user.
2. Inspect the relevant current source.
3. Inspect relevant tests and current release documentation.
4. Check `git status`.
5. Do not silently expand the requested scope.

When an implementation plan exists, treat it as the task contract.

Do not replace explicitly requested behavior with a different design unless a
concrete technical blocker is found. Report such blockers before broad redesign.

## Git safety

The baseline before Codex integration is tagged:

`baseline-v2.82-smartmirror2-20260914`

Never:
- rewrite existing history;
- use `git reset --hard`;
- use `git clean -fdx`;
- force-push;
- delete untracked runtime data;
- remove user profiles or caches as a shortcut.

Before implementation:
- record `git status --short`;
- record the current commit.

After implementation:
- provide `git status --short`;
- provide `git diff --stat`;
- summarize changed files.

Do not commit or push unless explicitly requested.

## Local runtime data

The following are machine-local/runtime state and must not be treated as normal
source files:

- `.venv/`
- `.venv-report/`
- `.install_logs/`
- `logs/`
- live user profiles under `profiles/`
- models and datasets outside the repository;
- backend artifact caches;
- local Hailo/DeepX compiler environments;
- evaluation runs and debug packs.

Do not delete, normalize or recreate these unless the task explicitly requires it.

## Build and cache policy

Force rebuild is OFF.

Always prefer reuse of an existing valid artifact.

Build only artifacts that are genuinely required and missing.

Do not:
- rebuild merely because the build device or GPU preference changed;
- delete working HEFs, DXNNs, TensorRT engines or receipts to force a test;
- introduce a new cache, hash, seal, registry or identity system unless explicitly
  required by the implementation plan;
- select artifacts based on better observed accuracy.

Known deterministic negative compile evidence must not be converted into repeated
cold builds.

## Testing strategy

Use the smallest sufficient test first.

Preferred order:

1. static/source inspection;
2. focused unit tests;
3. relevant regression tests;
4. existing release short/small acceptance tests when appropriate;
5. one targeted real hardware smoke;
6. larger hardware workflows only when explicitly required.

Do not start an overnight run or full evaluation campaign unless explicitly
requested.

Do not treat a large test count as proof that the real hardware path works.

## Hardware execution

Hardware tests must use the existing product architecture and existing runners.

Do not invent ad-hoc SSH implementations when a normal project path exists.

Before running a hardware test, state:
- exact command;
- target setup;
- backend;
- whether a build can occur;
- expected writes;
- approximate scope/runtime.

Do not start hardware work that can trigger:
- an unplanned cold build;
- destructive cleanup;
- a full campaign;
- force rebuild;
without explicit approval.

Prefer existing bounded hardware smokes.

## SSH and remote systems

Use the repository's existing SSH/configuration infrastructure.

Do not:
- modify SSH keys;
- modify remote user shell configuration;
- change system packages or drivers;
- change CUDA/DFC/DeepX installations;
- reboot machines;
- globally kill processes;
- delete remote caches;
unless explicitly required.

Only terminate processes created by the current controlled test, and use the
project's process ownership/lease mechanisms when available.

## Scientific result policy

A technically valid quality FAIL is a valid result.

Do not modify:
- quality margins;
- reference datasets;
- bootstrap policy;
- split selection;
- seeds;
- model-specific compiler settings;
simply to turn a FAIL into PASS.

Distinguish:
- implementation bugs;
- runtime/technical failures;
- compiler infeasibility;
- quality FAIL;
- INCONCLUSIVE;
- cancelled/incomplete work.

Known implementation bugs must be fixed or reported as technical limitations;
they must not be relabeled as compiler quality loss.

## Energy measurement

Do not start new energy measurements unless required by the implementation plan.

Do not silently change:
- measurement duration;
- repetitions;
- calibration;
- physical measurement scope;
- idle subtraction;
- power/clock settings.

Existing short screening measurements must not be described as long final
measurement evidence.

## Working style

For normal repository exploration, freely use read-only commands such as:
- `pwd`
- `ls`
- `find`
- `rg`
- `grep`
- `sed`
- `cat`
- `head`
- `tail`
- `git status`
- `git diff`
- `git log`
- `git show`

Do not ask the user to manually inspect information that can safely be inspected
locally.

When implementing:
- make the smallest coherent change;
- add or update regression tests for the actual failure;
- preserve backward compatibility unless the implementation plan says otherwise;
- do not refactor unrelated code.

## Implementation-plan workflow

For substantial fixes, the implementation plan is the authoritative task scope.

Expected sequence:

1. Read the complete implementation plan.
2. Verify the relevant current implementation.
3. Implement the requested changes.
4. Run the planned focused local tests.
5. Analyze failures before increasing test scope.
6. Run only the targeted hardware smokes specified or justified by the plan.
7. Do not escalate automatically to a full evaluation run.
8. Produce a completion report.

If an unexpected hardware failure appears, investigate the concrete cause.
Do not respond by blindly rebuilding artifacts, deleting caches, widening the
test scope, or changing scientific acceptance criteria.

## Completion report

At the end of an implementation task report:

1. root cause found;
2. files changed;
3. implementation performed;
4. tests executed and exact results;
5. hardware tests executed and exact results;
6. tests intentionally not executed and why;
7. remaining limitations;
8. `git status`;
9. recommended next action.

Never claim a hardware PASS when only software tests were run.
