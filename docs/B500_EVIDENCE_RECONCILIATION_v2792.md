# B500 evidence reconciliation and run-state contracts (v2.79.2)

Build/workflow: `v2.79.2-native-concurrent-three-stage-b500-evidence-reconciliation`.

## Motivation

The v2.78.4 seven-model B500 audit completed 407 Quality jobs but its legacy
Central Quality join matched only 12 matrix rows. Read-only reconciliation
showed 551 materialized required identities, 550 present identities, 537 mirror
representations, 386 completed primary Quality results, 21 Full-TensorRT
companions, 163 P2-only Quality-N/A rows, one blocked YOLO11/DeepX Full runtime
identity and one missing YOLO11/Hailo-10 Full identity. The historical run
remains `failed` and is never rewritten or promoted.

## Immutable scope

A run-level symbolic scope is sealed during profile resolution. For each model,
an exact `required_run_scope.json` is generated from the frozen candidate plan
and projected global run IDs before the BenchmarkSet generator or any compiler
can run. Every identity records `requested`, `terminal_outcome_required`,
`success_required`, `quality_applicability`, measurement endpoint and expected
setup/backend. Later failures may change terminal state only; they cannot remove
the identity.

## Logical measurements and mirrors

Direct setup rows and setup-less transport/report mirrors are retained as
separate provenance representations. They form one logical measurement only
when exact request and executable-artifact identities prove the relation. Two
real setup IDs are never deduplicated. DeepX precision aliases are canonicalized
only when the structured contract names the same DXNN SHA-256; different DXNN
bytes remain a conflict. Timing stays on the selected primary representation.

## Central Quality join

The primary join uses exact source-request SHA-256, model/case/run/backend,
logical identity, contract and setup. A direct setup representation outranks a
proven setup-less mirror. Companions such as `native_full_tensorrt` remain a
separate control population and cannot fill the primary matrix. Reports expose
`matrix_required`, `matrix_present`, `quality_applicable`, `quality_completed`,
`quality_blocked`, `quality_not_applicable`, `companions`, `unmatched` and
`ambiguous` as distinct denominators.

The expected reconciled B500 decomposition is 386/386 executable primary
Quality results, 21/21 companions, 163 Quality-N/A technical rows, zero
unmatched and zero ambiguous results.

## Strict detection records

Mini-AP and confidence statistics use one parser supporting exactly two complete
schemas: legacy `box_xyxy/confidence/class_id` and canonical
`x1/y1/x2/y2/score/class_id`. Mixed, partial, nonfinite or invalid records fail
explicitly. Existing detections can no longer silently produce a valid
`predictions=0` result. For the audited YOLOv7/DeepX Full payload this accepts
500 records and 3753 detections; the Central B500 AP50:95 result remains the
authoritative `pass`.

## Hailo attempts and timeout policy

Every decoded and fallback compiler invocation writes immutable start, heartbeat
and terminal receipts. The actual chronologically last terminal attempt is
selected even when it failed or timed out; earlier parser errors remain visible.
Receipts include endpoint, start/end nodes, source/compiler ONNX hashes, phase,
duration, timeout policy, return/raise state, semantic outcome, error class and
log tails. A returned structured failure is never scheduler `ok`.

An explicit Full hard timeout of `0`, `off`, `none`, `unlimited` or `disabled`
disables the hard watchdog for that invocation. Heartbeat, cooperative/manual
cancellation and any separately configured idle-stall policy remain active.

## ETA and launcher status

ETA uses a phase-local baseline, separates classification, detection and
compiler cohorts, waits for warm-up evidence and reports a range. With
insufficient comparable completions it reports `ETA=UNAVAILABLE`. Progress is
authoritative and ETA advisory. The detached launcher writes
`launcher_status.txt` atomically and periodically with phase, worker/monitor
PID, log age, completed/total progress and final workflow RC.

## Read-only replay

`scripts/reconcile_existing_run_v2792.py` writes an external reconciliation
folder and ZIP without inference, compilation, metric recomputation or source
run mutation. It verifies that every source file read remains byte-identical.
