# Closed Generic source collection recovery

`scripts/recover_generic_closed_collection.py` recovers collection only. It never
starts a benchmark, compiler, inference, FPS measurement or statistics worker.
The original EvaluationRun, source capture, status and owner records stay byte
identical; a successful recovery does not change an old failed/partial result.

The default is a local dry-run. `--apply` permits fresh normal leased collection
and a compressed archive of this single remote run. Add `--cleanup` only with
matching explicit operator authorization. The archive is verified by streaming
every tar member against SHA256/size; no full extraction or persistent plain
duplicate is required. Root must admit the temporary remote archive and local
compressed archive capacity before apply. No automatic retry is performed.

The original run must be closed, Generic quality-source-only, N5000, with Native
and Energy disabled, all selected captures `verified_exact`, and its normal
artifact-index terminal closure passing. Scientific request/candidate/report
bytes are checked against the original sealed index. Exact endpoint, subrun,
suite metadata, current registry and released run/physical locks are required.
The original controller PID must be absent; an existing/reused PID without a
recorded birth identity stays blocked. Normal resource-scoped controller and
remote lease observations must prove no associated work before new operations.

An input JSON has schema
`onnx-splitpoint/generic-closed-collection-recovery-input`, schema_version 1 and
`run_id`, `run_dir`, `session_id`, `setup_id`, `model_id`. Its `original` mapping
contains `{path, sha256, size_bytes}` descriptors for:

- `manifest`: original `run_manifest.json`.
- `profile`: archived `profile.yaml`.
- `run_lock`: normal released EvaluationRun lock.
- `dispatch`, `status`: original model/setup remote benchmark JSONs.
- `artifact_index`, `closure`: original index and terminal closure report.
- `transport_metadata`: original `RemoteBenchmarkRuns/.../run_meta.json`.
- `benchmark_set.json`, `benchmark_plan.json`: original downloaded
  `RemoteBenchmarkRuns/.../results/` metadata bytes.

A separate JSON authorization has schema
`onnx-splitpoint/generic-closed-collection-recovery-authorization`,
`schema_version: 1`, `authorized: true`, `action: recover_generic_closed_collection`, all five
identity fields, `operator`, `source`, `authorized_at`, the exact `evidence`
descriptor, `endpoint: {host,user,port}`, and exact `remote_run_dir`.
`cleanup: true` is additionally required for deletion.
The authorization must also bind `capacity_budget: {max_archive_bytes,
local_reserve_bytes, remote_reserve_bytes}` with integer byte values, a positive
archive cap and nonnegative reserves. Root includes other active work and its
retained reserve in the local reserve value. These are transfer budgets, not
scientific limits; no compression ratio is assumed.

The local normal filesystem guard requires cap plus reserve before any remote
mutation. The normal read-only remote storage guard and the archive writer
require the same remote cap plus reserve; the writer enforces compressed bytes
and fresh available blocks during writes. A budget abort removes only its own
new partial archive. Before SCP, actual archive size must fit the authorized
cap and fresh local free space minus reserve. A local dry-run reports its
literal capacity budget and local inspection; it makes no fresh remote-space
claim and performs no remote probe.

```sh
python -B scripts/recover_generic_closed_collection.py \
  --run-dir /absolute/original/run --session-id ORIGINAL_SESSION \
  --setup-id ORIGINAL_SETUP --model-id ORIGINAL_MODEL \
  --evidence-path /absolute/recovery/evidence.json \
  --authorization-path /absolute/recovery/authorization.json \
  --output-dir /absolute/recovery/NEW_OUTPUT
```

Review the dry-run receipt and capacity, then use a different new output with
`--apply` (copy only) or explicit `--apply --cleanup`. All new diagnostics,
lease journals, full archive and `COLLECTION_RECOVERY_RECEIPT.json` live there.
Before collector writes, a leased preflight checks root/ancestor and tree file
types plus original suite hashes. Symlinks and special files block recovery.
Cleanup requires fresh collect rc0, a complete verified archive containing all
original scientific hashes, unchanged original evidence, and a fresh remote
comparison of the entire tree against that archive. Only that exact owned
remote run is removed; caches and other runs are outside scope. Unknown or
unpreserved content blocks cleanup. Unresolved new remote operations use the
existing physical quarantine fence rather than claim release.

The collector now records `SPLITPOINT_COLLECT_ERROR` with the real failing shell
command while retaining its original exit semantics. Historical
`exit_trap_status=1 copy_failed=0` alone does not identify that command and is
not relabelled as a copy success.
