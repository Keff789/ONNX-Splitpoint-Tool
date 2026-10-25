# Historical completed-operator Quality import

`workflow.historical_quality_import.import_historical_operator_quality` is a
read-only, explicitly typed import API. It returns copies of existing ledger
rows. Callers retain responsibility for the normal index/merge and publication
transaction. The API neither writes CURRENT nor starts execution or statistics.

Each `HistoricalQualityImport` binds the original accepted Generic FPS result,
original plan, actual Native backing, Quality config, original normal Quality
admission and the Quality producer's own normal row/source authority. Descriptors
are rechecked in the normal source validation scope. The target must already be
an admitted historical H8 Generic row with the exact physical identity and
source authorities. No case numbers are special-cased.

Admission requires both original completion contracts to pass their unchanged
normal verifier. Their scientific operator payloads must agree exactly after
separating validated derived seals, source-observation references, an optional
identical full-model identity and the existing known Rev3 reporting-compatible
native implementation. Only the decoded-NMS/no-second-NMS family is supported.
All other implementation files, source/comparison endpoints, policies, geometry,
input tensors, model and compiled/source artifacts remain exact.

The actual original Native record and three independent Generic repeats must
bind the original plan's contract, input, method, counts and artifacts. Existing
postflight oracle attestations require their canonical self-hash, owned capture,
counts and artifact/input identities. Their context remains
`postflight_oracle_sentinel` with `owned_postflight_after_fifo_drain`. This never
recovers a missing original source-attestation body or creates a current
execution admission; the current execution-attestation verifier is unchanged.

Quality is revalidated against its own producer, request, config/base and
original normal admission, including the normal completed-record verifier. A
consumer observer may move paths only when its bytes and all other proof fields
remain identical. After these checks the API invokes the unchanged normal
`apply_central_quality_result_to_row` projection using the explicit
`historical_completed_operator_quality_import` relation. Both frozen contract
hashes and source-observation references remain separate. It does not claim an
exact-request match. Quality FAIL remains Quality FAIL.

Callers should bind a fresh current ledger snapshot, select exact unique UIDs,
apply this API to the complete ordered rows, then assert that all untargeted
rows and every target's FPS/repeat/provenance fields are unchanged before the
existing normal index/merge transaction. CURRENT adoption belongs to its owner.
This controller-only import does not change frozen hardware/runtime sources or
require a measurement refreeze.
