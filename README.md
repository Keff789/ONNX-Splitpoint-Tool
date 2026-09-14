# ONNX Splitpoint Tool v2.82

Aktueller Build: `v2.82-selected-energy-generic-roles-workspace-product-evidence`. Installation und gezielte Nachholung: `TESTANLEITUNG_2.82.md`. Befund-/Fixzuordnung: `IMPLEMENTIERUNGSABGLEICH_2.82.md`. Reale Hardwarefreigabe bleibt ein eigenes Gate.

# ONNX Splitpoint Tool v2.81

Aktuelle Lieferung: Compilerkontext für anstehende Kaltjobs, identische Native-Bildeingaben, H10-Tensorlayout und korrekte Negativ-/Qualitybilanz. Installation und Abnahme über den gemeinsamen `install_and_accept_v281.sh`-Starter im vollständigen Bundle. Details: `TESTANLEITUNG_2.81.md`.

## Tool overview and retained history

The ONNX Split-Point Tool analyses neural-network graphs, selects and exports
split candidates, builds accelerator artifacts, executes Generic and Native
pipelines, and produces validation, performance, energy, and scientific
reports for heterogeneous edge-AI systems.

Current identity:

- package and GUI release: `2.81` (historical development lineage `v2.79`)
- workflow/build: `v2.81-hailo-cold-preflight-shared-input-layout-terminal-status`

Historical release 2.79 adds the contract-bound Native three-stage model while preserving
all v2.78.4 Gate-A, audit-launch, Quality, Ranking and resume behavior. The
Native runner exposes `p2_output` as the primary hardware-performance endpoint
and `completed_detection` as the application endpoint. P1, P2 and optional
postprocessing are instrumented separately; the Quality oracle and cryptographic
evidence remain outside the measured hot loop. See
[`docs/NATIVE_THREE_STAGE.md`](docs/NATIVE_THREE_STAGE.md).
The historical roadmap separately reserved `3.0.0` for a frozen campaign release.

Historical release 2.80 repairs the Hailo8 dependency overlay handoff into the actual compute
child, and separates remote process cleanup from staging removal in the runtime
collector. Primary execution errors survive secondary cleanup failures. Existing
productive Force-OFF policy and artifact identity remain. Normal publication and
fresh-process reuse are tested independently from private GPU diagnostics.
See `TESTANLEITUNG_2.80.md` and `VERSION_2.80_BUILD_AND_TEST_REPORT.md`.
The complete bundle includes bounded reuse and hardware follow-up instructions.

Historical release 2.79.34 adds family-local Hailo compiler contexts, productive Force OFF,
private model-build diagnostics and separate Hailo8 dependency planning. CPU
remains the default for each DFC family. Compatible artifacts are reused before
any compiler GPU check; missing/incompatible contracts remain buildable.
The existing configuration, scope, receipt and terminal integrity contracts remain.
See `TESTANLEITUNG_2.79.34.md` and `VERSION_2.79.34_BUILD_AND_TEST_REPORT.md`.
Hardware GPU execution, model quality and scientific energy remain separate gates.

Historical release 2.79.29 removes the unnecessary OpenCV import from the prepared DeepX
Full benchmark and single-image output probe. Source-image geometry is read
using the same Pillow orientation as the unchanged prepared-tensor loader.
The Float32 probability-edge rule introduced in 2.79.28 and the staged remote
probe are retained unchanged. Source tensors, NMS and quality margins are not
relaxed. Regressions explicitly run without cv2 and reject corrupt images.
See `TESTANLEITUNG_2.79.29.md` and
`VERSION_2.79.29_BUILD_AND_TEST_REPORT.md`. Model quality and hardware acceptance
remain separate from the software replay.

Historical release 2.79.23 fixes reuse of compatible historical TensorRT receipts, Native
runtime deployment and error reporting, decoded pre-NMS completion, Full-System
calibration admission and DeepX compiler preflight. It keeps the existing
artifact identities, scientific quality thresholds and pipeline structure.
YOLOv7 graph selection is outside this maintenance release.

Historical release 2.79.22 uses persistent negative build evidence in the shared Hailo
backend, including normal EvaluationRuns and direct builds. An exactly matching
`PARSER_UNSUPPORTED` or `COMPILE_INFEASIBLE` record prevents the same compiler
attempt in a later run. New terminal build outcomes are recorded in the central
evidence store; existing compatible recovery indexes are also consulted.
Infrastructure failures such as CUDA out-of-memory remain retryable
`TRANSIENT_INFRASTRUCTURE` outcomes. Recovery recognizes the validated atomic
HEF/receipt/cache-meta bundles introduced in 2.79.21 while retaining path safety.
The original model, boundary, endpoint, compiler, recipe, calibration and
preprocessing identity remains authoritative. The change adds no independent
scientific hash scheme and does not reconstruct deleted evidence.

Historical release 2.79.21 reports the complete cache plan after the final split selection
and before regular backend builds. Normal compiler-independent selection
finishes before any backend compiler runs. For compiler-dependent Gate-A
selection, a candidate `selection_probe` cache report is saved and logged
before each required feasibility compiler; the final matrix follows once
selection is complete. Strict warm-cache mode blocks unexpected warm-cache
MISSes and UNKNOWN states before compilation; explicitly expected cold
builds remain permitted by the existing policy. Each expected cold build names its model,
boundary, backend, artifact role and reason. Selection changes show the
previously built boundaries, the current selection and expected cold builds.
Hailo HEF, receipt and cache metadata are published as one atomic generation;
HEF-only generations without a receipt are explicitly `legacy_unsealed`.
Historical HEFs with a valid receipt but missing cache metadata can be
validated and migrated without changing existing cache keys. Artifact-store
duplicates are validated and selected in a deterministic order. Existing
backend cache identities and receipt checks remain authoritative.

Historical release 2.79.20 closed artifact reuse for Hailo, TensorRT and DeepX. The
YOLO raw-head fallback now uses the normal content-addressed Hailo cache rather
than forcing a rebuild. TensorRT verifies compatible existing engines and
receipts before `trtexec`, keeps Full engines model-bound and split-independent,
and keeps Part1/Part2 split-bound. Partially populated stable namespaces still
search compatible legacy artifacts, while cache retention exposes explicit
protection and eviction reasons.

The read-only artifact-cache preflight presents Hailo-8, Hailo-10, DeepX,
TensorRT Full and TensorRT Part2 readiness per model after BenchmarkSet
generation but before backend compiler and runtime work. It distinguishes
confirmed misses from unknown/unavailable probes and reports expected and
unexpected cold builds. DeepX cache inspection uses `cache_verify_only`; its
opportunistic compiler prefetch and the normal backend build cannot cross the
barrier. DeepX keeps its existing cache system and gains transparent HIT/MISS
diagnostics. This release adds no new artifact identity, hash, signature,
manifest or sealing contract.

Historical release 2.79.19 was a small calibration and EvaluationRun closure. The
Full-System calibration still uses the one-factor linear fit through the
physical origin. Plausibility deviations in point-factor spread, configured
expected factor and nominal-current tolerance are now explicit warnings and do
not prevent saving an otherwise technically valid measurement. Non-finite or
non-positive values, insufficient measured increment, excessive idle drift,
singular fits, acquisition-integrity failures and the broad hard factor bound
remain blocking conditions.

The EvaluationRun fixes are intentionally narrow. The remote runtime closure
now carries the Hailo attempt-receipt and timeout-policy modules required by the
remote Hailo backend. Generic TensorRT Quality requests carry their already
computed decoder, NMS and quality-endpoint contract values at the required
top-level request/candidate positions. Runtime Required Scope is sealed from
the BenchmarkSet's final accepted cases after reject/backfill, so rejected
candidates are not reported as missing measurements. Canonical report refresh
also retires only stale artifact-index rows for report surfaces it deliberately
rebuilds or removes; unrelated missing artifacts still fail closed.

The same debug-pack audit closes three additional narrow paths: MobileNet and
RegNet receive the Classification Native split policy and are carried through
contract/expected-row and Quality-FIRST orchestration; Hailo-8 ResNet accepts
quantized HEF storage while preserving its FLOAT32 VStream contract, while raw
integer paths verify the exact native HEF stream width; and a
DeepX failure after completed remote dispatch remains a post-dispatch result
processing failure. Newly produced per-artifact DeepX contracts now carry the
model identity directly. Missing and conflicting `input_contract.model_id`
values remain strictly invalid and now receive a concrete binding reason. DeepX
YOLO11 end-to-end hardware replay remains `NOT_RUN` and is not inferred from
the software fix.

The previously full filesystem was repaired operationally by the operator and
is not represented as a source-code change. No additional hash, manifest,
signature or sealing layer is introduced. Real u.RECS/electronic-load hardware
validation and a real multi-host EvaluationRun are both `NOT_RUN` in the build
environment.

Historical release 2.79.18 simplified the Full-System calibration hardware
sequence. An explicitly confirmed already-off Jetson stays off without a
Jetson or M.2 toggle; an initially SSH-ready Jetson is switched off once for
the five captures and restored once. It also added one bounded retry for a
transient acquisition-integrity rejection while preserving the existing gates.

Historical release 2.79.17 gave the maintained Native Energy row-isolation
closure its own package/build identity and completed model-local direct-path
case preflight handling. Its Hailo-10 HEF-native `UINT8` boundary and Energy
semantics are preserved unchanged by 2.79.18.

Historical release 2.79.16 introduced the guided Full-System input gain
calibration and the maintained-source Hailo-10, actual-current and initial
model-local Energy-isolation closure, but reused the same 2.79.16 identity for
those later changes. Release 2.79.17 corrects that provenance ambiguity and
extends isolation to the remaining model-addressable case preflights.

Historical release 2.79.15 keeps the pragmatic direct M.2 idle calibration and
makes the effective energy contract unambiguously Full-System (`FS`) with a
narrow legacy `MB` default migration. It also fixes DeepX-to-TensorRT
Classification Part1: `imagenet_mean_std` materializes and compiles the existing
ONNX Sub/Div adapter with a distinct cache identity. Detection, Hailo build
settings, ranking, Quality thresholds and runner timing are unchanged.

Historical release 2.79.14 makes **Calibrate M.2 idle power** a direct, setup-scoped
Full-System measurement: verify status, measure with M.2 off, measure with M.2
on, restore M.2 on, and atomically save the difference. It asks for no name and
requires no energy-method manifest or calibration SHA. One readable result JSON
keeps the two means, states and delta; ordinary measurement files remain in the
off/on folders. The exact build identity is
`v2.79.14-simple-full-system-m2-idle-calibration`.

Historical release 2.79.13 repaired the installed operational scope and
editable-launcher update path, but its manifest-preparation calibration path is
superseded by the direct v2.79.14 flow.

Historical release 2.79.12 introduced the `m.2` default/migration, verified the
Full-System method before hardware mutation, propagated it through both off/on
measurements, bound the resulting evidence, and exposed the final gate reason.
Its build identity remains
`v2.79.12-platform-power-calibration-provenance-closure`.

Historical release 2.79.11 closed the interrupted diagnostic-run findings,
introduced the registry-backed three-card Platform Power UI and bound
TensorRT Full accelerator-idle normalization into scientific reporting. Its
build identity remains `v2.79.11-platform-power-energy-evidence-closure`.

Historical release 2.79.10 closed updater/workflow interlocking and the current
launcher/acceptance identity around the first platform-power implementation.
Its build identity remains
`v2.79.10-urecs-platform-power-release-closure`.

Historical release 2.79.9 introduced explicit u.RECS platform control to Tool Config. A one-shot
status refresh reports u.RECS reachability, authenticated Jetson SSH readiness
and M.2 accelerator enumeration without periodic polling. Historical release 2.79.10
closes the explicit-target, cross-process interlock, repeated-observation,
capture-state and atomic-registry semantics required before that initial
implementation is used on hardware. The v2.79.9 build identity was
`v2.79.9-urecs-platform-power-and-m2-idle-calibration`.

Historical Release 2.79.8 closes the pre-hardware profile/schema mismatch that prevented
the v2.79.7 YOLO11 gate from starting. The recovery policy is now admitted by
the published evaluation-profile schema, and focused acceptance exercises both
the validated evaluation-profile loader and the exact runtime snapshot loader
against the shipped gate profile. This is an admission-path correction only;
candidate generation, `cut_bytes_only`, Quality thresholds, runner semantics
and energy policy remain unchanged. The exact build identity is
`v2.79.8-yolo11-gate-profile-schema-closure`.

Historical release 2.79.7 closed the runtime-identity and evidence gaps exposed by the
YOLO11l R8B diagnostic run. The six Hailo-8, Hailo-10H and DeepX Full/b067
paths are verified against their exact backend, runner, producer, artifact and
completion contracts. Generic reciprocal-latency rows cannot satisfy a Native
throughput gate; Hailo Full admission requires the existing Native Full runner
and a measured makespan/completion result. The release also canonicalizes
Hailo-10H aliases, materializes the Hailo-8 Full hardware plan, mirrors
external DeepX receipts under the run root, and seals a duplicate-free terminal
artifact index. Candidate generation, `cut_bytes_only`, Quality thresholds and
energy policy remain unchanged. The exact build identity is
`v2.79.7-yolo11-six-path-runtime-identity-closure`.

Historical release 2.79.6 closed the remaining release and campaign-admission gaps. It
binds the immutable Three-Stage claim invocation, logical-primary matrix,
canonical DeepX runtime precision and physical required scope, reproduces the
v2.78.4 reconciliation exactly, permits explicitly unlimited long-run Hailo
cold builds, reseals and verifies the final artifact index, and adds the
fail-closed YOLO11 Full terminal-admission gate. Before the broad seven-model
run it also requires a real YOLOv7 `b066` strict `claim_gate_32` result with
32/32 oracle parity. The current aliases and the
machine-readable focused acceptance report identify this maintenance release
consistently. Candidate generation, `cut_bytes_only`, Quality thresholds and
energy policy remain unchanged. The exact build identity is
`v2.79.6-remaining-changes-yolo11-admission-closure`.

Release 2.79.2 is the installable closure for the concurrent Native runner and
the evidence defects exposed by the seven-model B500 audit. It keeps the
original v2.78.4 run failed and read-only, while future runs seal an immutable
`required_run_scope.json` before any BenchmarkSet generator or compiler can be
dispatched. Normalized mirror representations are grouped only under exact
request/artifact proof, Central Quality joins primarily by request SHA and
logical identity, and P2-only technical observations are reported as
Quality-not-applicable rather than missing task quality. The release also adds
strict canonical/legacy detection-record parsing, append-only Hailo attempt
receipts, semantic scheduler outcomes, phase-local cohort ETA ranges, periodic
atomic launcher status, and an offline run reconciler. The exact build identity
is `v2.79.2-native-concurrent-three-stage-b500-evidence-reconciliation`. Ranking, `cut_bytes_only`, candidate membership, GFLOPS, Quality
thresholds, energy policy and historical decisions are unchanged.

Release 2.79.1 restores hardware-oriented Native Detection timing without
weakening the Quality contract. The Native role exposes `p2_output` as the
primary physical performance boundary and `completed_detection` as the
application boundary. Detection task work stays inside the measured pipeline,
while cryptographic evidence and the frozen reference implementation run once
as an untimed postflight Quality oracle. Contract-bound adapters cover
classification logits, YOLOv7 anchor multiscale heads, YOLO26 decoded/NMS
outputs and YOLO11 DFL16/C80 heads. Legacy endpoint aliases remain explicit for
older report consumers; they are never presented as additional direct
measurements. Generic ranking, candidate generation, `cut_bytes_only`, Quality
policy and the established Development bridge remain unchanged.

Release 2.78.4 closes the narrow Gate-A launcher status gap exposed by the
real YOLO11 run. A configured stop after backend-artifact generation may be
accepted as the planned Gate-A completion only after its terminal receipt and
workflow evidence pass the dedicated audit. Arbitrary nonzero workflow exits,
missing or contradictory receipts, and unplanned partial runs remain blocked.
The historical `v2783` Gate-A profile, evidence index and output identities are
deliberately retained so the already attested common anchor can be reused
without another feasibility search.

The bundled seven-model overnight campaign uses a frozen B500,
score-independent 20-candidate audit profile. It verifies the retained YOLO11
`b067` anchor and exact Hailo-8/Hailo-10 cache receipts, executes that deployment
sentinel first, and leaves the deterministic audit membership and
`cut_bytes_only` ranking unchanged. Hailo-8 and Hailo-10 may build concurrently
only when the frozen scheduler admits the required memory; otherwise they fall
back to safe serial execution. Native and Energy are intentionally outside this
screening run.

The same build also closes the readiness gaps found by the retained Complete-
Set B5 run. It can recover exact positive and negative Hailo compiler evidence
without mutating the historical run, keeps runtime evidence in a separate
ledger, and packages a bounded YOLO11 Gate A. Gate A probes exact evidence and
cache first, performs a Part-1 parser preflight, tries Hailo-8 before any cold
Hailo-10H build. Its valid terminal decisions are `ANCHOR_FOUND` or
`CANARY_BUDGET_EXHAUSTED`; `EVIDENCE_CONFLICT` is a separate fail-closed
invalid outcome that blocks B5. It never starts the unchanged B5 automatically.
Generic Hailo no longer depends on a disabled Native quality policy; DeepX
suite identity separates ONNX models from sidecars; and the ResNet quantized
boundary aggregate distinguishes structural failure, numerical warning and
missing evidence without loosening mapping or shape checks.

Release 2.78.2 closes the two production-schema gaps found during the 2.78.1
evidence check. The read-only verifier now accepts the real historical
eight-field semantic reference placeholder while keeping every field that is
present fail-closed and hash-bound. New management-node CPU references are
published once under
`by_source_contract/<sha256>/canonical_cpu_reference.json`; changing a model,
runner, dataset, plan, or quality contract creates a new immutable snapshot
instead of overwriting evidence from an earlier run. Producer, workflow loader,
existing-evidence verifier, and offline replay all validate the selected path,
source contract, size, SHA-256, storage markers, and regular-file identity.
Legacy fixed-path references remain readable or migratable only when their
recorded status binds the exact source contract and bytes.

Release 2.78.1 adds an optional read-only verifier for already produced
evaluation, Hailo structural-canary, and Full-ONNX self-reference evidence.
It cross-checks model, ONNX, HEF, preprocessing, decoder/NMS, dataset,
prediction, metric arithmetic, raw-output signature, endpoint, completed-result,
and artifact-index identities and writes one externally located, hash-bound
verification receipt.  The receipt is explicitly retrospective:
it does not turn Development data into a prospective hold-out, does not change
the scientific status of the source run, and does not make source files
physically immutable.  No hardware, compiler, B500, ranking, adapter, or
threshold rerun/change is performed.

`VERIFIED` requires every recorded request, candidate and canonical CPU
reference container to remain physically available under its recorded SHA-256.
If two historical result rows name different byte-container hashes at one
later-overwritten canonical reference path, the verifier reports `INCOMPLETE`
with `historical_reference_version_unavailable` even when their canonical
prediction fingerprint is identical.  It never reconstructs or silently
promotes a missing historical container.

```bash
python -B scripts/verify_existing_adapter_evidence.py create \
  --run-dir /absolute/evaluation-run \
  --canary-dir /absolute/structural-canary \
  --self-reference /absolute/yolo26m.json \
  --self-reference /absolute/yolo11l.json \
  --expected-quality-items 500 \
  --out /absolute/new/existing_evidence_receipt.json
```

Release 2.78.0 corrects only the interpretation text emitted for a successful
Full-ONNX self-reference smoke. A PASS now states precisely that completed-task
detections for the tested input satisfy the configured similarity policy and
that dataset-wide AP/accuracy remains separate quality evidence. JSON status,
diagnosis and thresholds are unchanged, as are ranking, adapters, HEFs,
preprocessing, decoder/NMS contracts and existing hardware evidence. No
hardware, compiler or B500 rerun is required for this report-only correction.

Release 2.77.15 closes the YOLO11 Completed-v2 self-reference admission gap.
For a verified `frozen_host_tail` YOLO11 endpoint whose sealed Native decoder
is `ultralytics_regcls`, the semantic verifier now accepts the canonical Full
ONNX decoded pre-NMS tensor `[1,84,8400]` (`ultralytics_decoded`) as the
independent reference. Both paths still pass through the same sealed
score/NMS/geometry contract before comparison. The exception is deliberately
model-, mode- and decoder-bound: YOLOv7 and incompatible raw formats remain
fail-closed. This change makes a real numerical comparison possible; it does
not itself claim parity, rebuild artifacts or rerun hardware.

Release 2.77.14 fixed the first real two-model Hailo-8 structural-canary
payload. BenchmarkSets intentionally hardlink shared COCO images; Python's Tar
writer had represented the second regular source as an on-wire Tar hardlink,
which the verifier correctly rejected before SSH. The payload producer now
opens every source with no-follow semantics and materializes it explicitly as
a regular, byte-bound member. Tar links and symlinks remain forbidden. Terminal
output omits the long attested-source path list while the complete list remains
in `canary_result.json`. No compiler, workflow, model, adapter, threshold or
quality evidence changes.

Release 2.77.13 adds a permanent, read-only backend semantic smoke contract.
It compares canonical prepared RGB bytes across producers, checks original
versus Hailo-fixup ONNX semantics on CPU when ONNX Runtime is available, and
verifies frozen YOLO11/YOLO26 raw-head completion against an independent
completed-detection reference. A separate optional small artifact-reuse
hardware canary exercises an existing Full HEF without compiling or starting
a B500 workflow. Exact Full-quality-only completion can no longer be reported
as technically partial solely because a completed, error-free historical
hardware-smoke expected non-applicable performance rows. Scientific PASS/FAIL
remains separate, reference-comparison reports expose both decision and
comparison status, and a fast-fail point estimate is no longer labelled as a
computed 95% lower confidence bound. Ranking, thresholds, adapters, compiler
recipes and the underlying quality evidence are unchanged.

The hardware-independent semantic gate is driven by a path-neutral JSON spec:

```bash
python -B -m onnx_splitpoint_tool.backend_semantic_smoke --example > smoke.json
python -B -m onnx_splitpoint_tool.backend_semantic_smoke \
  --spec smoke.json --require-pass
```

Replace the example paths with existing prepared-input, ONNX and raw/reference
dump artifacts before running it. `--require-pass` makes both `SKIP` and `FAIL`
non-zero. The optional Hailo-8 canary reuses a sealed Full HEF and performs only
artifact/runtime and structural-output checks; it does not attest numerical
parity or accuracy and never compiles:

```bash
python -B scripts/run_v27713_hailo8_artifact_canary.py \
  --benchmark-set /absolute/benchmark_set/legacy_suite \
  --hardware-matrix /absolute/hardware_matrix.json --setup-id SETUP_ID \
  --max-images 1 --require-hardware --out /absolute/new-canary-output
```

Release 2.77.12 closes the live retry path of the sealed Phase-5 Full-quality
resume. YOLO11 pre-NMS model hashes use the canonical SHA-256 normalizer,
targeted rebuild decisions take precedence over failed archived target
checkpoints, and an already complete exact repair can be audited without
another workflow or hardware dispatch. Must-reuse, no-build, Ranking, Native
and Energy guards remain fail-closed.

Release 2.77.11 closed the two historical artifact gaps exposed by the sealed
Phase-5 Full-quality resume. It accepted correctly indexed zero-byte artifacts,
read-only attested the exact successful classification stderr omission, and
materialized empty stderr files for all future successful remote dispatches.
The wrapper derives the remaining one-to-three endpoints from the
preserved five-to-seven completed results, rejects failed or cancelled workflow
sessions, and fails closed if any must-reuse stage rebuilds. Existing quality
objects and build artifacts remain preserved; no compiler, Ranking, Native,
Energy, Canary or profile work is added.

Release 2.77.10 introduced the cohort-wide, read-only supersession attestation,
recovered only the recognized abandoned 2.77.9 session, and kept every repair
attempt's remote transport isolated.

Release 2.77.9 repairs only the admission and transport isolation of the
targeted Phase-5 Full-quality resume. The launcher clears the archived
fresh-run creation guard while retaining the required Standard-mode check, and
the three missing endpoints use a separate disposable remote transport
workspace so archived requests cannot contaminate the exact subset. The sealed
exact-5/8 admission, build reuse, five preserved result objects, frozen
Ranking, disabled Native/Energy paths and all v2.77.8 adapter corrections
remain unchanged. No new Canary or profile is added.

Release 2.77.8 closes the shared Phase-5 Full-detection quality path. Hailo
per-image inference is reprojected to the already sealed timed raw-head order,
and single-output YOLO matrices such as `[1,84,8400]` are represented as
`decoded_pre_nms` with their existing host decoder/NMS instead of being
mislabelled as final BN6 detections. Its one targeted resume derives and runs
only missing Full-quality identities, preserves every completed result object,
and forbids compiler, Ranking, Native and Energy work. No new Canary or profile
is added.

Release 2.77.7 was the bounded Phase-5 transfer-adapter closure. Explicit
Hailo-10 and empty DeepX-only Hailo target selections now remain authoritative
through repeated run-mode materialisation and workflow binding, while compiler
effort, calibration, timeouts and presets continue to come from Tool Config.
YOLO11 Native-Full uses the existing DFL16/C80 decoder only for the exact
parity-proven six-head 80/40/20 endpoint; archived 2.77.6 YOLO26 and YOLOv7
contracts remain readable. Ranking, thresholds, candidate generation, Quality,
energy and reporting policy are unchanged, and no new hardware gate is added.

Release 2.77.6 closed only the endpoint-resolution gap exposed by the real
2.77.5 Direct Probe.  The probe now exercises the normal automatic DFC parser
path first.  Only an exact ``base_conv`` failure may trigger one retry; before
that retry, the archived producer list and every tool-owned Identity alias are
attested against the byte-identical preserved ONNX.  No-op Identities removed
by DFC are mapped to their existing upstream producer only when expanding the
chosen producer nodes reproduces every graph-output tensor, output slot and
ordering exactly.  The same structured projection is used by future
production retries. Unknown failures, partial aliases, extra or reordered
producer slots and source mutations remain fail-closed. No candidates,
hardware, HEFs, calibration, ranking, Quality or reporting policy are added.

Release 2.77.5 was the preceding narrow managed-venv and compiler-evidence closure. Normal
POSIX virtual environments expose `bin/python` as a symlink; 2.77.4 resolved
that symlink to the system interpreter and consequently projected the system
`bin` directory instead of the managed Hailo profile. Direct managed launches
now preserve the lexical venv path, exactly matching activation semantics, and
the regression suite exercises a real interpreter symlink. The four archived
YOLO26 Part-1 parser calls also separate evidence completion from compiler
capability: successful parses and reproduced explicit-endpoint `base_conv`
rejections are both retained as valid outcomes. `onnxsim` availability is
reported as an optional DFC recovery capability and is never installed or used
to hide a negative compiler result. Ranking, candidate policy, Quality,
performance and scientific reporting are unchanged.

Release 2.77.4 is a bounded generator/managed-venv closure on the unchanged
2.77 ranking freeze. Declared `forced_cases` now remain the exact generator
attempt set, apart from explicitly attested pre-generation native-capability
replacements; a backend rejection can no longer trigger an unrelated legacy
backfill. `full_hef_policy=skip` is authoritative even for a prepared YOLO26
Full baseline and the mandatory early-Full path. Direct managed-venv launches
also project the venv `bin` directory into `PATH`, so vendor subprocesses such
as `onnxsim` resolve exactly as they do after normal venv activation. Ordinary
non-forced candidate search, ranking, Quality, Energy and scientific reporting
remain unchanged.

Release 2.77.3 is the preceding bookkeeping-only closure. Existing conceptual
Hailo Full contracts carry the resolved `hailo_build.build_full: false`
decision as `requested=false`, so backend reuse, contract promotion and service
paths neither copy nor queue those Full HEFs. Missing or true request markers
remain fail-closed.

Release 2.77.2 is a narrow orchestration and acceptance fix after the first
live Hailo parallel-build canary. That run proved genuine Hailo-8/Hailo-10H
Part-1 process overlap, but the heterogeneous matrix plan silently restored
two unrequested Full HEFs despite `hailo_build.build_full: false`. The explicit
false value is now projected as `full_hef_policy=skip` through RunPlan,
generation runtime and orchestration; `skip` removes Full from both matrix and
same-backend Hailo variants. The packaged ResNet50 b052 canary is Part-1-only,
stops after backend artifact generation and fails closed on any extra event,
HEF, cache hit or post-build stage. Its evidence ZIP includes both small HEFs,
receipts, executed verification sources and a SHA-256 manifest. This patch does
not change the frozen `cut_bytes_only` ranker or any scientific protocol.

Release 2.77.1 is a bounded report-only provenance hotfix on top of the frozen
2.77.0 Development ranker. `cut_bytes_only` remains unchanged: lower cut bytes
rank first, with deterministic ties resolved by Boundary ID and then Case ID.
Candidate provenance, stratified windows, seed, quotas, minimum gap, capability
filtering and every candidate-generation rule are unchanged.

The hotfix closes the archived-normalized Full-Run Cross-Runner identity path
exposed by the Smartmirror replay. A stored `endpoint_contract_complete=false`
without the newer explicitness marker is treated as a historical compatibility
default only when an exact, valid central request projection supplies a passed
endpoint attestation whose hash matches the complete endpoint contract and all
other identity fields agree. Explicit
contradictions still fail closed. The report publishes the number of migrated
legacy defaults. The Native replay now also rebuilds the canonical combined
summary through the existing exact, fail-closed validation join, so its 24
Central-Quality bindings no longer remain only in a sidecar while the combined
summary says `not_provided`. Existing Generic and Native measurements are
reprojected read-only into a separate output directory; the replay rejects any
identity, FPS or repetition drift while rebuilding that summary. Source
results, compiler artifacts, B500 Quality and hardware measurements are not
repeated or modified.

Release 2.77.0 froze the single global ranker and introduced endpoint-attestation
provenance normalization. Its compact regression re-normalized raw rows and did
not exercise already archived normalized rows lacking the explicitness marker;
the first Smartmirror replay therefore kept all measurements but excluded the
24 intended pairs as identity conflicts. It also copied the ungated Native
performance summary to the canonical combined-summary name after validating
the 24 Native rows separately. Release 2.77.1 repairs only these replay/report
integration paths and does not reopen the ranking decision.

Release 2.76.2 was the bounded evidence-closure maintenance release for the completed
three-model Development audit. Its guiding contract is endpoint validation,
not post-hoc deployment-candidate selection: every predeclared endpoint stays
visible with its payload validity, explicit terminal state, or unresolved/open
evidence status. A Quality fail may exclude a quality-preserving performance
claim, but never removes the case from Tool validation or from the diagnostic
technical observation surface.

The v2.76 line adds a read-only scientific reprojection into a separate output
root, an endpoint lifecycle ledger, fail-closed handling for nonzero runner
return codes (including SIGSEGV), complete identity/status projection and
backend-local retention of successful artifacts. The existing v2.75.49 audit
is reprojected without rebuilding models, repeating B500 Quality or executing
accelerator hardware. The repaired Development report passes the frozen
formula-decision gate and selects the single global `cut_bytes_only` revision.
Release 2.77.0 implemented and froze that decision before any Transfer-model
performance was inspected; 2.77.1 leaves it unchanged.

The 2.76.2 maintenance closes the final report-only gaps found after the
Native supplement: the frozen 24-row Native matrix now defines the exact join
domain, Generic request identities are restored from already-recorded quality
evidence, technical/Quality/claim cohorts stay separate, and micro plus
per-backend concordance, Hit@1 and Regret@1 are emitted automatically. It also
reports the Development leader over identical qualified actual strata and
rebases replay artifact paths to their final published directory. No compiler,
B500 Quality, Generic performance or Native hardware work is repeated.
It also accepts the canonical normalized full-run cycle/throughput field names
used before the compact scientific-row projection, carries the exact planned
intersection into Markdown, and distinguishes technical-semantic `claim_ok`
from actual scientific claim eligibility.

The release incorporates the complete, never separately deployed 2.75.50
intermediate block. It keeps strict claim eligibility fail-closed while
separating all measured performance observations, diagnostic partial-universe
rankings, Quality-screened results and final claim surfaces. Exact nested
setup identities survive normalization and deduplication, and the Cross-Runner
report exposes technical, Quality-screened and strict Generic/Native pairs
separately. Existing v2.75.49 Generic results and a separately completed Native
supplement can therefore be bound and replayed offline; no accelerator rebuild
or Generic rerun is required.

The resolved profile scheduler now reaches the Hailo build path directly.
Exactly one Hailo-8/Hailo-10H pair for the same candidate may compile in
parallel through two verified managed-venv subprocesses. `auto` is admitted
only when both venvs resolve and is then forced to `venv`; local, WSL and
unresolved backends remain serial. The resource gate requires at least eight
CPU tokens and 12 GB for the pair while retaining a 2-GB controller RAM
reserve. Insufficient resources produce a logged serial fallback, target
outputs and SDK logs are isolated, result merging remains deterministic and no
new compiler timeout is introduced. DeepX prefetch is deferred only while this
bounded pair mode actually owns the controller resources.

Release 2.75.49 is the narrow DeepX Full closeout of the otherwise successful
YOLOv7 v2.75.48 technical anchor. For the exactly registered `yolov7_paper`
decoder, a DeepX-Full semantic dump may bind a missing redundant model digest
in its backend/endpoint contract from the central model/decoder registry.
Every explicit digest is still validated and must agree with all other
explicit values and the registry; invalid or conflicting values remain
fail-closed. This does not change the decoder, anchors, preprocessing, Quality
policy, model bytes, dataset cohort or accelerator settings.

Start `profiles/yolov7_paper_v27549_standard_anchor_b500.yaml` as a **new** GUI
workflow, never Resume the v2.75.48 run. It is a scientifically identical
fresh repetition: only `yolov7_paper:b044`, the frozen B500 cohort and
settings, Generic plus Native/Native Full, and no Energy, ranking or remote
Official COCO. Result and prediction reuse across the release boundary is
forbidden; validated build caches remain reusable. The already accepted CPU
Official-COCO A/B evidence is unchanged, and Official COCO remains mandatory
for the later final Detection run. No B1000 rerun is part of this release.

Release 2.75.48 canonicalized bare and `sha256:`-qualified COCO annotation
digests, removed redundant Native completion-source model-hash propagation,
and admitted an already verified DeepX Full prepared input without a duplicate
transport-level model field. Explicit conflicts remained fail-closed.

Release 2.75.46 closes three real-evidence failures exposed by the three-model
Standard anchor and the B500/B1000 verifier. Native-Full raw-head attestation
uses the already selected ONNX-capable engine interpreter instead of requiring
the Hailo vendor runtime environment to parse ONNX; the parent process still
validates the receipt-bound file identity, primitive output descriptors and
canonical multiscale signature, and structured child exceptions remain in the
failure row.

The scientific projector now accepts the exact Standard setup-local TensorRT
Quality contract as well as the historical Full-only contract. Only the
declared Full-quality acceptance rows enter its identity matrix; ordinary
Generic Quality rows remain diagnostics. The calibration-size canary keeps
strict arm-local artifact binding while normalizing only the treatment-derived
DeepX output-contract file hash and its cascading sealed hashes before the
B500/B1000 semantic endpoint comparison. Runtime output signatures,
postprocessing, runner, policy, preprocessing and endpoint semantics remain
fail-closed.

The fresh three-model acceptance profile is packaged as
`profiles/resnet_yolo26s_yolov7_v27546_standard_anchor_b500.yaml`. Start it as
a new workflow; do not elevate a cross-version Resume as v2.75.46 evidence.

Release 2.75.45 restores the complete GUI path for large score-independent
ranking audits. The GUI exposes the real audit size, execution-union range,
expected Generic rows and Native/Energy state before a fresh start and requires
explicit confirmation; `Cases/model` can no longer make a 20- or 30-candidate
audit look like a small run. The profile editor also makes DeepX classification
preprocessing explicit: new profiles default to `imagenet_mean_std`, while
legacy profiles without the field retain `current_scale_only`.

The managed TensorRT cache now separates its retained non-current cache budget
from the selected run's active working set. A conservative 20/30-candidate
engine reserve may exceed 20 GiB when the existing physical-space preflight
passes. Stable namespace reuse and Resume remain intact; active, foreign,
unowned, linked or otherwise unsafe cache data remains protected and
fail-closed.

Release 2.75.44 fixes the B1000 Validation-authority level mismatch without
changing any dataset or scientific setting. The complete 50,000-image source
manifest is still verified first; its frozen deterministic selection contract
then projects the exact 500-image runtime cohort that is compared with the
archived manifest, image-ID and ground-truth authorities. Missing, duplicate,
drifted or ambiguous rows remain blocking, and no manifest or image is
rewritten.

Release 2.75.43 added narrow compatibility for the historical portable dataset
identity serializer used by the delivered v2.75.40 ImageNet evidence. Legacy
and normalized four-field item aggregates are accepted only when they match
the manifest rows; both are projected to the same normalized portable identity.
Unknown aggregates, file drift, payload drift, component drift and the frozen
validation authority remain blocking. No manifest or dataset image is rewritten.

Release 2.75.42 repaired four production-evidence integrations without changing
the v2.75.41 B500/B1000 experiment. Installed-tree verification attests every
release-owned file while reporting retained user profiles separately, and the
B500 loader accepts the real provisioner's immutable ImageNet kernel identity.
The updater compares file content with `rsync --checksum`, preserves the
existing `.venv`, and refreshes the project's distribution metadata and all
declared console entry points offline with the standard library. Setup-local
TensorRT Quality companions now carry one concrete physical setup and endpoint
identity from the effective plan through dispatch and management admission;
missing companion results are terminal in Standard and Final. Extra scripts
and all other unmanifested Source files remain blocking. The failed
nine-invocation night run is not scientific evidence and is not repeated
before the B1000 gate.

Release 2.75.41 adds the controlled follow-up to the 2.75.40 preprocessing
experiment.  It holds the corrected `imagenet_mean_std` input contract,
ResNet50, EMA calibration, DeepX optimization level 0, the ordered 500-image
validation cohort, Quality policy and setup-local TensorRT control constant,
while increasing only the train-derived calibration cohort from 500 to 1,000
images.  With 1,000 ImageNet classes, the larger class-stratified cohort covers
every class once.  A read-only preflight and a fail-closed paired verifier prove
the actual 500-item subset relation, exact cache/build provenance and the
paired Top-1/Top-5 effect before Standard+ can be released.
The measured v2.75.40 corrected arm reached 80.2/94.8 percent Top-1/Top-5
versus 81.2/95.2 percent for the setup-local TensorRT control; Standard+
therefore remains blocked until this calibration-size canary passes without a
threshold change.

Release 2.75.40 adds a paired ResNet50 DeepX preprocessing experiment. The two
packaged profiles hold the exact model, 500-image calibration/validation
cohort, compiler settings, setup-local TensorRT control and Quality policy
constant while changing only `current_scale_only` versus
`imagenet_mean_std` and an isolated exact-v2 DXNN cache. The corrected arm
uses a content-addressed ONNX Mean/Std adapter; it does not guess a vendor
preprocessing opcode. A hardware-free Float-ORT probe and a fail-closed paired
run verifier distinguish preprocessing loss from remaining
compiler/quantization loss. Full-only TensorRT export now also preserves the
signed physical `native_tensorrt` identity while retaining logical
`tensorrt` inside the nested plan identity.

Release 2.75.39 classified deterministic SSH connectivity failures before any
remote mutation. Such a failure suppresses result collection, never invents a
remote lease, and preserves the concrete connectivity cause as the primary
error. Once a remote lease has started, unresolved cleanup remains strictly
fail-closed and quarantines the run. The start-ready diagnostic profile is
`profiles/resnet50_v27539_deepx_full_quality_canary.yaml`; it preserves the
2.75.38 real-DXNN Full-only Quality path and executes no splits, Native
performance, Energy, ranking, or performance claims.

Release 2.75.36 closes the three issues exposed by the small ResNet50
hardware run. The Native structural validator accepts a runtime channel vector
such as `[2048]` as the singleton-only representation of canonical TensorRT
input `[1,2048,1,1]`, but only for a sealed `as_input`/`identity` contract with
exact tensor name, dtype, element count, byte count, bridge, command, metadata
and Quality-FIRST bindings. Arbitrary equal-element reshapes remain
fail-closed.

Persistent TensorRT engines now use a separate engine-semantic namespace key.
It binds canonical Full-ONNX content, precision/workspace/build contract and
the attested selected TensorRT/GPU ABI while excluding profile names, campaign labels,
tool-release strings and Hailo/DeepX build diagnostics. Exact split-graph
hashes remain bound by content-addressed leaf receipts. On the first 2.75.36
run, eligible older engines are copied only after receipt, ONNX, engine,
`trtexec`, precision/workspace and live GPU-deserialization verification; old
namespaces are left unchanged. Quality-FIRST consumers verify migrated
receipts with `--no-build` before reuse and rebuild normally on any mismatch.
The r2 build attests Jetson Orin through the read-only CUDA Driver API without
requiring `nvidia-smi`, binds only the actually selected GPU, avoids rehashing
old engines on an in-limit warm-retention pass, and preserves Engine/Receipt
bytes and timestamps on verified Quality-FIRST reuse.

Disabled Native Energy is now a genuinely not-applicable axis: its coverage
contract is inactive and coverage, ledger and count fields are `null` instead
of false zero-coverage evidence. Requested strict Energy remains unchanged and
fail-closed. The ready profile is
`profiles/resnet50_v27536_small_acceptance.yaml`.

Release 2.75.35 lets a frozen score-independent audit continue when a planned
candidate is explicitly rejected by compilation or capability checks. The
generator must still classify every frozen identity exactly once, in order,
as accepted or rejected. Missing, duplicate, reordered and outside-backfill
identities remain fail-closed; the prospective candidate plan is not rewritten.

The Evaluation Profile editor now preserves `min_gap: 0` and profile-owned
Audit size, minimum-valid count and seed across loading, saving and run-mode
changes. A ready small hardware acceptance profile is included at
`profiles/resnet50_v27535_small_acceptance.yaml` (ResNet50, Audit 4, minimum 3,
deployment 1, seven logical profiles, Native/Full on, Energy off). Its 2.75.36
successor retains the same scientific scope.

The narrow 2.75.34 Hailo VStream repair is retained: a channel vector such as
`[2048]` is accepted as the byte-identical singleton-squeezed form of canonical
Part-2 input `[1,2048,1,1]` and resolves to `as_input`/`identity`. Spatial axis
permutations and arbitrary equal-element-count reshapes remain fail-closed.
Release 2.75.36 carries this rule through the final structural validator.

The frozen score-independent audit/deployment union and identity preflight
introduced in 2.75.33 remain authoritative. The stable audit-first union cannot
be truncated by the deployment shortlist, narrowed by a second minimum-gap
pass or extended by outside backfill. Explicit unsupported observations are
retained separately from accepted runtime materializations.

The Generic Standard Development-ranking exception is now limited to an
actual predeclared `score_independent_audit` scope with Quality PASS and does
not elevate claim eligibility. Compact Debug Packs also retain each model's
bounded authoritative validation summary. The failed 2026-08-12 run is not
resumed; acceptance starts fresh with Energy disabled using the packaged small
profile.

Release 2.75.32 closes the score-independent audit path end to end. The
Effective Plan now counts Development audits, exposes the deduplicated
audit/deployment execution-union range and sizes Generic rows with a safe
upper bound. Native ranking groups producer rows by completed-task endpoint
rather than the case-local observation identity; incompatible setup,
precision, endpoint and host-postprocessing contexts remain separate.

Generic Standard Quality-PASS observations may contribute only to a
Development ranking audit while remaining ineligible for final claims. Native
audit status is derived from the frozen selection/freeze evidence and usable
metrics: a non-audit run reports `not_requested`, and a measured audit reduced
below its minimum by Quality vetoes reports that shortfall explicitly. Known,
byte-identical Native report aliases are omitted from Debug Packs with a
hash-verified canonical-path record.

Release 2.75.31 made the Debug Pack compact and canonical: raw tensor dumps,
rendered figures, copied resource trees and replay-body mirrors are excluded by
default, while the complete workflow log and bounded, hashed ranking/audit
evidence are retained exactly once. The GUI and both command-line pack paths
use the same policy.

The release also repairs score-independent profile validation, ingests measured
Native observations before ranking and simplifies model configuration to the
fields users actually choose. Role, generalization, validation tier and
candidate-universe mode are derived from the single model usage selection;
audit size, minimum and seed remain global selection settings.

Release 2.75.30 adds a prospective score-independent ranking audit for the
development models and makes Native ranking a first-class report. Selecting
`score_independent_audit` freezes 20 candidates after technical capability
filtering, requires at least 10 quality-eligible measurements per comparison
stratum, and executes the deduplicated union with the normal four-candidate
deployment shortlist. Predictor scores and measured runtimes never influence
the audit sample. Generic and Native observations are evaluated separately by
model, direction, hardware setup, precision and endpoint contract.

Release 2.75.29 made the user-facing `final` run mode a deliberately small
extension of Standard. **Final Quality (Standard+)** uses the same build,
runtime, Native, Energy, reporting, cache and sampled-integrity settings as
Standard and increases only the task-quality budget:

| Setting | Standard | Final Quality |
|---|---:|---:|
| Classification validation items | 500 | 5,000 |
| Detection validation items | 500 | 5,000 |
| Task-quality bootstrap repetitions | 500 | 5,000 |

Final Quality therefore does not automatically prepare or seal a campaign,
does not run a separate repeated full-dataset hash preflight, and does not
require a canary. The selected 5,000-item quality subsets are still
materialized and hashed as ordinary run evidence. Native remains enabled by
default, while Native Energy remains an explicit user switch and is disabled
by default, exactly as in Standard.

Create or edit an Evaluation Profile in the GUI, select run mode `final`,
review the effective summary and start the normal Evaluation Workflow. The
summary should show validation `5000/5000`, bootstrap `5000`, Hailo
`balanced`/optimization level 1, benchmark warmup/runs `3/5`, Native
frames/warmup/repetitions `1000/100/3` and relaxed sampled integrity. Compact
2.78.4 test commands are supplied directly with the release; no separate
build-report or test-guide file is generated for this maintenance release.

Release 2.75.28 introduced the stricter Final-campaign bootstrap, exact
`yolov7_paper` canary and inherited energy-method binding. Those campaign,
preflight, sealing and canary tools remain available for an intentionally
strict archival or advanced protocol, but they are no longer prerequisites
for the ordinary Final Quality run mode.

Release 2.75.27 introduced the explicit `evaluated_matrix` versus optional
`ranking_generalization` claim scopes, canonicalized the confirmatory-holdout
runtime role and removed volatile Hailo compile `elapsed_s` values from the
semantic TensorRT cache key. Those scientific and cache contracts are retained
unchanged in 2.75.28.

Release 2.75.24 gave the optional YOLOv7 Full-only Quality canary the normal
Standard Hailo cache identity (`balanced`, optimization level 1, relaxed
receipt verification, 500 calibration images) and added its receipt-attested
pre-upload gate. Those historical canary wrappers remain unchanged.

Release 2.75.22 repairs the scientific evaluation and reporting defects found
in the completed YOLOv7 Quality canary from 2.75.21. Detection now computes
the configured AP75 guardrail in addition to AP50 and AP50:95. Any configured
guardrail that is not emitted fails closed. Its algorithm/cache contract is
versioned separately, so old cache entries stay untouched and cannot suppress
the new calculation.

The hardware-free replay reads the six existing central Quality requests,
candidate predictions and CPU reference predictions from the original
EvaluationRun. By default it evaluates all six rows and also publishes the
canonical four-row Full-only view (Hailo-8, TensorRT@Hailo-8, Hailo-10H and
TensorRT@Hailo-10H). It writes a separate sibling output directory and never
mutates the 2.75.21 summary or cache. Technical completion, aggregate Quality
decision and scientific status are reported as separate axes; setup-local
TensorRT rows retain their request and setup identities in the scientific
report.

The explicit Full-only Quality-canary profile contract rejects generic Split,
Composed and performance rows before hardware dispatch. Debug packs now carry
HEF build receipts, setup-local TensorRT dispatch preflights, the remote
hardware matrix and strictly referenced replay inputs when those bytes exist.
No Accuracy threshold is relaxed.

Release 2.75.21 remains the setup-local TensorRT dispatch repair exposed by the
immutable 2.75.20 three-model Full-only Smoke
`resnet_yolo26s_yolo7_20260807_205238`. Release 2.75.20 had already repaired
the remote import boundary and final plan seal: all three accelerator runtimes
started, the three canonical ORT-CPU references remained on central
management, and 12 of 18 Native-Full rows completed. The remaining six
TensorRT rows on the Hailo-8 and Hailo-10H setups were not attempted because
Generic/Quality dispatch assigned `ort_tensorrt` only to DeepX.

The normalized dispatch now creates one setup-bound TensorRT Quality companion
per model on Hailo-8, Hailo-10H and DeepX. DeepX remains the only owner of the
model-wide TensorRT performance row, so Quality replication does not create
duplicate performance measurements or shared-output collisions. Canonical
`ort_cpu` remains a local, semantic-only management reference.

A fast local preflight validates the complete 3x TensorRT companion matrix,
unique setup identities, the single DeepX performance owner and the absence of
CPU-disguised TensorRT rows before threads, upload or SSH. Native summaries use
the planned 18-row denominator, missing setup-local TensorRT evidence receives
a precise non-attempted reason, and Full-only Split-Quality is explicitly not
applicable instead of invalid.

The archived 2.75.19, 2.75.20 and 2.75.21 runs, all numerical thresholds and
all hardware evidence remain unchanged.

Release 2.75.12 aligned the terminal cache-canary verifier with the real
Classification writer schema. Top-level `ok: true` remains mandatory, while
the redundant top-level `status` alias may be absent. If a writer emits that
alias, it must still be exactly `ok`. Runner and repetition status, execution
identity, requested and completed work, freshness, return code and positive
throughput remain strict terminal gates.

Release 2.75.11 repairs the terminal reporting failure observed after the
2.75.10 canary had already reused the intended cache and completed the real
Hailo-8/TensorRT inference. Recursive collection could see that one execution
both as a reduced direct fallback and as the complete runner result, then keep
the reduced row only because it was found first. The collector now selects one
canonical, complete runner result for the execution instead of allowing a
duplicate fallback view to hide it.

The canary PASS decision uses that canonical current-run result and its
execution envelope: backend/model/case identity, successful return status,
freshness, requested and completed work, and positive throughput. Redundant
command/result cross-links remain available for diagnostics but do not turn a
successful hardware execution red merely because an auxiliary projection
omitted them. Cache and primary artifact compatibility remain strict. Reuse is
still compiler-free: no cache deletion, DFC, `trtexec` build, ORT, Quality or
Energy fallback is permitted.

For `cache_verify_only`, compatibility is semantic and local: one complete
artifact set must match model, split, setup, backend, precision and boundary.
Historical producer and receipt hashes are retained for diagnostics, not used
as a hard-coded allowlist. Standard and final campaign paths remain strict.

Release 2.75.10 is the preceding historical-cache multi-hit repair. It selected
the preserved 4 August Smoke producer and its complete eight-role artifact
identity, executed the requested 10-frame hardware replay successfully, and
thereby exposed the duplicate fallback-row problem in the terminal collector.

Release 2.75.9 is the preceding Smoke-profile/cache-root discovery repair. It
correctly found both historical exact roots, exposing the overly broad rule
that treated any second verified root as fatal before hardware execution.

Release 2.75.8 closes the five general contracts exposed by the first broad
Native Smoke: the Hailo-8/TensorRT process now discovers Hailo sites from the
separate Hailo interpreter while retaining system TensorRT; receipt-signed
compiler ONNX siblings can promote cached Full-Hailo endpoints; the exact
sealed DeepX prepared input is transported into the later Native suite; and
Detection validation preserves completed-result artifacts while refusing to
score a different reference image as a zero-quality measurement. The bounded
cache replay now validates an exact sealed binding before any ONNX/Hailo import
or compiler path is reachable.

The final-campaign preflight additionally requires content-hashed adapter
implementation sources when confirmatory hold-outs are declared. Diagnostic
dequantization sweeps are explicitly non-claimable and cannot export a
production contract. See
[the hold-out adapter protocol](docs/HOLDOUT_ADAPTER_PROTOCOL.md).

Release 2.75.7 is the preceding remote runtime-closure repair. It transfers,
hashes and imports `cache_verify_policy` before the Hailo backend and verifies
the import in a clean isolated remote tree.

Release 2.75.6 remains the preceding start-contract repair. Its explicit fresh
Run ID is forwarded only for an attested `cache_verify_only` profile; normal
profile-driven runs still reject the override.

The underlying cache-replay path reads the authoritative DFC version directly
from the managed venv's package
metadata without importing DFC or starting a child process, so an existing
Hailo-8 receipt is looked up under its real compiler identity. The guarded
generator is clamped to the single attested case and artifact scope; a miss
cannot backfill another split, probe a Full HEF, or invoke the backend-agnostic
fallback. Misses are reported as visible `cache_miss_blocked` terminals with a
complete `hailo_cache_miss.json` diagnostic. Native/SSH dispatch is skipped
before `STAGE_START` after such a miss.

Release 2.75.3 introduced the hard `cache_verify_only` artifact policy. The
effective profile is resolved normally first, then its exact model, logical
profile, backend, setup, case, frame, warmup, repetition, Generic, Full and
Energy matrix is attested. Cache restore and compatible Hailo exact-v2-to-v3
migration remain available; DFC, DX-COM, `trtexec`, and implicit TensorRT build
paths remain independently fenced.

Release 2.75.2 is the preceding Native-runtime recovery. It verifies the remote
dependency closure, persists receipt-bound Hailo-8 child checkpoints, continues
later rows after a child failure, admits receipt-bound YOLOv7 source raw heads,
and allows terminal partial Performance matrices to proceed to non-claim Energy
observations. Its canonical Native execution contract and collection-receipt
cleanup rule remain active.

The Hailo cache is intentionally version-neutral. New requests use an exact-v3
key that also binds the compiler net name, normalized input shapes and RT-
metadata flag. A valid exact-v2 artifact created by 2.75.0 is restored and
backfilled under v3 without compiler dispatch only after the compiler-ONNX,
SDK, calibration, preprocessing, build-payload, receipt and HEF identities
match. Tool version, Run ID, timestamp and EvaluationRun path do not force a
rebuild. Tampered or coarse legacy ArtifactStore records remain fail-closed.

Zero-start reporting now derives `started`, `not_started` and terminal counts
from the objective ledger. A plan of `N` rows with no start is reported as
`blocked_before_measurement` and `0/N`, never as `0/0 completed`. Release
2.75.0 remains the underlying preprocessing/Completed-v2 contract repair.

Release 2.73.8 made Native Energy membership a technical invariant: runtime
success, a constructible Energy command/preflight, and either a Full baseline
or one verified Part-2 input. Quality, Semantics, pairing, and claim
eligibility remain post-hoc annotations. Generic Energy remained disabled in
that historical release; v2.79.16 preserves a later explicit Generic request
or reports a visible planning blocker instead of silently discarding it.

Release 2.73.7 added durable Native Performance and Native Energy checkpoints.
The current Standard campaign commits exactly 63 Performance rows in three
setup-local groups of 21 before validation or Energy begins. Each Energy row
is published atomically as `not_started`, `running`, `completed`, `failed`, or
`cancelled`; Resume continues at the first nonterminal row and reuses the
terminal Performance checkpoint. A terminal child handoff remains importable
even if its stdout pipe or parent process disappears immediately afterward.

Release 2.73.6 added exclusive EvaluationRun ownership, a frozen Resume
profile/plan contract, and recursive cancellation of workflow-owned process
trees. A second GUI/CLI writer is rejected before it can mutate the run, and
children launched through supported foreground helpers, including nested
POSIX sessions, are terminated and waited during `SIGINT`/`SIGTERM` or GUI
cancellation. A stale `running` lock after a hard
writer crash is fenced before Resume. Token-bound remote process leases and an
atomic cross-process journal extend the same ownership rule through executable
Native, validation, benchmark, and Energy SSH workloads.

Release 2.73.5 removed the withdrawn 2.73.2–2.73.4 parallel hardware-
acceptance implementation. Automated hardware acceptance now starts the
existing Evaluation Workflow with the same frozen profile-to-options mapping
as the GUI. It does not copy suites or reimplement CPU, Native, Quality,
remote, or Energy execution.

## What 2.73 provides

- Native Energy admits every runtime-successful Native row whose command and
  preflight contract can be constructed. Quality and pairing annotate later
  scientific eligibility instead of suppressing physical measurements.
- Classification uses exact hit counts at inclusive margins and a narrow
  floating-point fallback when counts are unavailable.
- Cross-host dataset identity is content-based rather than tied to absolute
  `/home/...` paths.
- Native Full and YOLOv7 validation consume the attested Completed-Task result
  produced by the measured hotloop. Physical raw outputs remain separate
  Boundary or fallback evidence.
- Technically complete `fail` and `inconclusive` decisions count as complete
  negative evidence while remaining ineligible for positive claims.
- Missing, conflicting, `unavailable`, or technically invalid evidence stays
  fail-closed.
- Pytest discovery is restricted to the current `tests/` tree, test state is
  isolated from the real user home, and source archives are built only from a
  verified allowlisted manifest.
- Native Full Resume selects backend-specific accelerator artifacts: DeepX
  requires `dxnn`; Hailo and TensorRT require `hef`; unknown Full backends fail
  closed.
- GUI and profile-driven CLI starts share one immutable start-snapshot options
  resolver, including Standard benchmark, Hailo, remote, and frozen hardware
  settings.
- Each EvaluationRun has exactly one active writer. Resume rejects a changed
  profile or effective plan before logs or archived run files are modified.
- Workflow cancellation recursively terminates children and separately-
  sessioned grandchildren launched through the supported foreground helpers,
  without signalling unrelated processes. Deliberate local daemonization
  before registration is outside this ownership contract and fails closed at
  the final quiescence gate when a survivor is known.
- Executable Native, validation, benchmark, and Energy SSH work is journaled
  before launch and cleaned by exact remote PID/PGID/start-time identity before
  the local SSH path stops. Leased SSH explicitly disables backgrounding and
  persistent multiplexing regardless of user/system SSH configuration.
  SCP/rsync clients remain locally process-tree-owned; they are not described
  as exact leased remote workloads.
- Native Performance has an atomically replaced, artifact-hashed checkpoint.
  Standard/Final execution requires the frozen 63-row campaign shape
  (`3 × 21`) and rejects missing, duplicate, unexpected, or identity-drifted
  report rows.
- Native Energy owns one atomic row checkpoint per planned measurement.
  Cancellation changes at most the active row to `cancelled`; rows that were
  never started remain `not_started` and are never relabelled with a pipe
  error. A managed Resume starts at that cancelled row and never reruns a
  completed Performance stage.

Local verification does not constitute hardware or scientific evidence and
does not authorize a Resume of an archived run without its frozen snapshot
contract. For 2.75.49, retain the accepted official-COCO YOLOv7 decoder A/B
evidence and review the fresh YOLOv7-only GUI effective plan, b044/B500 scope and
Native/Energy switches before starting hardware. Large audits require neither a
manual cache deletion nor a hidden retention-limit environment override when
the normal physical-space preflight passes. Historical canaries remain
available as diagnostics. Their verified Debug Packs can still be used for
later hardware-free AP50:95/AP50/AP75 replay. The supplied
`scripts/update_source_release.sh` synchronizes the verified ZIP into an
existing installation, removes obsolete source files with a recoverable backup,
and preserves `.venv`, external caches, runs, and custom profile YAMLs.

The complete historical suite inherited from 2.73.8 is not fully green in the
release environment: 31 named legacy tests failed in the frozen basis. The
acceptance script rejects every failure outside that set while allowing an
inherited failure to be resolved or absent in another local environment; it
does not relabel a non-green complete suite as successful. All focused 2.75.22
contracts and the new 2.75.26 Native/Energy repair contracts remain hard failures.

## Start the GUI

From the source tree:

```bash
./start_gui.sh
```

Alternative entry points:

```bash
python analyse_and_split_gui.py
onnx-splitpoint-gui
```

The two main GUI modes are:

1. **Analyze & Split** — load one ONNX model, inspect split candidates and
   compatibility, and export selected partitions.
2. **Evaluation Workflow** — load an Evaluation Profile, run the campaign,
   observe the Jobs tree, and open the Results Bundle.

## Evaluation Workflow CLI

```bash
python -m onnx_splitpoint_tool.workflow.run_evaluation \
  --profile smoke_regression_v1 \
  --models-root "$HOME/Models" \
  --out "$HOME/Models/EvaluationRuns" \
  --execution-mode generate_and_run
```

Equivalent wrapper:

```bash
python scripts/run_evaluation_workflow.py \
  --profile smoke_regression_v1 \
  --out "$HOME/Models/EvaluationRuns"
```

## Fresh Standard hardware run

The automated acceptance entry point delegates exactly once to the normal
`EvaluationWorkflowRunner`:

```bash
PY=/path/to/existing/.venv/bin/python \
  bash scripts/run_fresh_standard_workflow.sh \
    --profile '/path/to/ResNet&Yolo26s&yolo7.yaml' \
    --out "$HOME/Models/EvaluationRuns"
```

It resolves the same frozen Standard snapshot as the GUI and refuses Resume,
partial stages, imported results, disabled benchmarks, disabled remotes, or a
non-Standard profile before a run directory is created. `--only-model
yolo26s` is available for a real-path canary; omit it for the complete profile.
Use `--plan` for a hardware-free command preview.

## Final Quality hardware run

Final Quality uses the ordinary Evaluation Workflow, not the strict campaign
bootstrap. In **Evaluation Workflow → Profil erstellen/bearbeiten…**, select
run mode `final`, select the intended models and logical hardware profiles,
and choose the Native/Energy switches explicitly. Save the profile with
**Speichern & verwenden**, verify the effective plan, then start it from the
GUI or with the saved profile path:

```bash
python scripts/run_evaluation_workflow.py \
  --profile /path/to/final_quality_profile.yaml \
  --out "$HOME/Models/EvaluationRuns" \
  --execution-mode generate_and_run
```

The legacy CLI profile alias named `final` and the strict
`thesis_final_*` templates are separate profile inputs; they are not a
substitute for selecting the central `final` run mode in a saved Evaluation
Profile.

The current detached seven-model launcher admits a run only after both retained
hardware evidence roots pass fresh read-only verification:

```bash
bash scripts/run_v27915_seven_model_long_overnight.sh \
  --yolov7-claim-output /absolute/path/to/yolov7/evidence \
  --yolo11-r8b-output /absolute/path/to/yolo11/r8b/output \
  --preflight-only
```

Removing either option, using diagnostic one-image YOLOv7 evidence, or failing
any of the 32 strict parity items blocks the broad launch.

This retained seven-model profile is deliberately a **Generic screening
diagnostic**: `native_enabled: false` and `energy_enabled: false`. Its launcher
must not be used or described as the final Native/Energy campaign. The final
energy profile is prepared only after the retained v2.79.19 Full-System input
calibration smoke, the three setup-specific M.2 idle-power recalibrations, the
v2.79.21 final-selection cache preflight and a bounded EvaluationRun validation have passed.

## Safe local acceptance

Do not paste `set -euo pipefail` or an `exit` into an interactive SSH shell.
Run the supplied child script instead:

```bash
if bash scripts/run_local_acceptance.sh; then
  printf '%s\n' "PASS: local acceptance completed."
else
  rc=$?
  printf 'FAIL: local acceptance stopped (rc=%s).\n' "$rc" >&2
fi
```

The script:

- guards against accidental `source` use before changing shell options;
- verifies the current release/build identity and inherited bounded contracts;
- verifies the shipped source manifest read-only;
- runs the focused replay, identity, endpoint-ledger, Quality-projection,
  pipeline-repair and scheduler regressions plus lightweight syntax checks;
- does not create a build report, test report or release archive;
- skips all hardware work, Resume, and Standard/Final-Quality execution; the
  separate workflows above are the hardware entry points. The current focused
  release gate is `scripts/run_v27923_small_acceptance.sh`. Older strict
  campaign, cache-canary and replay wrappers remain optional historical
  diagnostics.

## Main EvaluationRun artifacts

```text
EvaluationRuns/<run_id>/
  profile.yaml
  run_manifest.json
  artifact_index.json
  evaluation_workflow.log
  jobs/
  models/<model_id>/
    analysis/
    benchmark_set/
    benchmark_results/
    validation/
    hardware/
  reports/
    native_energy_measurements/
      stages/native_energy/stage_result.json
      checkpoints/native_energy/journal.json
      checkpoints/native_energy/rows/*.json
    scientific/
      scientific_report.json
      scientific_report.md
      campaign_readiness.csv
      task_quality.csv
      performance_results.csv
      energy_results.csv
      thesis_tables/*.tex
      figures/*.{pdf,png}
  stages/run_native_producers/
    native_performance/stage_result.json
    native_performance/stage_snapshot.json
    native_coordinator/stage_result.json
```

## Current documentation

- [2.79.23 test guide](TESTANLEITUNG_2.79.23.md)
- [2.79.23 build and test report](VERSION_2.79.23_BUILD_AND_TEST_REPORT.md)
- [Historical 2.79.22 test guide](TESTANLEITUNG_2.79.22.md)
- [Historical 2.79.22 build and test report](VERSION_2.79.22_BUILD_AND_TEST_REPORT.md)
- [Historical 2.79.21 test guide](TESTANLEITUNG_2.79.21.md)
- [Historical 2.79.21 build and test report](VERSION_2.79.21_BUILD_AND_TEST_REPORT.md)
- [Historical 2.79.20 test guide](TESTANLEITUNG_2.79.20.md)
- [Historical 2.79.20 build and test report](VERSION_2.79.20_BUILD_AND_TEST_REPORT.md)
- [Historical 2.79.19 test guide](TESTANLEITUNG_2.79.19.md)
- [Historical 2.79.19 build and test report](VERSION_2.79.19_BUILD_AND_TEST_REPORT.md)
- Compact 2.78.4 test commands are provided in the release handoff.
- The one-time live acceptance procedure is in
  `V2772_HAILO_PARALLEL_CANARY_README.md`.
- [Historical test guide 2.75.47](TESTANLEITUNG_2.75.47.md)
- [Historical build and test report 2.75.47](VERSION_2.75.47_BUILD_AND_TEST_REPORT.md)
- [Historical test guide 2.75.46](TESTANLEITUNG_2.75.46.md)
- [Historical build and test report 2.75.46](VERSION_2.75.46_BUILD_AND_TEST_REPORT.md)
- [Historical test guide 2.75.45](TESTANLEITUNG_2.75.45.md)
- [Historical build and test report 2.75.45](VERSION_2.75.45_BUILD_AND_TEST_REPORT.md)
- [Historical test guide 2.75.44](TESTANLEITUNG_2.75.44.md)
- [Historical build and test report 2.75.44](VERSION_2.75.44_BUILD_AND_TEST_REPORT.md)
- [Historical test guide 2.75.43](TESTANLEITUNG_2.75.43.md)
- [Historical build and test report 2.75.43](VERSION_2.75.43_BUILD_AND_TEST_REPORT.md)
- [Historical test guide 2.75.42](TESTANLEITUNG_2.75.42.md)
- [Historical build and test report 2.75.42](VERSION_2.75.42_BUILD_AND_TEST_REPORT.md)
- [Versioning and the 3.0 gate](docs/VERSIONING.md)
- [Documentation index](docs/README.md)
- [Quick start](docs/QUICKSTART.md)
- [Evaluation profiles](docs/EVALUATION_PROFILES.md)
- [Evaluation workflow](docs/EVALUATION_WORKFLOW.md)
- [Validation datasets](docs/VALIDATION_DATASETS.md)
- [Model preparation](docs/MODEL_PREPARATION.md)
- [Artifacts and postprocessing contract](docs/ARTIFACTS_AND_POSTPROCESSING_CONTRACT.md)
- [Native producer EvaluationRun usage](docs/NATIVE_PRODUCER_EVALRUN_USAGE.md)
- [Native boundary validator](docs/NATIVE_BOUNDARY_INTERFACE_VALIDATOR.md)
- [Native HailoRT/TensorRT fast path](docs/NATIVE_HAILORT_TRT_FIFO_FASTPATH.md)
- [Energy window method validation](docs/ENERGY_WINDOW_METHOD_VALIDATION.md)
- [Clean release policy](docs/CLEAN_RELEASE.md)

Historical release notes are intentionally omitted from the clean source ZIP.
The consolidated lineage remains in `docs/VERSIONING.md`.

## v2.79.3: productized Native Three-Stage endpoint evidence

This release productizes the validated YOLOv7 Hailo-8 concurrent P1/P2/Post path. It parameterizes the sentinel corpus size, passes concrete corpus/reference/output paths to the vendored runtime, and projects non-empty stage timing and oracle parity evidence into the normal Native result. All v2.79.2 B500 reconciliation, sealed scope, strict parser, attempt receipt, launcher, ETA, and read-only reprojection fixes remain unchanged.



## v2.79.4: release-line acceptance consistency

Build ID: `v2.79.4-native-productized-three-stage-release-consistency`

This maintenance release keeps the v2.79.3 productized Native Three-Stage and B500 evidence logic unchanged while aligning the generic v2.79 smoke alias, updater entrypoint, local acceptance, and release-line provenance tests with the installed maintenance version.

## v2.79.5: release-launcher and evidence closure

Build ID: `v2.79.5-release-launcher-evidence-closure`

This maintenance release retains the v2.79.4 scientific and runtime contracts
while closing current-release acceptance aliases, the seven-model Generic
launcher identity, and evidence/source-snapshot provenance. The frozen
seven-model B500/audit-20 profile is carried forward without changing its
scientific configuration.

## v2.79.6: remaining-changes and YOLO11 admission closure

Build ID: `v2.79.6-remaining-changes-yolo11-admission-closure`

This maintenance release closes the remaining Three-Stage invocation,
logical-matrix, runtime-precision, physical-scope, legacy-reconciliation,
long-run Hailo, final artifact-index and YOLO11 Full admission gaps. The
focused release gate emits a machine-readable phase report and keeps the
established ranking, Quality thresholds and energy policy unchanged.

## v2.79.7: YOLO11 six-path runtime-identity closure

Build ID: `v2.79.7-yolo11-six-path-runtime-identity-closure`

This maintenance release makes YOLO11 Full/b067 admission backend-bound across
Hailo-8, Hailo-10H and DeepX-M1. Native Full throughput must come from the
Native producer service with exact completion/makespan evidence; Generic
reciprocal-latency rows remain diagnostic only. The release also closes Hailo
plan/alias, DeepX receipt and terminal artifact-index consistency.

## v2.79.8: YOLO11 gate-profile schema closure

Build ID: `v2.79.8-yolo11-gate-profile-schema-closure`

This maintenance release makes the shipped YOLO11 R8B gate profile valid for
the strict evaluation-profile schema and verifies it through both real
pre-hardware loader paths. It preserves the v2.79.7 runtime-identity,
recovery, runner, ranking, Quality and energy contracts.

