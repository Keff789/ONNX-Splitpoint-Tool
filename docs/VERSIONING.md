# Versioning policy

## Current release: 2.79.22 — Global Hailo negative build evidence

Build/workflow: `v2.79.22-global-negative-build-evidence`.

The shared Hailo backend checks persistent evidence before expensive compilation
and records terminal outcomes for later runs. Exactly matching deterministic
parser and mapping failures are reusable; transient infrastructure errors and
aborts never act as permanent negative cache entries. Existing compatible
recovery indexes remain usable. Atomic HEF/receipt/cache-meta bundles can be
recovered without accepting arbitrary symlinks or mixed generations.

This extends the existing evidence identity and recovery format. It does not
add a scientific identity scheme or infer missing entries from a model name and
boundary alone. It does not recover deleted logs or run any hardware migration.

| Purpose | Value |
|---|---|
| GUI and GitHub release label | `2.79.22` / `v2.79` |
| Python package version | `2.79.22` |
| Development lineage | `v2.79` |
| Workflow-contract version | `v2.79.22-global-negative-build-evidence` |
| Auditable build identifier | `v2.79.22-global-negative-build-evidence` |
| Build-contract version | `2` |
| Final campaign release | `3.0.0` |

The active gate is `scripts/run_v27922_small_acceptance.sh`. It retains relevant
2.79.21 cache/publication regressions and the existing evidence/Gate-A tests,
plus the new persistence, normal-backend, recovery and preflight tests. Final
results belong to the delivery's machine-readable reports; hardware remains
`NOT_RUN` in the build environment. Versioned historical aliases and release
documents remain available.

## Historical release: 2.79.21 — Final-selection cache preflight and atomic publication

Build/workflow: `v2.79.21-cache-preflight-atomic-publication`.

The final accepted split selection determines a complete cache-preflight matrix
before regular backend builds. Compiler-dependent Gate-A selection first
saves and logs a `selection_probe` cache report before each required
feasibility compiler. Strict warm-cache mode blocks unexpected warm-cache
MISSes and UNKNOWN states at that point, while explicitly expected cold
builds remain permitted. The complete final matrix follows the resulting selection. Cold-build diagnostics name the model, boundary,
backend and reason, including a `selection_changed` record when prior builds
belong to other boundaries. Hailo cache publication commits HEF, receipt and
cache metadata together. HEF-only generations without receipts are explicitly
`legacy_unsealed`. A historical valid receipt permits validation and migration
when only cache metadata is missing, retaining the existing cache key. Artifact-store duplicates undergo deterministic validation
and selection using existing artifact identity and receipt contracts.

| Purpose | Value |
|---|---|
| GUI and GitHub release label | `2.79.21` / `v2.79` |
| Python package version | `2.79.21` |
| Development lineage | `v2.79` |
| Workflow-contract version | `v2.79.21-cache-preflight-atomic-publication` |
| Auditable build identifier | `v2.79.21-cache-preflight-atomic-publication` |
| Build-contract version | `2` |
| Final campaign release | `3.0.0` |

Offline regression results are recorded in the current release report. Real
seven-model cache preflight and hardware execution remain `NOT_RUN` in the
build environment. Earlier documents and versioned smoke aliases remain
available as historical diagnostics; its historical acceptance gate is
`scripts/run_v27921_small_acceptance.sh`.

## Historical release: 2.79.20 — Artifact Reuse Closure

Build/workflow: `v2.79.20-artifact-reuse-closure`.

Version 2.79.20 makes persistent backend artifacts authoritative before an
expensive compiler is started. The YOLO raw-head Hailo fallback uses the normal
cache, TensorRT validates and reuses existing engines before `trtexec`, and
TensorRT Full is model-bound rather than split-bound. Partially populated
stable namespaces continue an artefact-wise compatible legacy search instead
of treating namespace existence as proof of completeness. Retention and cache
decisions expose explicit reasons.

A read-only artifact-cache preflight produces the per-model Hailo-8, Hailo-10,
DeepX, TensorRT Full and TensorRT Part2 matrix and separates confirmed misses
from unavailable/unknown probes. It counts expected and unexpected cold builds
after BenchmarkSet generation and before backend compiler/runtime work. DeepX
cache inspection is `cache_verify_only`; normal compiler dispatch remains
behind the barrier. DeepX retains its established persistent cache and adds
transparent Full/Part1 HIT/MISS diagnostics.

No additional artifact identity, hash, manifest, signature or sealing layer is
added. Ranking, Pipeline-FPS, Quality and Energy semantics remain unchanged.
Real seven-model cache preflight, compiler reuse and multi-host execution are
`NOT_RUN` in the build environment.

| Purpose | Value |
|---|---|
| GUI and GitHub release label | `2.79.20` / `v2.79` |
| Python package version | `2.79.20` |
| Development lineage | `v2.79` |
| Workflow-contract version | `v2.79.20-artifact-reuse-closure` |
| Auditable build identifier | `v2.79.20-artifact-reuse-closure` |
| Build-contract version | `2` |
| Final campaign release | `3.0.0` |

## Historical release: 2.79.19 — calibration warning and EvaluationRun closure

Build/workflow: `v2.79.19-calibration-warning-evalrun-closure`.

Version 2.79.19 retained the one-factor Full-System correction through the
physical origin while moving point-factor spread, the expected-factor range
and nominal-current tolerance into non-blocking plausibility warnings. It also
closed the incomplete remote Hailo runtime staging, Generic TensorRT Quality
field projection, final accepted-case Required Scope, canonical report/index
reconciliation, Classification split policy/orchestration and narrow Hailo-8
ResNet and DeepX result-binding defects found in the failed Complete-Set run.
Technical calibration and Quality failures remained fail-closed.

## Historical release: 2.79.18 — simplified Full-System input calibration

Build/workflow: `v2.79.18-simplified-full-system-input-calibration`.

Version 2.79.18 admitted an explicitly confirmed already-off Jetson without a
Jetson or M.2 toggle and restored an initially ready Jetson with one off and one
on transition. It retained the five capture phases and actual current/voltage
fit and added one bounded retry for transient acquisition-integrity rejection.
The known EvaluationRun defects addressed by 2.79.19 remained open.

## Historical release: 2.79.17 — Native Energy row-isolation closure

Build/workflow: `v2.79.17-native-energy-row-isolation-closure`.

Version 2.79.17 assigns a unique release identity to the maintained closure and
extends the direct non-variant Native preflight from BenchmarkSet-only isolation
to all model-addressable case-selection failures. Missing `case_map_only` model
entries, empty/duplicate/missing requested cases, no discovered cases and no
Native-supported selected case are recorded as model-local exclusions. Other
models continue to transfer, run and measure Energy. Malformed global selection
configuration remains fail-closed for the complete stage, and a stage with no
remaining runnable model fails with the collected per-model reasons.

The release retains the guided Full-System input gain calibration, separate
nominal and actual electronic-load current entry, Hailo-10 HEF-native `UINT8`
boundaries, raw quality-unqualified Energy retention and explicit Generic-Energy
planning introduced in the maintained v2.79.16 source. The Profile Editor text
now reflects the actual contract: physically executable Native rows may collect
raw Energy despite a Quality problem, but that Energy is not qualified for the
scientific claim. No new hash, manifest, signature or sealing layer is added.

## Historical release: 2.79.16 — guided Full-System input calibration

Build/workflow: `v2.79.16-guided-full-system-input-calibration`.

Version 2.79.16 introduced the setup-scoped Full-System input gain trim. Its
later maintained source also added Hailo-10 HEF-native I/O, actual-current entry
and initial model-local Energy isolation while retaining the same version and
build ID. Version 2.79.17 removes that identity ambiguity and closes the
remaining model-addressable case preflights.

## Historical release: 2.79.15 — FS energy and DeepX Part1 preprocessing

Build/workflow: `v2.79.15-fs-energy-and-deepx-part1-preprocessing`.

Version 2.79.15 kept the direct Full-System M.2-off/on measurement, made `FS`
plus `command` explicit in the energy defaults and fixed the DeepX
Classification Part1 ImageNet mean/std adapter path.

## Historical release: 2.79.14 — simple Full-System M.2 idle calibration

Build/workflow: `v2.79.14-simple-full-system-m2-idle-calibration`.

Version 2.79.14 introduced the direct calibration without operator name,
prepare step, method manifest, calibration binding or SHA chain. Physical state
checks, interlocks, recovery and the atomic registry commit remain in force.

## Historical release: 2.79.13 — platform-power operational repair

Build/workflow: `v2.79.13-platform-power-calibration-operational-repair`.

Version 2.79.13 repaired installed-scope and editable-launcher behavior while
retaining the manifest-preparation calibration path that v2.79.14 supersedes.

## Historical release: 2.79.12 — platform-power calibration provenance closure

Build/workflow: `v2.79.12-platform-power-calibration-provenance-closure`.

Version 2.79.12 verifies a configured full-system energy-method manifest before
any platform mutation, passes the exact manifest and SHA through M.2-off/on
measurements and binds their evidence hashes into the saved calibration. It
uses the real `m.2` u.RECS token with a narrow legacy-default migration, exposes
final-gate reasons and verifies updated installations in installed manifest
scope without weakening strict release/archive verification.

## Historical release: 2.79.11 — platform-power, energy and evidence closure

Build/workflow: `v2.79.11-platform-power-energy-evidence-closure`.

Version 2.79.11 removed the global platform banner, exposed three independent
registry-backed setup cards and bound accelerator-idle normalization into
TensorRT Full scientific reporting. The first hardware calibration exposed the
manifest and controller-token gaps closed by 2.79.12.

## Historical release: 2.79.10 — platform-power release closure

Build/workflow: `v2.79.10-urecs-platform-power-release-closure`.

Version 2.79.10 aligned package, workflow, smoke, acceptance, updater and the
current long-run aliases and introduced the canonical shared/exclusive
workflow/platform interlock. Its initial single-selected-setup Platform Power
surface is superseded by the registry-backed three-card UI in 2.79.11.

## Historical release: 2.79.9 — u.RECS platform power and M.2 idle calibration

Build/workflow: `v2.79.9-urecs-platform-power-and-m2-idle-calibration`.

Version 2.79.9 introduced one-shot u.RECS, Jetson SSH and accelerator-presence
status plus the initial M.2 idle-power calibration implementation. It is kept
immutable; release-consistency and safety corrections are published as 2.79.10.

## Historical release: 2.79.8 — YOLO11 gate-profile schema closure

Build/workflow: `v2.79.8-yolo11-gate-profile-schema-closure`.

Version 2.79.8 made the published YOLO11 R8B gate profile admissible under the
strict evaluation-profile schema. Release acceptance loaded that exact profile
through both the validated evaluation-profile loader and the runtime snapshot
loader before hardware could start.

## Historical release: 2.79.7 — YOLO11 six-path runtime-identity closure

Build/workflow: `v2.79.7-yolo11-six-path-runtime-identity-closure`.

Version 2.79.7 bound every YOLO11l Full/b067 gate result to its exact Hailo-8,
Hailo-10H or DeepX runtime identity. Generic synchronous latency remained
diagnostic, while Native Full required the canonical runner, exact completion
counts and measured makespan FPS.

## Historical release: 2.79.2 — concurrent Native and B500 evidence reconciliation

Build/workflow: `v2.79.2-native-concurrent-three-stage-b500-evidence-reconciliation`.

This release makes the previously canary-proven YOLOv7 Hailo-8→TensorRT
P1/P2/Post pipeline installable through the normal runner and adds the
read-only evidence fixes derived from the v2.78.4 seven-model audit. A sealed
pre-dispatch required scope cannot shrink after compiler failure; mirrored
provenance rows remain distinct representations of one logical measurement;
Central Quality is joined by exact request SHA and logical identity; companion
controls and Quality-N/A P2 rows receive separate denominators. Hailo compiler
attempts are append-only and retain the actual last failure, and explicit
`hard_timeout_s=0/off` disables only the hard watchdog, not heartbeat or manual
cancellation. The historical audit is not promoted to PASS.

`2.79.1` was the Native three-stage and Quality-oracle separation release.
It preserves the 2.78.4 campaign, Gate-A, ranking and resume contracts while
adding the physical `p2_output` endpoint, the application
`completed_detection` endpoint, contract-bound postprocessing adapters and
postflight Quality attestation outside the measured hot loop. The canonical
Hailo-10 profile identifier is `hailo10_to_tensorrt`, and the seven-model
launcher validates a normal venv Python symlink semantically through
`sys.prefix`. No ranking formula, candidate universe, Quality threshold or
compiler recipe is changed.

`2.78.4` is the narrow Gate-A planned-stop audit closure. The launcher may
classify the configured stop after backend-artifact generation as successful
only when the terminal Gate-A receipt and workflow evidence agree on an
accepted anchor. An arbitrary nonzero workflow exit, missing or contradictory
evidence, and every unplanned partial run remain fail-closed. The historical
`v2783` Gate-A profile, build-evidence index and output identities are retained
unchanged so already attested common-anchor artifacts can be reused. The patch
does not change the ranking method or the deterministic 20-candidate audit
membership. For the bundled seven-model run, the attested YOLO11 `b067`
deployment anchor is verified and executed first; the independent audit then
runs unchanged, with overlap deduplicated and both roles recorded. The frozen
profile uses B500 screening quality, six Generic backend profiles and a
resource-gated Hailo-8/Hailo-10 pair build. Native and Energy remain disabled
for this campaign. A detached overnight launcher freezes the hardware registry,
revalidates the retained Gate-A output and exact Hailo cache receipts, and then
starts a fresh run without rerunning Gate A.

`2.78.3` restores the existing C++ Native FIFO fastpath as the Detection
`raw_model_outputs` performance endpoint and retains the frozen Python
Completed-Task path as a separate application endpoint. The two endpoint
phases run sequentially with fresh runtime instances so Decode/NMS backpressure
cannot contaminate the raw pipeline makespan. Exact shared HEF, TensorRT engine,
image, prepared input, boundary and raw-output identities are required before
the combined result is admitted. The existing Generic Runner, candidate
selection, ranking freeze, Quality gates and earlier bridge evidence are not
changed. A separate offline instrumentation command profiles the Completed-Task
host tail from sealed raw-head dumps.

The integrated build additionally recovers exact, read-only Hailo compiler
evidence from the retained B5 run and keeps runtime evidence separate. Its
packaged YOLO11 Gate A is Part1-only, Hailo8-first, cache/evidence-first and
strictly bounded; it cannot cold-build Hailo10H before a Hailo8 pass and blocks
the unchanged B5 after budget exhaustion. The same build repairs the Generic-
Hailo/Native quality-selection dependency, DeepX model-versus-sidecar identity
and the ResNet quantized-boundary aggregate while preserving hard mapping,
shape and structural failures.

`2.78.2` fixes the real historical reference-descriptor compatibility gap in
the 2.78.1 read-only verifier and replaces future mutable management CPU
reference output with source-contract-addressed immutable snapshots. The
producer, active workflow loader, existing-evidence verifier, and scientific
replay bind path layout, source contract, file size, SHA-256, storage markers,
and non-symlink regular-file identity. A source-contract change publishes a
new file and leaves every earlier snapshot untouched; a repeated contract is a
verified cache hit, while the same contract producing different bytes is a
hard collision. Bound legacy fixed-path evidence is supported without
weakening these rules.

`2.78.1` adds a read-only verifier that cross-binds already produced
evaluation, Hailo structural-canary, and Full-ONNX self-reference evidence.
The resulting receipt is a retrospective hash-bound snapshot with unchanged
claim scope, not a prospective protocol freeze or a physical immutability
guarantee. It does not rerun hardware, compilation, B500 quality, ranking, or
adapters. Every recorded request, candidate, and canonical CPU-reference file
must still exist at its historical SHA-256. If a later producer reused and
overwrote the reference path, the missing historical byte-container yields
`INCOMPLETE` even when its canonical prediction fingerprint is unchanged.
Artifact-index rows, metric arithmetic, frozen raw-output signatures,
comparison endpoints, and completed detections are validated fail-closed.

`2.78.0` is a report-scope correction. A successful Full-ONNX self-reference
probe now claims only that completed-task detections for the tested input pass
the configured similarity policy. It explicitly does not evaluate or override
dataset-wide AP/accuracy. JSON status, diagnosis, policies and thresholds are
unchanged; ranking, adapters, HEFs, preprocessing, decoder/NMS contracts and
existing hardware evidence remain frozen.

`2.77.15` is the bounded YOLO11 Completed-v2 Full-reference projection. A
verified frozen YOLO11 Native raw-head endpoint using `ultralytics_regcls` may
be compared with the canonical decoded Full-ONNX `[1,84,8400]` reference.
Model family, execution mode, decoder identity, tensor evidence, geometry and
Completed-v2 attestations remain mandatory; YOLOv7 and unsupported formats
remain fail-closed. The release enables comparison only and makes no numerical
parity claim without an actual probe result.

`2.77.14` is the two-model structural-canary payload closure. Shared COCO
validation images may be regular filesystem hardlinks. The producer now reads
each admitted path through a no-follow file descriptor and emits independent
regular byte members; on-wire Tar hardlinks and symlinks remain forbidden.
The full source-path attestation remains in the receipt while stdout is compact.
No compiler, workflow, adapter, ranking or quality-evidence behavior changes.

`2.77.13` is the backend-semantic-smoke and reporting closure. It adds
hardware-independent, read-only A/B gates for prepared input identity,
original/fixup ONNX CPU parity and YOLO11/YOLO26 raw-head host-tail parity,
plus an optional small artifact-reuse hardware canary. It also separates exact
Full-only technical completion from scientific quality decisions, projects
Quality decisions truthfully into reference comparisons and labels skipped
fast-fail bootstrap bounds as not computed. Ranking, thresholds, compiler
recipes and captured quality evidence remain frozen.

The permanent semantic smoke is invoked with `--spec FILE`; `--require-pass`
makes missing optional dependencies a failing gate. The separate optional
Hailo-8 canary accepts only existing receipt-bound Full artifacts, never
compiles, and claims structural artifact/runtime completion rather than
numerical parity or accuracy.

`2.77.12` is the retry closure for the sealed targeted Full-quality resume. It
normalizes YOLO11 pre-NMS SHA-256 identities, gives the exact targeted rebuild
decision precedence over failed archived target checkpoints, and audits an
already complete exact repair without another workflow or hardware dispatch.
Must-reuse, no-build, Ranking, Native and Energy guards remain fail-closed.

`2.77.11` is the historical-artifact closure for the sealed targeted
Full-quality resume. It admits correctly indexed zero-byte artifacts, strictly
attests only the known successful classification stderr omission, and creates
empty stderr artifacts on future successful dispatches. Dynamic five-to-seven
preservation, workflow-outcome checks and must-reuse-stage guards remain
fail-closed. It adds no Canary, profile, compiler, Ranking, Native or Energy
layer.

`2.77.10` is the attestation-and-recovery closure for the sealed targeted
Full-quality resume. It read-only admits only semantically and index-bound
downstream supersession, precisely recovers the known abandoned 2.77.9
session, and isolates transport by repair attempt.

`2.77.9` is the admission-and-transport repair for the sealed targeted
Full-quality resume. It clears the archived fresh-run creation guard while
retaining the required Standard-mode check and isolates its disposable remote
transport workspace from archived requests. Every exact endpoint,
preservation and no-build invariant from 2.77.8 remains sealed. It adds no
Canary, profile, compiler, Ranking, Native or Energy layer.

`2.77.8` is a bounded Full-detection quality closure on the same frozen ranking
and measurement protocol. It reuses the sealed timed Hailo raw-head order,
represents YOLO pre-NMS matrices explicitly, and resumes only missing quality
identities while preserving completed results. It adds no Canary, profile,
compiler, Ranking, Native or Energy layer.

`2.77.7` was a bounded Phase-5 adapter closure on the same frozen ranking and
measurement protocol. It preserves explicit physical Hailo targets, including
an empty DeepX-only selection, and adds the parity-proven YOLO11 DFL16/C80
Native-Full route. It adds no candidate, threshold, Quality, energy, reporting
or hardware-gate layer.

`2.77.6` was the preceding bounded parser-endpoint closure on the same frozen ranking
policy. It replays the automatic DFC path before any override and permits only
one exact `base_conv` retry whose archived ONNX outputs and tool-owned Identity
aliases are fully attested. It adds no candidates, hardware work, HEF build,
ranking, Quality or reporting layer.

`2.77.5` was the preceding bounded POSIX managed-venv symlink and direct
compiler-evidence closure. `2.77.4` was the preceding bounded
generator and managed-venv closure on the same frozen
ranking and campaign protocol. A `forced_cases` declaration is the exact
generator attempt set except for already-attested native-capability
replacements; generator-time backend failures cannot search unrelated cases.
An explicit Full skip also blocks prepared and mandatory early-Full paths.
Managed DFC child launches receive the venv `PATH`, so vendor executables are
resolved consistently. It adds no ranking, schema or reporting layer.

`2.77.3` is the preceding bookkeeping-only closure. It carries explicit Hailo
Full `requested=false` markers through reuse, promotion and service-plan paths
so unrequested Full HEFs are not copied or queued. Missing or true markers
remain fail-closed.

`2.77.2` is a narrow Hailo orchestration and live-acceptance fix on the same
ranking freeze. It makes `hailo_build.build_full: false` authoritative through
RunPlan, generation runtime and orchestration, including matrix variants, and
packages a fail-closed ResNet50 b052 Part-1 pair canary. It changes no ranking,
candidate-generation, Quality, Energy or scientific protocol.

`2.77.1` is a report-only replay hotfix on the ranking freeze established by
`2.77.0`. The one permitted Development ranking revision remains globally
`cut_bytes_only`. Smaller cut boundaries rank first; equal byte counts are
ordered by Boundary ID and then Case ID. Candidate provenance remains the real
ONNX-boundary analyser, and stratified windows, seed, quotas, minimum gap and
capability rules are unchanged. Alternative methods remain report-only
scientific comparators. The hotfix reconciles a false endpoint-completeness
value from an archived normalized row only when the explicitness marker is
absent and an exact, valid central request projection supplies a passed
endpoint attestation whose hash matches the complete contract. Genuinely
explicit contradictions remain fail-closed, and a report
counter makes every migration visible. It also rebuilds the Native combined
summary through the canonical exact validation join after offline validation,
instead of leaving those bindings only in a sidecar; reconstructed identity,
FPS and repetition evidence must match the preserved performance summary
exactly. It repeats no hardware, compiler or Quality work and does not reopen
ranking.

`2.76.2` was the bounded evidence-closure maintenance release for the completed three-model
Development audit. It preserves every predeclared endpoint with its payload
validity, explicit terminal state, or unresolved/open evidence status,
separates technical, Quality and claim axes, and reprojects old runs read-only
into a separate output root. Nonzero runner return codes are fail-closed even
when a result file already exists, and successful backend-local artifacts are
retained when a sibling backend fails. Its repaired replay passed the one
permitted Development decision gate and selected global `cut_bytes_only` for
the subsequent v2.77 freeze.

The never separately deployed `2.75.50` intermediate block is incorporated in
the v2.76 line. It restores exact setup identity,
separates diagnostic, Quality-screened and strict claim cohorts, permits
diagnostic correlations on explicitly incomplete measured subsets while
keeping global Top-k/Regret fail-closed, and preserves enriched rows for exact
Generic/Native pairing. Hailo-8 and Hailo-10H builds for one candidate may run
as a resource-checked pair only through two verified managed-venv subprocesses,
with deterministic merging and serial fallback otherwise. No shorter compiler
timeout is introduced.

The 2.76.2 maintenance binds Cross-Runner reporting to the frozen 24-row
Native intersection, restores exact Generic request identities from existing
quality evidence, separates technical, Quality and claim cohorts, publishes
micro/per-backend concordance plus Hit@1/Regret@1, fixes the Development
leader comparison on qualified actual strata and publishes stable final replay
paths. It repeats no hardware, compiler, B500 Quality or Generic measurements.
The patch additionally closes the normalized full-run metric-alias path that
the archived compact-row replay did not exercise and keeps Markdown identity
diagnostics identical to the JSON report.

`2.75.49` is the narrow DeepX-Full follow-up to the v2.75.48 YOLOv7 anchor.
When the DeepX-Full semantic backend and endpoint omit the same redundant
model digest, the exact registered `yolov7_paper` model/decoder supplies that
binding. Every explicit digest remains mandatory-to-validate and must agree
with the other explicit values and the registry. Invalid or conflicting
values remain fail-closed. The fresh v2.75.49 profile keeps b044, B500,
decoder/anchors, datasets, Quality, Native Full and all hardware settings
unchanged while Energy and ranking remain off; no v2.75.48 result or
prediction cache is reused.

`2.75.48` was the bounded YOLOv7 anchor closeout. Bare and `sha256:`-qualified
COCO annotation identities compare canonically. Native completion and DeepX
Full no longer require the already verified model digest to be copied into a
runtime source endpoint; if one is present it must still be valid and match.
The decoder binds to the registered model contract, while `resolve_model`
continues the single authoritative byte-identity check before mutation.

The CPU A/B probe with official `pycocotools.COCOeval` remains the hard
pre-anchor gate. The single b044/B500 technical anchor uses paired internal
Quality and defers Official COCO to the final Detection run. Energy, ranking,
B1000 and further decoder or accelerator tuning remain outside this release.

`2.75.46` closes the Native-Full ONNX-attestation, Standard Quality projection
and DeepX calibration-size comparison defects found in real evidence. The
selected ONNX-capable engine interpreter supplies only primitive graph-output
metadata; the parent remains the receipt/hash/signature authority and records
child exception diagnostics. Standard setup-local Full-quality identities are
projected separately from ordinary Generic diagnostics. B500/B1000 comparison
normalizes only the treatment-derived output-contract file hash and cascading
sealed hashes after strict arm-local validation; semantic endpoint fields stay
invariant.

`2.75.45` makes large score-independent audits usable from the GUI without a
hidden environment override. Fresh audit starts show and require confirmation
of the real audit/union/row scope and Native/Energy settings. The 20-GiB
managed TensorRT setting now bounds only retained non-current caches; the
selected run's active working set remains protected by the physical-capacity
preflight and keeps its stable Resume/reuse namespace. Unsafe cache ownership
or activity remains fail-closed. New profiles explicitly select
`imagenet_mean_std` for DeepX classification, while legacy profiles without the
field preserve `current_scale_only`.

`2.75.44` repairs the B1000 Validation-authority comparison by retaining the
complete 50,000-image source-manifest verification and projecting its exact
frozen 500-image runtime cohort before comparing manifest, image-ID and
ground-truth authorities. It does not rewrite a manifest or image and does not
change the B500 pin, B1000 selection, preprocessing, Quality policy or hardware
procedure. Missing, duplicate, drifted or ambiguous projected rows remain
fail-closed.

`2.75.43` restored compatibility with the exact legacy portable dataset
item-identity serializer used by retained production evidence. Legacy and
normalized four-field aggregates are accepted only when they match the rows;
the emitted portable identity remains normalized. File, payload, component,
inventory, frozen-authority, B500, B1000, disjointness and subset gates are not
weakened, and no manifest or dataset file is rewritten.

`2.75.42` repaired real-evidence admission without changing the scientific
calibration-size axis. An explicit installed-tree manifest scope tolerates
only unmanifested regular top-level user profiles while retaining byte-exact
release ownership. The B500 authority accepts the production ImageNet export
kernel identity. Source updates compare content with `rsync --checksum`, retain
the existing virtual environment and refresh only this project's distribution
metadata, editable path and declared console entry points through a fail-closed
standard-library helper; no package download or build backend is required.
Setup-local TensorRT Quality companions acquire exactly one physical setup and
Full endpoint identity in the effective plan, preserve that identity through
dispatch and management admission, and make a missing required result terminal
for Standard and Final. The failed nine-invocation night run is not admitted
as Final-Quality evidence; the controlled B1000 canary remains the next
hardware step.

`2.75.41` isolates the next DeepX variable after the preprocessing result.  It
keeps the corrected ImageNet Mean/Std adapter, ResNet50, EMA method, DeepX
optimization level 0, compiler identity, ordered 500-image validation cohort,
Quality policy and setup-local TensorRT control fixed and changes only the
train-derived calibration cohort from 500 to 1,000 images.  The 1,000-item
class-stratified cohort covers every ImageNet class once; the verifier checks
the actual manifest identities and requires the previous 500-item cohort to be
a strict subset.  A new exact-v2 cache namespace prevents accidental reuse of
the 500-calibration DXNN while retaining restart-safe reuse of an already
completed 1,000-calibration build.

`2.75.40` introduces an explicitly paired DeepX classification preprocessing
experiment. Both arms bind the same model, ordered calibration and validation
cohorts, compiler identity, Quality policy and setup-local TensorRT control;
only the numeric preprocessing contract and an isolated exact-v2 DXNN cache
differ. `imagenet_mean_std` is implemented as a content-addressed ONNX adapter
before the unchanged source graph, while DX-COM retains its proven loader.
The release also keeps the signed physical `native_tensorrt` producer alias at
the Full-only export boundary and stores the logical `tensorrt` name only in
the nested plan identity.

`2.75.39` distinguishes a deterministic SSH failure before any remote
mutation from an indeterminate transport loss after a remote lease. The
pre-mutation case suppresses result collection, records no remote lease and
preserves the concrete connectivity cause as the primary error. Once a lease
may exist, unresolved cleanup remains fail-closed and quarantines the run. The
packaged `profiles/resnet50_v27539_deepx_full_quality_canary.yaml` retains the
same two-endpoint effective plan as 2.75.38.

`2.75.38` repairs the DeepX-only Full Quality execution path. The first
2.75.37 hardware continuation showed that `deepx_m1_full` was incorrectly
routed through generic ONNX Runtime on CPU and then correctly rejected for
missing DeepX runtime identity. The endpoint now dispatches into the real
DXNN/DX-COM path, requires `deepx_dxnn_sha256:<sha256>`, and forbids CPU
fallback. Full-only Quality endpoints are not Performance measurements and
are excluded from those completeness/cardinality gates; exact Quality
evidence remains mandatory. The packaged
`profiles/resnet50_v27538_deepx_full_quality_canary.yaml` runs only DeepX Full
and its setup-local TensorRT Full Quality companion, with no Split, Native,
Energy, ranking, or performance claim.

`2.75.37` made the receipt-attested Hailo-Full preupload gate explicitly not
applicable when no Hailo-Full backend is requested. Requested Hailo endpoints
continue to require their exact verified HEF receipt and remain fail-closed.

`2.75.36` closed the three defects exposed by the
`resnet50_v27535_small_acceptance_20260812_210312` hardware run. The original
Hailo-10H to TensorRT `b119` runtime binding executed correctly as
`as_input`/`identity`, completed three independent 1,000-frame repetitions and
passed the 500-image task-quality gate. The downstream structural validator
nevertheless rejected runtime shape `[2048]` against canonical TensorRT shape
`[1,2048,1,1]`. It now admits only singleton-axis insertion/removal under the
sealed no-transform contract and retains all exact name, dtype, element, byte,
bridge, command, metadata and Quality-FIRST checks. Equal element count alone
is never sufficient.

TensorRT engine persistence is now independent of volatile suite provenance.
The existing suite key still protects transport and resume identity, while a
separate engine key binds canonical Full-ONNX bytes, engine-building options
and an attested TensorRT/GPU ABI. Candidate graphs remain content-addressed in
their leaf paths. Profile/campaign labels, tool release strings and vendor
compiler/cache diagnostics cannot select a new engine root. The first run may
copy eligible leaves from older managed roots only after self-hashed receipt,
source ONNX, engine, `trtexec`, precision/workspace and live GPU
deserialization verification. Source roots remain unchanged, and subsequent
warm runs do not repeat the legacy scan. Quality-FIRST preparation verifies a
migrated receipt through the builder's `--no-build` path and rebuilds on any
failure.

The `r2` build hardens that ABI and reuse path for the deployed systems. It
attests CUDA-visible device 0 through the read-only CUDA Driver API, so Jetson
Orin does not depend on `nvidia-smi`; multi-GPU keys bind the selected target
rather than unrelated inventory. Empty GPU or linked-library evidence remains
fail-closed. Existing stable namespaces use a metadata-only retention fast
path while inside configured limits. A verified Quality-FIRST `--no-build`
hit preserves engine and receipt bytes, hashes and timestamps, retains the
receipt-bound build command as provenance, and records that no compiler was
dispatched for the current invocation.

When Native Energy was not requested, its evidence axis is now explicitly
`not_applicable`: the coverage contract is inactive and ledger, identity,
coverage and count values are null rather than false or zero. Requested strict
Energy and explicit all-split requirements remain fail-closed. The release
ships `profiles/resnet50_v27536_small_acceptance.yaml` with the same frozen
ResNet50 `4/3/0` acceptance scope and Energy disabled.

`2.75.35` separates a frozen candidate's generation outcome from the frozen
selection identity. Every prospectively selected candidate must still be
accounted for exactly once and in deterministic order, but a compiler- or
capability-rejected candidate is now retained as an explicit unsupported audit
observation instead of forcing `accepted == frozen union`. Accepted and
rejected identities must form a complete, disjoint partition; missing,
duplicate, reordered or outside-backfill identities remain hard failures.

The Evaluation Profile editor also preserves an explicit `min_gap: 0` and no
longer replaces profile-owned audit size, minimum-valid count or seed when a
run mode is loaded or changed. The release ships
`profiles/resnet50_v27535_small_acceptance.yaml`: Standard, ResNet50 only,
Audit 4, minimum valid 3, deployment 1, gap 0, the seven established logical
profiles, Native/Full enabled, and Energy disabled.

`2.75.34` repaired Hailo-10 channel-vector boundary metadata. A runtime shape
that differs from canonical `[1,C,1,1]` only by squeezed singleton axes is
byte-layout invariant and resolves to `as_input`/`identity`. The rule does not
flatten spatial tensors or accept arbitrary equal-element-count shapes. A
candidate-local layout-resolution error is persisted as an explicit no-claim
row while the suite continues; identity, package, cache-root and other global
contract failures still abort. That runtime repair is retained unchanged in
2.75.35; the failed 2026-08-12 run stopped upstream and therefore did not
validate it on Hailo-10 hardware.

`2.75.33` made the stable, deduplicated predeclared audit/deployment union the
generator's exact scope. The deployment shortlist no longer truncates an
already frozen audit, adjacent frozen boundaries are not removed by a second
minimum-gap pass, and outside backfill is forbidden. Candidate Plan,
Prediction and Candidate-Universe identities and hashes are checked before
generation; truncation, reordering, identity drift and unclassified partial
materialization fail closed. Explicit generator rejection is now a recorded
outcome, not a selection mutation.

Generic Standard rows may support Development ranking only for an actual
predeclared `score_independent_audit` case with Quality PASS. Final, missing
preset identity, explicit audit vetoes and all claim-elevation paths remain
excluded. Compact Debug Packs retain bounded authoritative model validation
summaries. Targeted retry of an archived terminal Native-Energy row is not
included; corrected audits start fresh with Energy disabled.

`2.75.32` repairs the real Development audit path after the first hardware
run. Effective planning now counts score-independent audits for Development
models and exposes the audit/deployment union range. Native producer rows use
their completed-task comparison identity for cross-candidate ranking, while
incompatible execution strata stay isolated. Generic Standard screening
passes may support Development-only ranking analysis without becoming claim
eligible. Audit summary status reflects the frozen request and usable metrics,
and compact Debug Packs omit only known byte-identical aliases with a verified
canonical hash record.

`2.75.31` was the corrective release for diagnostic packaging, Native ranking
and Evaluation Profile editing. Debug Packs now use one shared compact policy,
retain the complete main workflow log, carry bounded hashed audit evidence and
avoid raw tensors, rendered figures, resource copies and replay-body mirrors.
Native performance observations are joined before ranking within their full
execution stratum. The profile editor exposes one model usage field and derives
role, generalization, validation tier and candidate-universe mode internally;
score-independent audit size, minimum and seed are global selection settings.

`2.75.30` changes the user-facing Final run mode into **Final Quality
(Standard+)**. It is mechanically derived from Standard and changes only the
task-quality fields: Classification and Detection validation rise from 500 to
5,000 items each, `quality.profile_id` becomes
`task_quality_final_5000`, `quality.dataset_tier` becomes `final`, and
task-quality bootstrap repetitions rise from 500 to 5,000. Build and runtime
effort, Native and Energy defaults, reporting, cache behavior, sampled
integrity and campaign enforcement remain Standard.

Consequently, the ordinary Final Quality path is a development-mode workflow
with warning enforcement and sampled integrity (24 registry items), not a
sealed strict campaign. It has no mandatory `prepare-evaluated-matrix`,
campaign `preflight`, manual seal, repeated full-manifest content scan or
canary. The selected 5,000-item quality subsets still carry their normal
content/evidence hashes. Energy remains a separate user switch, disabled by
default and otherwise identical to Standard.

The central run-mode registry advances to schema 12. Untouched older Final
defaults migrate to Final Quality; user-customized leaves remain customized.
`onnx-splitpoint-run-modes show final` displays the effective registry and
`onnx-splitpoint-run-modes reset final` restores the packaged current default.
The 2.75.28 strict campaign/bootstrap and canary commands remain available only
for an explicitly selected archival or advanced protocol.

`2.75.28` makes the Final software bootstrap executable without changing the
2.75.27 scientific claim boundary. Final templates and the canary bind the
exact `yolov7_paper` model ID and use pipeline-contract sources derived from
the canonical runtime preprocessing and postprocessing definitions. The
campaign CLI returns success for `final_ready` and `development_ready` while
remaining fail-closed for `blocked` and unknown states.

Full-system energy uses `inherited_validated_method` provenance for the Joris
measurement implementation instead of manufacturing a new calibration claim.
The runtime binds FS scope, setup address, channel 0, acquisition rate and
fast-firmware expectation before collector/workload admission. The three
setup bindings remain distinct and energy repeats remain serial.

`2.75.27` gives every new Final profile an exact scientific scope. The normal
`evaluated_matrix` scope retains all dataset, pipeline, model identity, final
quality, Native/Full, repeated performance and serial full-system-energy gates
without pretending to validate unseen-model ranking. The optional
`ranking_generalization` scope retains the complete prospective hold-out,
protocol-freeze, approval and fitted-model contract. Unknown or explicitly
empty values fail closed; legacy profiles infer their former intent. Both
hold-out spellings now canonicalize identically at execution time.

The 2026-08-10 warm replay completed technically but rebuilt every TensorRT
namespace because volatile Hailo build `elapsed_s` values entered the portable
suite key. That field alone is excluded in 2.75.27; graph bytes, runtime and
builder contracts remain bound. A first post-fix canary establishes the new
namespace, and an identical second pass must show zero physical TRT builds
before the full Final run is authorized.

`2.75.26` accepts the intentional logical `tensorrt` to signed
`native_tensorrt` alias only at the final Central-Quality mirror boundary.
Producer identity and signatures remain strict. It also changes the default
Native Detection mean-IoU threshold from 0.90 to 0.85 while retaining the 0.80
match-ratio, 0.50 pair-IoU and 0.25 confidence settings. New run-mode, GUI,
profile and remote snapshots materialize 0.85; archived explicit 0.90 policies
remain unchanged. Central AP/Top-1 gates, cache/artifact identities, schemas,
builds and resume behavior are unchanged. The existing policy hash binds the
new default automatically.

`2.75.25` accepts the intentional logical `tensorrt` to signed
`native_tensorrt` alias at the earlier Quality-chain mirror boundary. Producer
identity and signatures remain strict. The nine Native TensorRT Full rows stay
setup-local across Hailo-8, Hailo-10H and DeepX; the single DeepX owner remains
limited to generic TensorRT performance. Partial Native coverage demotes both
Energy admission and result rows to `screening_comparable=false`. That release
did not change any cache, schema, hash, profile, threshold or resume contract.

`2.75.24` restores the optional Full-only Quality canary to the ordinary
Standard Hailo artifact identity (`balanced`, opt1, relaxed, 500 calibration)
and runs that canary in cache-only mode. Both required Full HEFs must be
receipt-attested by the existing artifact promoter before bundle creation or
remote upload. Missing or invalid evidence fails the artifact stage and blocks
all upload-dependent stages without introducing a new cache/hash schema or
resume path. Normal Standard runs are outside this gate. The synchronized
source updater additionally preserves custom top-level profile YAMLs.

`2.75.23` repairs the saved Standard-profile path for the explicit Full-only
Quality canary. Run-mode materialization preserves `quality_canary`, and the
Full-recipe validator accepts schema-valid `variant: full` declarations while
still binding the declared backend. A sealed YOLOv7 launcher asserts exactly
four Full-quality rows, zero generic rows, two remote invocations and disabled
Native/Energy before hardware. It checks local capacity, runs 500 calibration
images, all 5,000 COCO-val images and 5,000 paired bootstrap repetitions, then
creates and verifies a replay-complete Debug Pack outside the EvaluationRun.
It does not change any Accuracy threshold or issue performance claims.

`2.75.22` adds AP75 to the paired Detection evaluator, fails closed when any
configured guardrail is missing and advances both the Detection algorithm and
cache-result contracts. It can replay all six existing central Quality rows
from prediction JSONs without inference or accelerator access, while also
publishing an exact four-row Full-only projection. Replay output and cache are
written outside the historical EvaluationRun. Scientific reporting retains
setup/request identity and separates technical execution, aggregate Quality
decision and scientific status. An explicit Full-only Quality-canary scope
prevents generic Split/Composed/performance rows, and debug packs include the
relevant HEF, TensorRT-dispatch, remote-matrix and strictly referenced replay
provenance. Thresholds and archived evidence are unchanged.

`2.75.21` remains the setup-local TensorRT Quality-dispatch repair exposed by the
immutable 2.75.20 three-model Full-only Smoke
`resnet_yolo26s_yolo7_20260807_205238`. The dispatch plan now binds one
`ort_tensorrt` Quality companion per model to each of Hailo-8, Hailo-10H and
DeepX, while DeepX remains the sole TensorRT performance owner. A local
fail-closed preflight verifies that contract before upload or SSH. Canonical
`ort_cpu` references remain on central management. Summaries use the expected
18-row denominator, setup-local non-attempts are classified precisely, and
Split-Quality is explicitly not applicable in a Full-only run. No archived
artifact, Accuracy threshold or physical result is changed.

`2.75.20` remains the preceding remote-boundary and plan-finalization repair.
It moved Hailo contract promotion into a management-free remote module and
sealed Stage, Task and Quality fields before synchronizing the final run-plan
hash across executable and formal aliases.

`2.75.17` remains the preceding narrow Vendor-Full compatibility repair. The
redundant `runtime_quality_gate_policy_sha256` mirror is observable but is not
part of the Vendor binding identity. For Hailo-8/YOLOv7, the checker consumes
the authoritative loader's canonical `raw_head` endpoint instead of requiring
the historical `raw_detection_head` alias. All purpose-bound Run selection and
storage safeguards from 2.75.16 remain in force.

The release disables the volatile persistent remote suite-bundle cache even if
the former opt-in environment switch is set. TensorRT engine reuse remains,
with an exact portable BenchmarkSet/plan/ONNX namespace that excludes
timestamps, Run IDs and absolute Run paths. Non-mutating block/inode/mount
preflights apply before the first remote mutation and before either transfer
mode. Their selected-plan bound includes transport, TensorRT engine/build
space, outputs, logs and a 2 GiB default safety reserve; same-host workers are
serialized within one workflow process. A terminal storage error is preserved
without follow-on remote writes or leased diagnostics.

The managed TensorRT root is bounded by default to six namespaces and 20 GiB.
Retention first revalidates and locks every candidate, then executes only a
complete admissible plan. Historical Runs and legacy, foreign, active,
receipt-less or unsafe caches are never deleted; only inactive Tool-owned,
receipt-/engine-verified entries under the v2.75.16 managed root may be
rotated. A full cold preflight
still precedes retention, so this is not an implicit cleanup pass for an
already rejected filesystem.

The latest 2.75.21 YOLOv7 Quality canary completed technically and produced the
prediction bytes required by the 2.75.22 offline replay. A technical pass is
not a positive Accuracy decision and does not by itself establish scientific
readiness.

`2.75.12` makes the real Classification result shape authoritative in the
bounded cache canary. Top-level `ok: true` is mandatory; the redundant
top-level `status` alias may be absent, but any emitted value must agree with
success. The Runner and repetition records retain their mandatory
`status: "ok"` fields, and every existing execution, compiler-fence and workload gate
remains unchanged.

`2.75.11` makes the complete runner result the single canonical terminal view
of a Native execution. A reduced direct-fallback projection may no longer win
deduplication merely because recursive discovery found it first. The canary
therefore decides execution success from the canonical current-run envelope;
redundant result/command cross-links remain diagnostic. In `cache_verify_only`,
locally validated complete artifact sets are selected by semantic compatibility
instead of a hard-coded historical producer digest. No set is mixed, compiler
fences and the exact diagnostic workload remain enforced, and Standard/Final
paths retain their stricter receipt contracts.

`2.75.10` resolves the preserved Smoke cache even when the same sealed binding
is registered in more than one suite root. Equivalence requires both the known
historical producer binding and the complete eight-role artifact-set digest;
different valid builds are excluded and their files are never mixed. The
field canary then completed the exact 10/2/1/2 hardware replay, but its terminal
collector retained an earlier reduced fallback row instead of the later
complete runner row and incorrectly reported failure.

`2.75.9` is the preceding Smoke-profile and multi-root discovery repair. Its
field canary found the correct Hailo cache and two verified TensorRT roots, but
blocked before inference because every multiple exact hit was treated as
ambiguous without comparing the complete sealed identities.

`2.75.8` repairs the broad Native runtime, artifact handoff, same-input
Detection validation and dependency-free cache-replay contracts. It also binds
endpoint-adapter implementation source hashes before confirmatory hold-outs can
pass a final-campaign preflight.

`2.75.7` repairs the bounded cache canary's remote Python closure. The exact
`cache_verify_policy.py` source is synchronized, SHA-256 verified and imported
before the remote Hailo backend that depends on it. A clean isolated-tree
regression imports that backend without access to the source checkout, closing
the `ModuleNotFoundError` observed on the Orin before cache replay and hardware
dispatch.

`2.75.6` is the preceding profile-driven start-contract repair. The wrapper's
explicit fresh Run ID is accepted and forwarded only after the resolved profile
proves `cache_verify_only`; normal profile-driven runs cannot use this override.

The retained exact-hit/fail-fast path resolves the Hailo compiler identity from
managed-venv distribution metadata
without importing or starting DFC, constrains generation to exactly the
attested boundary and Part1 scope, suppresses backfill/Full/direct fallback,
and emits a complete cache-miss diagnostic. Full Hailo receipt semantics are
revalidated immediately before Native dispatch. The source updater preserves
the installation venv and external state while deleting obsolete source files
into a recoverable backup.

`2.75.3` introduced the bounded compiler-free cache-canary policy on 2.75.2. Its
`cache_verify_only` policy permits verified cache restore and compatible Hailo
exact-v2-to-v3 migration, while every Hailo DFC, DeepX DX-COM and TensorRT
compiler dispatch remains fenced. The resolved model/profile/backend/setup/case
and runtime matrix is attested before run creation and checked again before
Native dispatch. A mismatch or cache miss is diagnostic and cannot authorize
Performance, Energy, ranking, or scientific claims. The public first canary is
`profiles/cache_verify_resnet50_b052_hailo8.yaml`.

`2.75.2` is the Native runtime-recovery patch on 2.75.1. It verifies the
actual remote import closure, persists hash-bound Hailo-8 child checkpoints,
continues later rows after one child failure, admits receipt-bound YOLOv7
source raw heads with an empty compiler `end_nodes` declaration, and permits a
terminal partial Performance matrix to continue as non-claim Energy
observations. Remote cleanup requires a collection receipt. One canonical
Native execution contract supplies plan, workflow, variant and child runtime
values; silent variant overrides are rejected.

`2.75.1` is the preceding runtime/cache patch on 2.75.0. It bridges the canonical
Detection `target_hw` field into TensorRT Quality, promotes Hailo receipts after
successful builds, seals one shared pre-timing DeepX tensor for Generic and
Native replay, and applies missing Quality evidence as a row-local claim veto.
Technically runnable Native Performance/Energy rows are no longer cancelled by
an unrelated incomplete Full-Quality matrix.

Hailo reuse is stable across Tool patch/minor releases: Tool version, Run ID,
timestamps and EvaluationRun paths are excluded from the key. The exact-v3
key additionally binds compiler net name, normalized input shapes and the RT-
metadata flag. A compatible 2.75.0 exact-v2 record is explicitly tested across
the 2.75.2-and-later boundary with compiler dispatch blocked and is backfilled
into v3 only after its
compiler-ONNX, SDK, calibration, preprocessing, build-payload, receipt and HEF
identities match.

`2.75.0` is a contract-repair minor release on the validated `2.74.1` source.
It makes image preprocessing task-bound: Detection uses centered letterbox,
RGB and pad `114`; Classification uses direct resize and ImageNet numeric
normalization. Output shape or format no longer selects geometry. The shared
contract and its SHA-256 bind CPU Central Quality, accelerator evidence, Hailo
calibration, cache keys and atomic HEF build receipts. Unsealed legacy HEFs and
old preprocessing caches are intentionally invalidated.

The release also closes the Full-evidence gaps exposed by the 2.74.1 Overnight
run. Runtime tensor attestation identifies Hailo YOLOv7 raw heads before timing,
DeepX BN6 uses the same class-aware Completed-v2 postfilter as its measured
hotloop, and the self-reference path consumes persisted, verified Completed-
NMS artifacts. Full Quality retains exact producer, model, dataset,
preprocessing and endpoint provenance; no fuzzy join was added. Missing
selections, zero requested rows, unavailable semantics and mixed Raw/BN6
contracts now fail closed.

Run-mode projection preserves configured Native and Native-Energy controls.
Disabled Energy is an explicit completed `not_applicable` state rather than a
missing stage. Smoke may report an explicitly selected missing producer cohort
as partial; Standard and Final fail. The effective validated case map remains
the denominator throughout variant execution.

`2.74.1` was the bounded Native-Energy/Completed-Quality projection repair on
the `2.74.0` recovery. That recovery retained P0.1 process ownership, P0.2
atomic checkpoints, and P0.3 Native Energy planning from the frozen `2.73.8`
pre-P0.4 lineage while keeping the withdrawn P0.4 Endpoint/Central-Join layer
absent.

`2.73.8` made Native Energy plan membership follow the technical predicate
`runtime_success AND energy_command_preflight_ok AND (full_baseline OR
split_has_valid_part2_input)`. Quality, Semantics, pairing, and claim status
are downstream annotations and cannot remove a measurement row. Generic
Energy stays disabled.

`2.73.7` added atomic lifecycle checkpoints around the existing Standard
Native path. Native Performance is committed only after the frozen 63-row
campaign (`3 × 21`) has been imported without missing, duplicate, unexpected,
or identity-drifted rows. Native Energy creates one authoritative row file per
planned measurement and distinguishes `not_started`, `running`, `completed`,
`failed`, and `cancelled`. A synthetic Resume reuses the terminal Performance
checkpoint and continues at the first nonterminal Energy row. Cancellation
never fans a parent pipe error out to rows that were not started. These
contracts apply to new runs/fixtures; the archived diagnostic run is not a
Resume target.

`2.73.6` added a kernel-backed single-writer lock per EvaluationRun, froze
the creation-time profile/effective-plan Resume contract, and recursively
cancelled workflow-owned process trees across nested POSIX sessions. Lock and
Resume mismatches fail before logs or archived Run files are changed.

`2.73.5` removed the withdrawn parallel hardware-acceptance implementation
developed in the intermediate 2.73.2–2.73.4 attempts. Those attempts copied
and replayed benchmark suites in a second coordinator and are not supported
release paths. The replacement launcher performs configuration only, then
invokes the existing Evaluation Workflow once. GUI and profile-driven CLI
starts share the same validated frozen profile snapshot and Standard values;
the runner rejects a wrong run mode or Resume/partial/result-import shortcut
before creating the run directory. Realistic regressions execute the nested
`legacy_suite` harness and its actual `b036`/`b044` case runner as a child
process. No CPU, SSH, Native, Quality, or Energy executor is duplicated.

The independent Full-Resume repair retained from the audit binds DeepX Full to
`dxnn` and Hailo/TensorRT Full to `hef`; unknown Full backends remain
fail-closed. A flaky completed-tail timing test now uses a deterministic
virtual clock without changing product timing code or thresholds.

`2.73.1` hardened local acceptance without changing runtime or scientific
semantics. Default Pytest discovery and every full-suite acceptance invocation
are bounded to the current `tests` tree, so old Tool copies embedded below a
long-lived workspace are not imported. The acceptance runner executes in its
own Bash process and verifies the checked-in source manifest read-only before
tests and before the deterministic archive build. It cannot log out the
calling SSH shell or silently regenerate hashes for an overlaid source tree.

`2.73.0` repaired the remaining Quality-evidence contracts before a fresh
Standard run. Classification uses exact integer hit counts at inclusive
margins and a narrow ULP fallback for other numeric sources. Cross-host input
identity is content-based, and Native Full plus YOLOv7 Split/Full validation
consume the attested Completed-Task result produced by the measured hotloop.
Complete negative `fail` and `inconclusive` decisions are now scientifically
complete but remain ineligible for positive claims. Missing, `unavailable`,
conflicting and technically invalid evidence remains partial or failed
according to run mode. The 2.72.7 runtime-success Native Energy admission rule
is unchanged. Tests are isolated from the user's home directory and clean
source archives use a strict current-release allow-list. Hardware acceptance
is still a separate gate and this source release does not authorize a final
Standard campaign by itself.

`2.72.8` is the narrow DeepX exact-row artifact-rehydration repair discovered
while checking one newly admitted 2.72.7 Energy row from an already cleaned
Standard run. The original DeepX semantic output existed and was validated;
normal post-Energy cleanup later removed the complete remote EvaluationRun
tree. Fresh runs measure before that cleanup. For a bounded archived preflight
or resume, 2.72.8 may now restore only the successful contract's contiguous
`uint8` HWC NumPy feed, source image, semantic manifests and their sealed
payloads below the frozen run root. Every byte is SHA-256 verified. DXNN,
TensorRT engines, interpreters, caches and Tool code remain ineligible for
restaging. The strict preflight, the 2.72.7 `45/45` runtime-success admission
policy, all Quality/pairing annotations and every claim gate are unchanged.

`2.72.6` is the Hailo-8 Python Detection preflight artifact-closure repair
discovered by the bounded 2.72.5 exact-row Resume. The exact known generator
contract now restores its sealed `CMakeLists.txt` and `main.cpp` in addition to
the already bound runtime data. The two source literals are parsed statically
from the packaged runner without import or execution, written only below the
Resume attempt, and admitted only after their bytes match the archived hashes.
Hailo-8 C++ contracts, Hailo-10, DeepX and unknown contract forms remain
fail-closed. Measurement, Quality, accuracy, Energy and claim semantics are
unchanged.

`2.72.5` is the backend-bound exact-row Resume repair discovered after the
bounded 2.72.4 Hailo-8 campaign gate. Hailo-8 Split contracts bind the single
`prepared_input` artifact. The released Hailo-10 role contract is unchanged,
and backend forms without a verified resume contract remain fail-closed.
Resume validates schema version and the exact backend/model/case/setup identity
before any remote probe. The selection-wide preflight and atomic merge remain
fail-closed, and verified unselected Energy rows are reused unchanged. Runtime,
Quality, accuracy, Energy and claim rules are unchanged.

`2.72.4` is the canonical Quality-contract and central-request mirror repair
discovered by the bounded 2.72.3 Hailo-8 campaign gate. Detection Quality uses
the canonical terminal ONNX output format and names even when the physical
Hailo endpoint consists of six raw heads. Physical endpoint attestation and
the timed frozen decoder/NMS completion contract remain separate and exact.
Raw `remote_runs` requests are suppressed only when the signed result-copy
manifest proves one setup-scoped, byte-identical canonical destination.
Missing, drifted, ambiguous, remote-only and cross-setup copies remain
fail-closed. Accuracy thresholds and Standard/Final gates are unchanged.

`2.72.3` is the Quality-identity and portable-boundary repair discovered by the
bounded Hailo-8 campaign-gate Smoke. It gives the selected profile an
independent immutable authority hash, recovers setup identity from canonical
Quality request paths, deduplicates byte-equivalent central-result mirrors,
and rebases only an exact sibling semantic dump when a producer path came from
another host. Hailo Full exports its exact compiled raw-head endpoint and uses
the timed frozen decoder/NMS invariant for per-image Quality evidence.
Smoke-only diagnostic Energy may measure an exactly bound negative observation
without making it claim-eligible. Accuracy thresholds and Standard/Final gates
are unchanged.

`2.72.2` is the mixed-runtime and Energy-contract repair discovered by the
short 2.72 hardware Smoke. It preserves the existing system TensorRT and Hailo
Python environments, combines them only in the Child process through
`site.addsitedir()`, verifies the import/source closure before runtime access,
prioritizes verified Completed-v2 evidence, and seals the exact Energy
admission, coverage and exclusion ledgers carried forward by 2.72.3.

`2.72.0` is the completed-Detection and evidence-contract release. It keeps
physical runtime endpoints separate from the completed task endpoint, assigns
different identities to transport bytes, semantic contracts and result
content, and separates exact evidence binding from observed Quality outcome
and scientific eligibility. Planned Energy completion and full-matrix Energy
coverage are reported independently. Existing 2.71.4 runs remain immutable
Screening references. The delivered source ZIP is generated only from a
verified source manifest by `scripts/build_source_release.py`.

`2.71.4` is the cleanup-safe completion of the contract-bound archived-artifact
restaging repair. The reproduced `153940` resume confirmed that normal cleanup
had removed the complete frozen remote run root while its canonical storage
parent remained present. After every required local source has been opened and
hash-verified, 2.71.4 may recreate only that direct-child run root and the
required nested directories, one component at a time. The storage and run-root
device/inode identities are pinned across mutation, and every required remote
artifact receives an authoritative final probe before cohort admission.
Symlinks, non-directories, path escapes and identity drift remain fail-closed.
The Native and Energy hotloops are unchanged.

`2.71.3` introduced frozen resume execution context, exact
contract-bound source resolution, atomic file replacement and a
selection-wide preflight. It did not yet admit restoration when cleanup had
removed the complete frozen run root.

`2.71.2` is the Evidence/status/Energy-resume repair discovered by the
archived three-model `153940` Smoke. It accepts sealed YOLOv7 multiscale-head
Completed-V2 references through their frozen decoder while retaining the
Direct-BN6 gate for integrated normalization, preserves the explicit DeepX
Full input-contract mode, treats valid positive, negative and inconclusive
raw-head Screening decisions as technical evidence without granting claims,
and represents an explicitly unnecessary host postprocess as N/A. Energy
reporting takes completeness from the authoritative aggregate instead of a
nested repeat, and exact-row resume replaces selected incomplete rows only
after every selected fresh series verifies. It never splices new repeats into
an old incomplete series.

`2.71.1` is the live Evidence-binding repair discovered by the archived
three-model `082223` Standard/Development run. It seals Direct-BN6
completed-task workloads, binds the physical DeepX raw-head endpoint, permits
portable deterministic semantic results only for non-claimable Screening
policies, and keeps incomplete semantics separate from technical failures.
Final remains fail-closed without exact completed-result identity.

`2.71` is the Standard evidence and Native-energy repair discovered by the
archived three-model `215810` Development run. It projects verified
Completed-V2 aliases before consumer gates, separates complete negative
semantic decisions from technical failures, introduces an explicit
non-claimable Screening-energy tier, preserves `strict=false`, and records an
empty concrete plan before starting any physical energy measurement. It builds
on `2.70m`, the completed-energy consumer, remote hotloop and replay repair
discovered by the three-model 2.70l Native-energy Smoke. It preserves physical
endpoint identities while admitting a Completed-V2 comparison endpoint only
after strict completion attestation, canonicalizes the sealed Hailo-10H/DeepX
aliases, fixes isolated DeepX and packed Hailo-10 Full hotloops, and accounts
for every energy attempt in Scientific reports. The YOLOv7/Hailo-8 numerical
gate remains unchanged and requires a repeated hardware replay. It builds on
`2.70l`, the Native evidence, energy and reporting repair discovered by the
three-model 2.70k Smoke audit. It provides an explicitly versioned,
geometry-bound comparison endpoint for raw-host-tail and integrated-BN6
detection paths, restores Quality and Full-hotloop handoff to the energy
planner, carries Mean-IoU/E2E/host-tail evidence through Scientific reports,
and makes structural claim failure terminally fail-closed. It builds on 2.70k,
the Native contract-completion repair discovered by the overnight three-model
2.70j Standard audit. That release derives the ONNX Full self-reference from
the sealed runtime tensor, applies a deterministic reject-and-backfill policy
for Native-incompatible multi-input Split candidates, normalizes strictly
verified host-postprocessing evidence across old and new field names, and
separates the backend-independent completed-task comparison endpoint from each
physical backend endpoint. It builds on 2.70j, which verifies
portable Split bindings with the sealed pipeline identity, versions the
class-aware YOLOv7 similarity rule at match ratio 0.80 plus mean matched IoU
0.90, separates structural, numerical and task-quality evidence, and carries
E2E scope, completed endpoint and host-postprocessing evidence losslessly into
the final reports. Archived schema-2 policies remain at their original 0.90
ratio threshold. It builds on the 2.70i Native Full evidence repair, the 2.70h
Remote-suite bootstrap repair, the 2.70g Native-validation and
Generic bridge repair, the 2.70f BenchmarkSet logger/backfill repair, and the
2.70 generator/reuse baseline
in the public v2 release family established by `2.61e`. It does not claim that final hardware measurements are
already complete. The package form is deliberately PEP 440 compliant, so
dependency resolution and isolated installation remain well-defined.
Evaluation provenance records package version, workflow contract, build
identifier, feature contract and hashes for the package and its claim-critical
modules. The GUI uses the release label.
The supported campaign deliverable is the source archive; a wheel is neither
required nor shipped. Release 2.71 includes the complete runtime-contract
binding introduced by 2.62.1. Following the explicit post-2.66 method decision,
the command-marker window is the scientific primary for new measurements and
the Chapter-4 result remains available as a same-trace legacy shadow.
The workflow-contract identifier changes independently of the package label
when persisted evidence semantics change. Historical smoke tests therefore
retain their original identifiers while also accepting the current contract.
The current hardware-independent release smoke is
`onnx-splitpoint-smoke-v27922`; the earlier versioned smoke aliases remain
available as backward-compatibility checks.

## Changes before 3.0

- The public letter label may advance for a tested pre-3.0 campaign hotfix;
  each such label also receives a unique PEP-440 package patch version.
- Backward-compatible fixes after 2.75.0 use later package patch releases.
- Further material workflow additions use a later minor release.
- Letter labels remain provenance for this pre-3.0 campaign line and are not a
  substitute for the package version or workflow-contract identifier.

## 3.0.0 release gate

Version `3.0.0` is cut only when all of the following are true:

1. development and confirmatory-holdout model roles, model hashes, candidate
   universes, score-independent audit plans, fitted ranking models and
   prediction freezes are archived and verify;
2. the declared Native Split and matching Native Full paths pass runtime,
   interface and dataset-level task-quality gates;
3. final energy rows use measured completed work units, an exact command window,
   a calibrated physical scope, at least three repetitions and the declared
   confidence interval;
4. the frozen Standard and Final profiles pass hardware-independent acceptance
   tests and the selected hardware acceptance campaign;
5. the source archive is reproducible, hash-indexed and directly usable in an
   isolated environment;
6. no required final-campaign readiness check is deferred or failed.

Changing a predictor, adapter, filter, threshold, contract or workaround after
opening confirmatory-holdout outcomes creates a protocol amendment or new
protocol version and requires an untouched confirmation model where the change
could have adapted the method to those outcomes. The complete compatibility and
adapter-freeze procedure is defined in `docs/HOLDOUT_ADAPTER_PROTOCOL.md`.

## v2.79.4: release-line acceptance consistency

Build ID: `v2.79.4-native-productized-three-stage-release-consistency`

This maintenance release keeps the v2.79.3 productized Native Three-Stage and B500 evidence logic unchanged while aligning the generic v2.79 smoke alias, updater entrypoint, local acceptance, and release-line provenance tests with the installed maintenance version.

## v2.79.5: release-launcher and evidence closure

Build ID: `v2.79.5-release-launcher-evidence-closure`

This maintenance release carries the v2.79.4 scientific contracts forward
unchanged and closes current-release acceptance aliases, the detached
seven-model Generic launcher identity, and evidence/source-snapshot provenance.

## v2.79.6: remaining-changes and YOLO11 admission closure

Build ID: `v2.79.6-remaining-changes-yolo11-admission-closure`

This maintenance release closes the remaining immutable invocation,
logical-primary matrix, runtime-precision, physical-scope, exact legacy
reconciliation, unlimited cold-build, final artifact-index and YOLO11 Full
terminal-admission gates. Its focused acceptance emits an atomic JSON phase
report while preserving the established scientific ranking and measurement
contracts.

## v2.79.7: YOLO11 six-path runtime-identity closure

Build ID: `v2.79.7-yolo11-six-path-runtime-identity-closure`

This release requires backend-bound Native Full and composed-path evidence for
the YOLO11 admission gate. It rejects Generic reciprocal-latency rows as Native
throughput and closes Hailo-8 plan materialization, Hailo-10H aliases, DeepX
receipt mirroring and duplicate-free terminal artifact-index sealing.

## v2.79.8: YOLO11 gate-profile schema closure

Build ID: `v2.79.8-yolo11-gate-profile-schema-closure`

The shipped YOLO11 R8B profile is now accepted by the strict schema and is
exercised by both real pre-hardware loader paths during focused acceptance.
