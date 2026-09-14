# Documentation index — current release 2.79.20

Current build/workflow:
`v2.79.20-artifact-reuse-closure`.

Release 2.79.20 closes artifact reuse around the existing backend caches.
The Hailo raw-head fallback no longer forces a build, TensorRT checks valid
engines before `trtexec`, Full TensorRT engines are split-independent, partial
stable namespaces still search compatible legacy artifacts, and retention
decisions are explicit. A read-only per-model cache-preflight matrix reports
Hailo-8, Hailo-10, DeepX, TensorRT Full and TensorRT Part2 as HIT, MISS,
UNKNOWN or NOT_APPLICABLE before expensive runtime work. DeepX retains its
existing cache system and gains explicit cache diagnostics. No additional
hash, manifest, signature or sealing layer is introduced.

Historical release 2.79.19 keeps the Full-System origin fit and separates technical
validity from numerical plausibility. Point-factor spread, expected-factor and
nominal-current deviations are visible warnings that permit saving; unusable
or acquisition-invalid measurements remain blocked. See
[PLATFORM_POWER_CONTROL.md](PLATFORM_POWER_CONTROL.md).

The EvaluationRun closure adds the two missing Hailo backend modules to remote
runtime staging, propagates three existing Quality contract values to the
required request/candidate fields, seals runtime scope from the final accepted
BenchmarkSet cases, and reconciles only intentionally rebuilt/retired report
records in the artifact index. Unrelated missing artifacts remain terminal
errors. No additional hash, manifest, signature or sealing layer is
introduced.

The same debug-pack audit also closes Classification Native split policy and
orchestration for MobileNet/RegNet, the Hailo-8 ResNet distinction between
quantized HEF storage and a FLOAT32 VStream with strict native width for raw
integer paths, and post-dispatch DeepX result-status retention. Missing
and conflicting `input_contract.model_id` values both remain invalid and are
now exposed by concrete binding reasons; newly produced per-artifact contracts
carry that model identity directly. The DeepX YOLO11 end-to-end hardware
hypothesis is still `NOT_RUN`, not a claimed success.

The prior disk-capacity problem was fixed operationally, not in source. Real
u.RECS/electronic-load validation and a real multi-host EvaluationRun are
`NOT_RUN` in the build environment.

Historical release 2.79.18 simplified the calibration state sequence and added
one bounded acquisition-integrity retry. An explicitly confirmed already-off
Jetson stays off; an initially ready Jetson performs one off and one restore
transition; M.2 is untouched.

Historical release 2.79.17 gave the maintained Native Energy row-isolation
closure a unique identity and completed its model-local preflight handling.
Its Native, Hailo-10, Quality and Energy behavior is unchanged in 2.79.18.

Historical release 2.79.16 introduced the calibration and carried the initial
maintained closure under an unchanged identity; v2.79.17 resolves that release
provenance ambiguity and extends row isolation to the remaining model-addressable
case-preflight conditions.

Historical release 2.79.15 made `FS`/`command` the explicit energy contract,
migrated only a missing or exact legacy `MB` registry default, and applied the
existing ImageNet mean/std ONNX adapter to DeepX Classification Part1.

Historical release 2.79.14 replaces the failed manifest-preparation path with a direct
Full-System M.2-off/on calibration. It asks for no name, creates no calibration
binding or SHA chain, and saves one readable result JSON plus the ordinary raw
measurement folders.

Historical release 2.79.12 verifies the full-system energy method before
platform mutation, passes it through both M.2 idle measurements and hash-binds
the resulting off/on evidence. It also migrates only the former shipped `m2`
token to the real `m.2` controller command and exposes exact gate reasons.

Historical release 2.79.11 provided the three-card Platform Power UI,
immutable accelerator-idle calibration binding, exact TensorRT Full
normalization and the interrupted-run evidence closure.

Historical release 2.79.10 closed source-update/workflow interlocking and the
release launcher identity. Historical release 2.79.9 introduced the Tool Config platform-status banner,
u.RECS UDP rail toggles and an initial M.2 idle-power calibration. Release
2.79.10 supplied the explicit-target, global-interlock, repeated-observation,
capture-state and atomic-registry closure required for hardware use. Status
checks remain one-shot/manual; no continuous ping loop is introduced. See
[PLATFORM_POWER_CONTROL.md](PLATFORM_POWER_CONTROL.md).

Historical Release 2.79.8 closes the strict profile/schema admission mismatch in the
YOLO11 R8B gate and validates the shipped profile through both real
pre-hardware loader paths. Runtime identity, recovery, ranking, Quality,
energy and hardware measurement contracts remain those established by
v2.79.7.

Historical release 2.79.7 made the YOLO11 Full/b067 admission backend-bound across
Hailo-8, Hailo-10H and DeepX-M1. Native Full claims require the canonical
Native producer result with exact completion and makespan evidence; Generic
reciprocal-latency observations remain diagnostic. Hailo plan/alias, DeepX
receipt and terminal artifact-index consistency are closed as well. The
current seven-model launcher still requires the real YOLOv7 `b066` strict
`claim_gate_32` receipt with 32/32 parity.
Candidate generation, `cut_bytes_only`, Quality thresholds and energy policy
remain unchanged.

Historical release 2.79.2 closed the concurrent Native normal-runner packaging gap and
the B500 evidence-accounting defects. See
[B500_EVIDENCE_RECONCILIATION_v2792.md](B500_EVIDENCE_RECONCILIATION_v2792.md)
for immutable pre-dispatch scope, logical mirror provenance, exact request-SHA
Quality joins, Quality-N/A accounting, Hailo attempt receipts, strict detection
record parsing, launcher status and ETA rules. Build identity: `v2.79.2-native-concurrent-three-stage-b500-evidence-reconciliation`.

Historical release 2.79.1 added the contract-bound Native three-stage model. It reports
P1, P2 and optional postprocessing separately, uses `p2_output` as the primary
hardware endpoint and `completed_detection` as the application endpoint, and
runs the frozen Quality oracle outside performance timing. See
[NATIVE_THREE_STAGE.md](NATIVE_THREE_STAGE.md). The release preserves all
2.78.4 Gate-A, seven-model launch, Quality, Ranking and resume behavior.

This clean package contains current operational documentation and the retained
v2.75.47/v2.75.46/v2.75.45/v2.75.44/v2.75.43/v2.75.42 guide/report pairs needed to audit the patch
transitions. The
consolidated release lineage remains in [VERSIONING.md](VERSIONING.md).

Release 2.78.4 adds a fail-closed audit for the planned Gate-A stop after
backend-artifact generation. It admits the terminal anchor result only when
the Gate-A receipt and workflow evidence agree with that configured stop;
unrelated nonzero exits remain failures. The existing `v2783` Gate-A profile
and evidence identities stay unchanged so exact accepted artifacts remain
reusable.

The release also packages a frozen seven-model B500/audit-20 profile and a
detached overnight launcher. The launcher revalidates the retained YOLO11
`b067` Gate-A anchor, source model and exact Hailo-8/Hailo-10 cache receipts,
freezes the hardware registry, and starts a fresh workflow without rerunning
Gate A. The anchor is the first deployment sentinel; the independent audit
membership and `cut_bytes_only` ranking remain unchanged. Pair compilation is
resource-gated, while Native and Energy are disabled for this screening run.

The carried-forward 2.78.3 work adds the Native Detection dual-endpoint
contract. The same
Native deployment is measured in isolated `raw_model_outputs` and
`completed_task` phases, with exact shared-artifact and raw-output binding.
The raw endpoint is the hardware-performance surface; the completed endpoint
retains decoder/NMS application throughput and remains the Energy surface.
The bundled offline completion-tail canary profiles the frozen host tail from
exact raw-head dumps without accelerator execution.

The integrated B5-readiness recovery also adds an immutable Hailo build-
evidence ledger and a bounded YOLO11 Hailo-8-first Gate A. The gate reuses
exact evidence and cache entries before cold compilation, never starts B5,
and stops fail-closed as `ANCHOR_FOUND`, `CANARY_BUDGET_EXHAUSTED`, or
`EVIDENCE_CONFLICT`.

Release 2.78.2 accepts the real eight-field historical semantic-reference
placeholder in the read-only verifier and makes future management CPU
references immutable per source contract. The producer and all consumers bind
the exact path, source-contract hash, file hash and size; contract changes
therefore preserve earlier evidence rather than replacing it at one mutable
canonical path. Legacy references remain fail-closed and are migrated only
when their recorded status exactly matches the current contract and bytes.

Release 2.78.1 adds a read-only cross-evidence verifier for existing
evaluation, structural-canary, and self-reference artifacts.  Its receipt is
retrospective and preserves the source run's existing claim scope; it is not a
prospective protocol freeze. `VERIFIED` requires the physical historical
request, candidate, and CPU-reference containers to remain available at their
recorded hashes. Metric arithmetic, the append-only artifact index, frozen raw
tensor signatures, completed detections, and comparison endpoints are
cross-bound as well. A superseded container at a reused canonical reference path is
reported as `INCOMPLETE` (`historical_reference_version_unavailable`), never
silently reconstructed from an equal prediction fingerprint. Release 2.78.0
scopes the successful Full-ONNX
self-reference interpretation
to the tested input and configured similarity policy. It explicitly preserves
dataset-wide AP/accuracy as separate quality evidence. JSON decisions,
thresholds, ranking, adapters, artifacts and hardware results are unchanged.

Release 2.77.15 admits the canonical decoded YOLO11 Full-ONNX pre-NMS tensor
as a reference for an attested frozen Native `ultralytics_regcls` host tail.
The allowance is bound to YOLO11, the frozen execution mode and the sealed
decoder; it does not extend to YOLOv7 or unsupported raw formats. It enables a
real detection comparison but does not assert its numerical outcome.

Release 2.77.14 materializes shared, locally hardlinked validation images as
independent regular payload members through no-follow file descriptors. The
strict remote verifier continues to reject every Tar hardlink and symlink.
Full source paths remain in the result receipt but are compacted out of stdout.

Release 2.77.13 adds path-neutral, read-only semantic smokes for exact prepared
RGB identity, CPU parity of original versus Hailo-fixup ONNX models and frozen
YOLO11/YOLO26 host-tail parity against independent completed detections. Its
optional small artifact-reuse hardware canary neither compiles nor starts a
B500 workflow. Reporting now separates exact technical Full-only completion
from negative scientific decisions, exposes comparison availability separately
from Quality PASS/FAIL, and never presents a skipped fast-fail bootstrap point
estimate as a computed 95% lower confidence bound.

Operational gates:

- Generate a semantic-smoke template with
  `python -B -m onnx_splitpoint_tool.backend_semantic_smoke --example`, fill in
  existing artifact paths, then execute it with `--spec FILE --require-pass`.
  A required gate treats `SKIP` exactly like `FAIL`.
- Run `scripts/run_v27713_hailo8_artifact_canary.py` only against an existing,
  receipt-bound Full BenchmarkSet. With `--hardware-matrix`, `--setup-id` and
  `--require-hardware`, it performs bounded remote structural inference without
  compiling. Its PASS covers identity, runtime and structural completion only;
  numerical parity and task accuracy remain outside this canary.

Release 2.77.12 normalizes
YOLO11 pre-NMS SHA-256 identities, gives an exact
targeted rebuild precedence over a failed archived target checkpoint, and
audits an already complete exact repair without another workflow or hardware
dispatch. Release 2.77.11 additionally admits exact, correctly indexed
zero-byte stage artifacts and read-only attests the known successful
classification stderr producer omission. Future dispatches always materialize
the empty stderr file.
Its wrapper supports five-to-seven preserved results and fails closed on a
failed/cancelled workflow or any must-reuse-stage rebuild. Release 2.77.10
introduced cohort-wide read-only supersession attestation and precise abandoned
session recovery. Release 2.77.9
clears the archived fresh-run creation guard only for the
sealed targeted missing-Full-quality resume, retains the Standard-mode check,
and isolates its disposable remote transport workspace from archived requests.
Every exact-5/8, no-build, no-Ranking, no-Native and no-Energy invariant is
retained. Release 2.77.8 closes the shared Full-detection adapter: Hailo quality uses
the timed raw-head order, YOLO pre-NMS matrices retain their host decoder/NMS,
and the exact missing-quality resume preserves completed results while running
no compiler, Ranking, Native or Energy work. Release 2.77.7 preserves explicit
Hailo-10 and empty DeepX-only physical
targets through run-mode/workflow projection and admits YOLO11 Native-Full only
for the parity-proven DFL16/C80 80/40/20 raw endpoint. It changes no ranking,
candidate, threshold, Quality, energy or reporting policy and adds no new
hardware gate. Release 2.77.6 replays the automatic DFC parser path before any explicit
endpoint retry and fail-closed attests every archived graph output and
tool-owned Identity alias before projecting DFC-visible producers. The
projection is admitted only when producer expansion preserves the exact
graph-output slots and order, and the four archived ONNX files are bound by
their observed SHA-256 identities. Release
2.77.5 preserves the lexical managed-venv path when `bin/python` is a
POSIX symlink, reports optional `onnxsim` recovery availability, and treats
reproduced explicit-endpoint `base_conv` rejections as complete negative
compiler evidence rather than a Tool failure. Release 2.77.4 closed the two
live YOLO26 canary control-path leaks without a
new workflow layer: forced cases are the exact generator scope, and an explicit
Full skip also blocks prepared/mandatory Full paths. Managed DFC child launches
now receive the venv `PATH`, allowing vendor tools such as `onnxsim` to resolve.
Ordinary non-forced backfill, ranking and scientific protocols are unchanged.

Release 2.77.3 closes the stale Hailo Full request bookkeeping exposed by the
v2.77.2 canary. Explicitly unrequested Full contracts remain visible as
provenance but are not copied, promoted, counted as missing or queued.

Release 2.77.2 is the narrow live-canary closure on the unchanged 2.77
ranking freeze. `hailo_build.build_full: false` is now authoritative across
RunPlan, generation runtime and final orchestration, including heterogeneous
matrix variants. The packaged ResNet50 b052 acceptance permits exactly two
uncached Part-1 HEFs and records self-contained hash-manifested evidence. It
does not change ranking, model selection, Quality or Energy policy.

Release 2.77.1 is the report-only replay hotfix on the 2.77.0
`cut_bytes_only` freeze. Lower cut bytes still rank first and exact ties still
use Boundary ID then Case ID. Candidate provenance and the stratified
generation policy remain unchanged; Transfer models may validate but never
retune this policy. The hotfix closes the archived-normalized endpoint path: a
legacy false default without an explicitness marker can be reconciled only by
an exact, valid central request projection with a passed, hash-matching
endpoint attestation, while explicit conflicts remain fail-closed. It also
rebuilds the canonical Native combined summary through the exact validation
join after replay and rejects any performance/repetition drift. A
separate-output, read-only
reprojection preserves
the original EvaluationRun, while a full endpoint ledger distinguishes planned,
materialized, measured and explicit terminal states. Technical observations,
Quality decisions and claim eligibility remain separate; nonzero runner return
codes cannot be hidden by an already-written success report. Successful
backend artifacts remain usable when a sibling backend rejects the same case.
The preceding v2.76 report passed the frozen formula-decision gate and selected
the single global `cut_bytes_only` revision implemented here.

The v2.76 line also incorporates the never separately deployed 2.75.50
intermediate block. It repairs the Scientific and Cross-Runner projection
exposed by the completed development audit. Measured observations remain
visible without weakening strict claim gates; setup identity, candidate
coverage and technical, Quality-screened and claim-eligible Generic/Native
pairs are reported separately. The scheduler supports a bounded,
resource-checked Hailo-8/Hailo-10H managed-venv pair build with deterministic
serial fallback for local, WSL, unresolved or undersized configurations.

The 2.76.2 maintenance closes the final Cross-Runner projection gaps. The
frozen Native matrix defines the 24-pair join domain, the broader Generic
universe stays outside that domain without being reported as missing, and
technical, Quality and claim cohorts plus their concordance/Hit@1/Regret@1
aggregates are explicit. Published replay paths refer to their final output
root rather than the temporary atomic staging directory.
Normalized full-run metric aliases and archived compact-row aliases now feed
the same Cross-Runner metric extraction, and the human-readable report carries
the same frozen-denominator diagnostics as its JSON source.

Release 2.75.49 closes the one remaining DeepX-Full semantic binding gap in
the YOLOv7 anchor. The centrally registered `yolov7_paper` model/decoder may
supply a missing redundant digest in the DeepX backend/endpoint contract;
every present value must still be valid and mutually consistent. The packaged
v2.75.49 profile is an otherwise identical fresh b044/B500 repetition with
Native Full on and Energy/ranking off; v2.75.48 results are never reused.

Release 2.75.48 closed the three earlier integration gaps exposed by the YOLOv7
technical anchor: canonical COCO annotation-digest comparison, registry-bound
Native completion without redundant source-hash propagation, and the same
optional transport identity for DeepX Full prepared inputs. Explicit identity
conflicts remain fail-closed. The CPU A/B keeps mandatory Official COCO; the
one technical B500 anchor uses paired Quality and remains Energy/ranking-off.

Release 2.75.46 decouples Native-Full raw-head ONNX parsing from the Hailo
vendor interpreter, projects the exact Standard setup-local Quality contract,
separates its Generic diagnostics and repairs the artifact-derived DeepX
B500/B1000 endpoint-hash comparison without weakening semantic invariants.

Release 2.75.45 restores fresh GUI starts for large score-independent audits,
shows their actual execution scope before confirmation, exposes the DeepX
classification preprocessing axis and separates the active TensorRT working
set from the retained non-current cache budget. Stable Resume/reuse and all
unsafe-cache protections remain fail-closed.

Release 2.75.44 verifies the complete ImageNet Validation source manifest and
then projects the frozen 500-image runtime cohort before comparing it with the
archived runtime manifest, image-ID and ground-truth authorities. It therefore
repairs the invalid 50,000-source-versus-500-runtime identity comparison while
leaving every file, selection and scientific gate fail-closed.

Release 2.75.43 accepts the exact historical portable dataset item-identity
serialization already admitted by the common manifest verifier. It still emits
the normalized portable identity and leaves every file, payload, component and
frozen-authority gate fail-closed.

Release 2.75.42 repairs installed-tree source attestation and binds the real
ImageNet calibration-export identity. Its deterministic updater uses
`rsync --checksum`, preserves the existing virtual environment and refreshes
project distribution metadata plus console entry points without a network or
build backend. Setup-local TensorRT Quality companions carry their
physical effective-plan identity through management admission, and Standard or
Final fails if a required companion result is absent. The v2.75.41 ResNet50
calibration-size experiment remains unchanged; its B1000 canary comes before a
new three-model Final run. The failed nine-invocation night run is not valid
Final-Quality evidence.

Release 2.75.41 provides the controlled ResNet50 DeepX calibration-size
follow-up.  The corrected ImageNet Mean/Std path remains fixed while the
train-derived calibration cohort increases from 500 to 1,000 images.  The
read-only preflight proves dataset readiness and the paired verifier requires
the real 500-item cohort to be a subset of the 1,000-item cohort before it
evaluates the unchanged 500-image validation set.

Release 2.75.40 provides the paired ResNet50 DeepX preprocessing profiles,
isolated exact-v2 cache contracts, a content-addressed ImageNet Mean/Std ONNX
adapter, the hardware-free Float-ORT control and the fail-closed A/B verifier.
It also preserves the signed physical TensorRT producer alias in Full-only
Quality export.

Release 2.75.39 preserves deterministic pre-mutation SSH failures as the
primary error and skips collection when no remote command started. Confirmed
leases retain fail-closed cleanup quarantine. Use
`profiles/resnet50_v27539_deepx_full_quality_canary.yaml` for the diagnostic;
the real DXNN identity, forbidden ORT CPU fallback, Quality-only Performance
N/A contract and backend-scoped Hailo-Full gate remain unchanged.

Release 2.75.36 retained the normal `final` run mode as **Final Quality
(Standard+)**: the Standard execution path with 5,000 Classification
validation items, 5,000 Detection validation items and 5,000 task-quality
bootstrap repetitions. Preparation, sealing, a separate repeated full-dataset
hash preflight and canaries are not prerequisites. Strict campaign and canary
tools from 2.75.28 and earlier remain available as optional historical or
advanced workflows.

That corrective 2.75.36 release completed the singleton-squeezed Hailo channel-vector
contract in the final structural validator, separates persistent TensorRT
engine identity from volatile suite provenance and reports disabled Energy as
not applicable. Verified older managed TensorRT receipts may be imported into
the new stable namespace without modifying their source. The frozen audit
union and profile-editor repairs from 2.75.35 remain unchanged.

## Start and configuration

- [Quick start](QUICKSTART.md)
- [Evaluation profiles](EVALUATION_PROFILES.md)
- [Evaluation workflow](EVALUATION_WORKFLOW.md)
- [Validation datasets](VALIDATION_DATASETS.md)
- [Model preparation](MODEL_PREPARATION.md)

## Runtime and evidence contracts

- [Artifacts and postprocessing](ARTIFACTS_AND_POSTPROCESSING_CONTRACT.md)
- [Native producer EvaluationRun usage](NATIVE_PRODUCER_EVALRUN_USAGE.md)
- [Native boundary validator](NATIVE_BOUNDARY_INTERFACE_VALIDATOR.md)
- [Native HailoRT/TensorRT fast path](NATIVE_HAILORT_TRT_FIFO_FASTPATH.md)
- [Energy window method validation](ENERGY_WINDOW_METHOD_VALIDATION.md)
- [Hold-out adapter protocol](HOLDOUT_ADAPTER_PROTOCOL.md)

## Release and packaging

- [2.79.20 test guide](../TESTANLEITUNG_2.79.20.md)
- [2.79.20 build and test report](../VERSION_2.79.20_BUILD_AND_TEST_REPORT.md)
- [Historical 2.79.19 test guide](../TESTANLEITUNG_2.79.19.md)
- [Historical 2.79.19 build and test report](../VERSION_2.79.19_BUILD_AND_TEST_REPORT.md)
- [Versioning and the 3.0 gate](VERSIONING.md)
- [Clean release policy](CLEAN_RELEASE.md)
- Compact 2.78.4 test commands are provided directly in the release handoff.
- [Historical 2.75.47 test guide](../TESTANLEITUNG_2.75.47.md)
- [Historical 2.75.47 build report](../VERSION_2.75.47_BUILD_AND_TEST_REPORT.md)
- [Historical 2.75.46 test guide](../TESTANLEITUNG_2.75.46.md)
- [Historical 2.75.46 build report](../VERSION_2.75.46_BUILD_AND_TEST_REPORT.md)
- [Historical 2.75.45 test guide](../TESTANLEITUNG_2.75.45.md)
- [Historical 2.75.45 build report](../VERSION_2.75.45_BUILD_AND_TEST_REPORT.md)
- [Historical 2.75.44 test guide](../TESTANLEITUNG_2.75.44.md)
- [Historical 2.75.44 build report](../VERSION_2.75.44_BUILD_AND_TEST_REPORT.md)
- [Historical 2.75.43 test guide](../TESTANLEITUNG_2.75.43.md)
- [Historical 2.75.43 build report](../VERSION_2.75.43_BUILD_AND_TEST_REPORT.md)
- [Historical 2.75.42 test guide](../TESTANLEITUNG_2.75.42.md)
- [Historical 2.75.42 build report](../VERSION_2.75.42_BUILD_AND_TEST_REPORT.md)


## v2.79.4: release-line acceptance consistency

Build ID: `v2.79.4-native-productized-three-stage-release-consistency`

This maintenance release keeps the v2.79.3 productized Native Three-Stage and B500 evidence logic unchanged while aligning the generic v2.79 smoke alias, updater entrypoint, local acceptance, and release-line provenance tests with the installed maintenance version.

## v2.79.5: release-launcher and evidence closure

Build ID: `v2.79.5-release-launcher-evidence-closure`

This maintenance release keeps the scientific and runtime contracts unchanged
while aligning all current v2.79 acceptance aliases, the detached seven-model
Generic launcher, and release-evidence provenance with the installed version.

## v2.79.6: remaining-changes and YOLO11 admission closure

Build ID: `v2.79.6-remaining-changes-yolo11-admission-closure`

This maintenance release closes the remaining claim, identity, scope,
reconciliation, cold-build, artifact-index and YOLO11 terminal-admission gates
and records every focused acceptance phase in JSON.

## v2.79.7: YOLO11 six-path runtime-identity closure

Build ID: `v2.79.7-yolo11-six-path-runtime-identity-closure`

The gate now proves exact runner/provider/backend identity for all requested
YOLO11 Full and b067 paths and retains recovery provenance without accepting a
diagnostic Generic latency row as Native throughput.

## v2.79.8: YOLO11 gate-profile schema closure

Build ID: `v2.79.8-yolo11-gate-profile-schema-closure`

The current gate profile passes strict schema validation and exact runtime
snapshot materialization before any hardware dispatch.
