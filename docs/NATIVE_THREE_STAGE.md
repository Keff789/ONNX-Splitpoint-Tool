# Native three-stage execution (current release 2.79.20)

Release 2.79 preserves exactly two scientific runner roles: **Generic** and
**Native**.  The Native role may internally execute three concurrent service
stages:

1. **P1** – input/preprocessing and the producer accelerator segment;
2. **P2** – boundary handoff and TensorRT suffix;
3. **Post** – contract-bound task completion.

## v2.79.20 current release binding

The current build is
`v2.79.20-artifact-reuse-closure`. Three-Stage scheduling, scientific
endpoints, pipeline-FPS semantics, Hailo-8/Hailo-10 runtime format, ranking and
Energy policy remain unchanged. The release changes only the persistent
artifact preparation around the runners: cached Hailo/DeepX artifacts and
receipt-valid TensorRT engines are preferred before a compiler, and TensorRT
Full is model-bound while Part1/Part2 remain split-bound.

The read-only cache preflight reports availability before hardware work and
does not replace runtime or Quality validation. A cached raw-head artifact
still has to satisfy the existing semantic output contract.

## v2.79.19 historical release binding

The v2.79.19 build was
`v2.79.19-calibration-warning-evalrun-closure`. It closed complete Hailo remote
module staging, exact duplicated Quality contract fields, final-accepted-case
Required Scope, and canonical report/artifact-index reconciliation.

Required Scope is sealed after BenchmarkSet reject/backfill from the final
accepted cases and before runtime dispatch. A rejected candidate is therefore
not a missing runtime measurement. Model-local empty/inconsistent accepted
scope blocks only that model; global infrastructure and configuration errors
remain global blockers.

## v2.79.18 historical release binding

The v2.79.18 build was
`v2.79.18-simplified-full-system-input-calibration`. It changed the separate
Full-System calibration flow, not Three-Stage execution.

## v2.79.17 historical release binding

The v2.79.17 build was
`v2.79.17-native-energy-row-isolation-closure`. It preserves Three-Stage
execution and endpoint semantics while using the simple Full-System M.2-off/on
result for TensorRT Full accelerator-idle normalization. Unknown, split, Hailo
and DeepX roles never receive that correction.

The maintained source with that same version and build ID changes only the
Hailo-10 Native-Split InferModel boundary: HEF-native `UINT8` input and output
are converted with the exact `QuantInfo` of their respective VStreams. The
canonical floating-point boundary toward TensorRT remains explicit. Hailo-8,
Hailo-10 non-Split paths and the outer `P1 → FIFO → P2` pipeline are unchanged.
No new InferModel parallelism is introduced; any pre-existing configured
inflight depth remains part of the unchanged execution contract. Real Hailo-10
hardware validation of this maintained-source change remains pending.

## v2.79.14 historical release binding

The v2.79.14 build was
`v2.79.14-simple-full-system-m2-idle-calibration`.

## v2.79.13 historical release binding

The v2.79.13 build was
`v2.79.13-platform-power-calibration-operational-repair`.

## v2.79.12 historical release binding

The v2.79.12 build was
`v2.79.12-platform-power-calibration-provenance-closure`. It introduced the
full-system method and off/on evidence provenance used through v2.79.13.

## v2.79.11 historical release binding

The v2.79.11 build was
`v2.79.11-platform-power-energy-evidence-closure`. It introduced the three-card
Platform Power UI and the first exact TensorRT Full idle-power binding.

## v2.79.10 historical release binding

The v2.79.10 build was
`v2.79.10-urecs-platform-power-release-closure`. It closed release identity,
acceptance, update safety and long-run launcher admission around the first
Tool Config platform-power feature.

## v2.79.9 historical release binding

The v2.79.9 build was
`v2.79.9-urecs-platform-power-and-m2-idle-calibration`. It introduced Tool
Config platform status/control and the M.2 idle-power calibration surface.

## v2.79.8 historical release binding

The v2.79.8 build was `v2.79.8-yolo11-gate-profile-schema-closure`. It repairs
only the strict pre-hardware profile/schema admission path and exercises both
real profile loaders during focused acceptance. It does not change Three-Stage
or Native Full execution, endpoint, Quality-oracle, ranking or energy
semantics.

## v2.79.7 historical release binding

The v2.79.7 build was
`v2.79.7-yolo11-six-path-runtime-identity-closure`. It keeps concurrent
P1/P2/Post claims separate from Native Full claims and binds both to their exact
runner and backend identities. A synchronous Generic Full latency row is
diagnostic only and cannot satisfy Native Full throughput admission. YOLO11
Hailo Full requires exact completion/makespan evidence from the Native producer
service. The YOLOv7 `claim_gate_32`, endpoint semantics, adapter selection,
Quality-oracle placement, ranking and energy policy remain unchanged.

## v2.79.6 historical release binding

The v2.79.6 build was
`v2.79.6-remaining-changes-yolo11-admission-closure`. It closed the remaining
invocation, scope, reconciliation, cold-build, artifact-index and first YOLO11
terminal-admission gaps before the runtime-identity findings of v2.79.7.

## v2.79.5 historical release binding

The v2.79.5 build was `v2.79.5-release-launcher-evidence-closure`. That
maintenance closure aligned release-line launchers, acceptance identity and
evidence provenance without broadening Native hardware admission.

The physical observation boundaries are:

- `p2_output`: primary hardware-performance and Generic↔Native bridge endpoint;
- `completed_detection`: application endpoint after the optional Post stage;
- `classification_logits`: terminal classification endpoint, with a no-op Post
  adapter.

## Adapter registry

- `classification_logits_noop`
- `yolov7_anchor_multiscale_sparse`
- `yolo26_decoded_nms_materialize`
- `yolo11_regcls_dfl16`

Unknown or ambiguous output contracts fail closed.  The adapter is selected
from an already-attested physical P2-output contract, never merely from a
similar tensor shape.

## Timing and evidence

The measured hot loop contains the real P1, P2, queue/handoff and task
postprocessing work.  It deliberately excludes per-frame SHA-256, canonical
JSON evidence construction, generic contract discovery and the slower reference
decoder.  A fully bound Quality oracle runs in pre/postflight and in B500/Final
Quality outside timing.

Every direct v2.79 observation reports P1/P2/Post mean, P50, P95, min and max,
queue-wait summaries, `p2_output_fps`, `completed_detection_fps`, their ratio,
contract family, adapter id, postprocess location and oracle status.  Historical
rows may be projected only with `directly_measured=false` and an explicit
`projection_source`; a second endpoint is never fabricated.

## Preserved scientific decisions

This release does not change `cut_bytes_only`, Stratified Windows, candidate
generation, B500/Final Quality, compiler recipes, the Development/Transfer split
or the existing Generic↔Native bridge.  Multi-tensor boundaries remain
Generic-only evidence.

## v2.79.1 concurrent normal-runner admission

Version 2.79.1 wires the hardware-proven YOLOv7 `b066` Hailo-8 to TensorRT
Three-Stage runtime into the normal benchmark-set runner. Admission is narrow
and fail-closed: detection, `dual`, `fast_oracle_outside_timing`, Hailo-8,
`b066`, `uint8_dequant_fp16`, a YOLOv7 identity, and a non-empty quality
binding are required. A successful result reports `p2_output` and
`completed_detection` from one `concurrent_three_stage_single_invocation` and
sets `three_stage_concurrency_directly_measured=true`.



## v2.79.2 historical release binding

The installable build `v2.79.2-native-concurrent-three-stage-b500-evidence-reconciliation` retains the narrow, fail-closed YOLOv7 `b066`
concurrent admission and binds both endpoints to one Quality-selected HEF/engine
pair. `p2_output` and `completed_detection` are measured in one invocation,
while the frozen Quality oracle remains postflight. B500 reconciliation changes
do not alter Native stage timing, adapters, ranking or energy semantics.
