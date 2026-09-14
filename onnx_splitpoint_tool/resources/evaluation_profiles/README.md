# Evaluation profiles

Versioned benchmark/evaluation presets live in this directory. They are input
profiles; the user-facing Smoke, Standard and Final Quality effort policy is
resolved separately from the central run-mode registry.

## Version 2.75.41 DeepX calibration-size Canary

The current ResNet50 follow-up keeps the proven ImageNet mean/std adapter,
EMA, DeepX optimization level 0, the ordered 500-image validation cohort and
the setup-local TensorRT Full control from the v2.75.40 B arm. It changes only
the train-derived calibration cohort from 500 to 1,000 images and writes into
an isolated v2.75.41 DeepX cache namespace. The launch profile lives at the
project-root path
`profiles/resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml`.
Before selecting it, pin the proven B500 manifest and run the read-only
v2.75.41 dataset preflight. The historical A/B profiles remain available and
must not be rerun for this experiment.

## Version 2.75.40 paired DeepX preprocessing experiment

The start-ready root profiles
`profiles/resnet50_v27540_deepx_preprocess_a_current_scale_only.yaml` and
`profiles/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std.yaml` freeze the
same ResNet50 Standard 500/500/500 cohort and setup-local TensorRT control.
Only the explicit preprocessing mode and isolated exact-v2 DeepX cache root
differ. Use the A/B verifier before admitting the corrected arm to Standard+.

## Version 2.75.39 DeepX Full Quality Canary

`resnet50_v27539_deepx_full_quality_canary.yaml` selects only DeepX Full and
its setup-local TensorRT Full Quality companion. It emits no Split, Native,
Energy, ranking, or performance claim. DeepX Full executes the real DXNN,
exports an exact `deepx_dxnn_sha256:<sha256>` identity and forbids ORT CPU
fallback. Missing Performance rows are not applicable to these Quality-only
endpoints; both exact Quality records remain mandatory. The receipt-attested
Hailo-Full preupload gate is explicitly not applicable because no Hailo
backend is requested; Hailo profiles continue to require verified receipts.
The 2.75.39 transport repair skips collection after a proven pre-mutation
connectivity failure, preserves that primary error, and retains fail-closed
cleanup after any confirmed remote lease. The historical 2.75.38 root profile
remains available for provenance.

## Version 2.75.36 evidence and TensorRT-cache repair

Singleton-only Hailo runtime shapes now pass the final structural validator
only under the sealed `as_input`/`identity` contract. TensorRT engine
namespaces bind graph content and builder/target ABI rather than profile or
vendor-build provenance. Disabled Energy is explicitly not applicable. The
packaged `resnet50_v27536_small_acceptance.yaml` profile exercises these paths
with Energy off.

## Version 2.75.35 Hailo channel-vector boundary repair

Singleton-squeezed Hailo VStream channel vectors are mapped to canonical
`[1,C,1,1]` inputs as a byte-identical identity handoff. Candidate-local layout
resolution failures remain no-claim failures but no longer terminate the
remaining remote-suite candidate loop.

## Version 2.77.0 ranking freeze

The global productive selector is `cut_bytes_only`: smaller cut boundaries
rank first, with deterministic Boundary-ID and Case-ID tie breaks. Candidate
`source` remains `onnx_real_boundary_hardware_aware` because it records real
ONNX-boundary provenance, not the ranking formula. Stratified windows, audit
universe, seed, quotas, minimum gap and capability filtering are unchanged.
Alternative methods remain report-only comparators, and Transfer performance
cannot reopen this Development decision.

## Version 2.75.33 frozen audit execution

The stable deduplicated union of predeclared audit and deployment candidates
is the exact generator scope. Deployment `requested_cases`, a second
minimum-gap filter and external generator backfill cannot narrow, reorder or
extend that union. Candidate Plan, Prediction and Candidate-Universe
identities are verified before generation. Compact Debug Packs retain bounded
authoritative model validation summaries. These contracts are retained in
2.75.36.

## Version 2.75.32 audit planning and ranking

Development score-independent audits are included explicitly in the Effective
Plan, including their minimum valid count and deduplicated union range with the
deployment shortlist. Native ranking uses a completed-task comparison stratum;
case-local producer identities no longer fragment a four-candidate audit into
singletons. Generic Standard Quality-PASS rows may support Development-only
ranking analysis without becoming final-claim eligible.

## Version 2.75.31 simplified model usage

The GUI asks only whether a model is used for Development or as a Hold-out.
It derives evaluation role, generalization, validation tier and universe mode
from that choice. Audit size, minimum and seed are selection-wide fields, and
universe completeness comes from generated hashed freeze artifacts rather than
a manually asserted checkbox. The saved profile remains explicit for the
workflow even though those derived fields are no longer presented separately.

## Version 2.75.30 Final Quality

The ordinary `final` run mode is **Final Quality (Standard+)**. It uses the
complete Standard execution configuration and changes only the quality budget
to 5,000 Classification validation items, 5,000 Detection validation items
and 5,000 task-quality bootstrap repetitions (`task_quality_final_5000`,
dataset tier `final`). Native/Energy choices and all build, runtime, cache,
reporting and sampled-integrity settings remain Standard.

Selecting this run mode does not select a strict template from this directory.
It does not require campaign preparation, preflight, sealing, a repeated
full-manifest hash pass or a canary. Normal hashes for the selected 5,000-item
quality subsets remain part of the run evidence.

`final_splitpoint_evaluation_v1.yaml`,
`thesis_final_evaluated_matrix_v1.yaml` and
`thesis_final_campaign_v1.yaml` remain available only as explicit advanced or
historical inputs. The two thesis templates keep their strict `TODO`,
`campaign_inputs/...`, preflight and scope contracts when intentionally used.

## Version 2.75.27 strict final scopes (historical/optional)

`thesis_final_evaluated_matrix_v1.yaml` restricts its claim to the measured
ResNet50/YOLO26s/YOLOv7 by Hailo-8/Hailo-10H/DeepX matrix. Ranking and
prospective hold-out requirements are not applicable, while its declared
dataset, identity, quality, Native-Full and serial full-system-energy gates
remain strict.

`thesis_final_campaign_v1.yaml` is the explicit stronger
`ranking_generalization` template and retains the complete confirmatory
hold-out/protocol-freeze/ranking-bundle contract. Do not use it merely to
obtain Final Quality's larger validation budget.

## Version 2.67 confirmatory protocol (historical/optional)

Confirmatory profiles use `evaluation_role: confirmatory_holdout`;
`evaluation_role: holdout` remains a supported 2.62 alias. A strict profile can
set `campaign.require_protocol_freeze: true` and provide a versioned
`campaign.protocol_freeze` block. The freeze binds five prospective inputs:
candidate universe, DAG analysis, predictions, statistical/ranking policy and
energy protocol. Any unamended change to one of those inputs fails its strict
preflight.

Energy method validation is declared once under `energy.window_method_ab`.
`command_marker_window` is the frozen scientific primary method.
`chapter4_legacy_window` is reprocessed in shadow mode on the same raw capture
to preserve a sensitivity bridge to the Chapter-4 calibration method. The
shadow result cannot replace or invalidate the primary and requires no new
PicoScope measurement. The former names `candidate_v263` and
`chapter4_baseline` remain input/read aliases for archived configurations.
