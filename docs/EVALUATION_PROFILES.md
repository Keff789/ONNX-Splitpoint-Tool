# Evaluation profiles

Evaluation Profiles store the choices for one run: run mode, models,
candidate selection, logical hardware profiles and the Native/Energy switches.
Detailed effort settings come from the central run-mode registry and are
archived as an immutable snapshot in every EvaluationRun.

## YOLOv7 DeepX-Full closeout anchor for 2.75.49

`profiles/yolov7_paper_v27549_standard_anchor_b500.yaml` is the only fresh
hardware profile for this closeout. It is scientifically identical to the
v2.75.48 anchor and pins the known `yolov7_paper.onnx`
SHA-256, exactly `b044`, B500 calibration and validation effort, 500 bootstrap
repetitions and three Native performance repetitions. Generic, Native and
Native Full are enabled. Generic Energy, Native Energy, ranking and the
score-independent audit are disabled. The preceding CPU A/B probe requires
Official COCO; this technical anchor uses the paired internal B500 Quality
gate. Official COCO becomes mandatory again for the final Detection run.

The accepted CPU decoder A/B evidence must still report `status=completed`,
`acceptance.status=accepted` and `official_coco.*.status=ok` for the same model
and validation cohort; the narrow DeepX semantic binding repair does not
require that CPU inference to be repeated. Start this profile as a new
workflow, never Resume the v2.75.48 result. Quality prediction caching is
disabled so old predictions cannot satisfy the new run;
receipt/hash-validated Hailo, DeepX and TensorRT build artifacts remain
eligible for normal reuse. B500 stays frozen and no B1000 rerun is part of the
2.75.49 protocol.
The permitted DeepX reuse root remains pinned to
`~/Models/BackendArtifacts/deepx/v2.75.44/thesis_standard_b500_imagenet_mean_std`;
the decoder treatment does not justify a DXNN rebuild, but the existing
receipt/hash gate still decides every reuse.

## DeepX calibration-size Canary for 2.75.41

Use `profiles/resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml`
only after the packaged read-only dataset preflight passes.  The profile keeps
the successful 2.75.40 Arm-B ImageNet Mean/Std path, EMA, DeepX optimization
level 0, 500 validation images and 500 bootstrap repetitions fixed, but uses a
new exact 1,000-image train-derived calibration manifest and an isolated
v2.75.41 cache namespace.  It still executes only DeepX Full plus the
setup-local TensorRT Full control; Native, Energy, splits, ranking and
performance claims remain disabled.

The formal comparison must use the completed 2.75.40 Arm-B run as its 500-item
baseline.  It verifies source, preprocessing, compiler, validation cohort and
Quality-policy equality, proves the actual 500-item calibration cohort is a
subset of the 1,000-item cohort and reports paired Top-1/Top-5 changes on the
unchanged validation images.  Standard+ remains blocked unless the new DeepX
result passes its unchanged setup-local TensorRT guardrail.

## Paired DeepX preprocessing profiles for 2.75.40

Historical procedure only: do not repeat these A/B runs for 2.75.42 or any
later release, including 2.75.44. The completed experiment ran
`profiles/resnet50_v27540_deepx_preprocess_a_current_scale_only.yaml` first and
`profiles/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std.yaml` second.
Both freeze Standard quality effort at 500 calibration images, 500 validation
images and 500 bootstrap repetitions and pin the same classification dataset
manifests. They execute only DeepX Full plus its setup-local TensorRT Full
control: zero Generic rows, two Quality results and one remote invocation.
Native, Energy, ranking, splits and performance claims are disabled.

The two source profiles differ only in name, explicit classification
preprocessing mode and isolated exact-v2 cache root. A missing exact artifact
is built once even though `force_build` is false; an identical complete
receipt is restart-safe. Compare completed runs with
`scripts/verify_v27540_deepx_preprocessing_ab.py` and use
`scripts/deepx_classification_preprocessing_probe.py` for the paired float-ORT
control.

## DeepX Full Quality Canary for 2.75.39

Use `profiles/resnet50_v27539_deepx_full_quality_canary.yaml` for the targeted
DeepX quality diagnostic. It contains exactly `deepx_m1_full` and the matching
setup-local TensorRT Full Quality companion. The effective plan must contain
two expected Full Quality results, one remote invocation, zero Generic Split
rows, and no Native, Energy, ranking, or performance claim. Because it requests
no Hailo backend, the Hailo-Full preupload gate is explicitly not applicable.
The DeepX endpoint must execute the attested DXNN and export a
`deepx_dxnn_sha256:<sha256>` runtime identity; ORT CPU fallback is forbidden.
Both Full endpoints are Quality-only, so Performance completeness is not
applicable while their two exact Quality records remain mandatory.
The 2.75.39 transport repair preserves a deterministic pre-mutation SSH error
as the primary cause and suppresses collection for a remote command that never
started. Cleanup after a confirmed lease remains fail-closed. The historical
2.75.38 profile remains available for provenance.

## Historical acceptance profile for 2.75.36

`profiles/resnet50_v27536_small_acceptance.yaml` fixed Audit 4, minimum valid
3, deployment 1, seed `20260710`, gap 0, the seven established logical
profiles, Native/Full on and both Energy
paths off. The scope intentionally matches the completed 2.75.35 run so the
validator and TensorRT-cache repairs can be compared directly.

## Simplified model configuration (2.75.35)

For each model, the editor now exposes only ID, task, usage
(`Development` or `Hold-out`), ONNX path, model family, input shape and note.
It derives the former role, generalization, validation-tier and
candidate-universe fields from task and usage. Universe completeness is proven
by the generated hashed freeze artifacts instead of a manual checkbox.

Score-independent audit size, minimum valid candidates and seed are configured
once in the candidate-selection section. They are no longer repeated as hidden
per-model overrides. Existing profiles are normalized on save and still pass
the same schema validation used by the workflow.

For a Development ranking audit, choose `score_independent_audit`; do not
leave `stratified_windows` selected. Before starting, the Effective Plan must
show the audit count and the configured audit/deployment execution-union
range. For the small ResNet acceptance profile (`audit_size: 4`, deployment
shortlist: 1), the range is 4–5 candidates.

Since 2.77.0, productive candidate ranking is frozen globally to
`cut_bytes_only`: lower cut bytes first, then Boundary ID and Case ID for exact
ties. Score-independent audits and stratified graph-position windows still
define coverage; the selector chooses inside that unchanged universe. Other
ranking methods remain available only for scientific comparison reports.
Transfer-model results must not alter the frozen method, seed, quotas,
windowing or minimum-gap policy.

The frozen audit-first, deployment-second deduplicated union is the exact
generator attempt scope in 2.75.35. `requested_cases` controls the deployment
shortlist only; it cannot truncate the frozen union. Plan, Prediction and
Candidate-Universe identities must agree before generation. Accepted plus
explicitly rejected/unsupported identities must partition the complete frozen
union in deterministic order. Unclassified truncation and outside backfill
still fail closed.

The older packaged profile
`profiles/resnet50_v27535_small_acceptance.yaml` established this scope. Use
the versioned 2.75.36 successor when reproducing that historical acceptance.

For Hailo-to-TensorRT quality preparation, singleton-squeezed channel vectors
such as `[2048]` are accepted as the byte-identical canonical shape
`[1,2048,1,1]`. Other unresolved or ambiguous layout relationships remain
fail-closed and produce no performance or ranking claim.

## User-facing run modes

| Mode | Purpose | Validation items (CLS/DET) | Task-quality bootstrap |
|---|---|---:|---:|
| `smoke` | Short wiring and contract check | 16 / 12 | 25 |
| `standard` | Balanced development evaluation | 500 / 500 | 500 |
| `final` | Final Quality on the Standard path | 5,000 / 5,000 | 5,000 |

Since 2.75.30, `final` is labelled **Final Quality (Standard+)**. Its mode is
created by copying Standard and changing only its label/description and these
task-quality fields:

- `data.validation_items.classification: 5000`
- `data.validation_items.detection: 5000`
- `quality.profile_id: task_quality_final_5000`
- `quality.dataset_tier: final`
- `quality.bootstrap_repetitions: 5000`

All execution settings remain Standard: relaxed sampled integrity (24 registry
items), development campaign mode with warning enforcement, Hailo
`balanced`/optimization level 1, Generic benchmark warmup/runs `3/5`, Native
frames/warmup/repetitions `1000/100/3`, and the same cache, reporting and
Energy policies. Native defaults to on; Native Energy defaults to off and
remains an independent profile switch.

To use it, open **Evaluation Workflow → Profil erstellen/bearbeiten…**, select
run mode `final`, choose the models and logical hardware profiles, set the
Native/Energy switches, then use **Speichern & verwenden**. Verify the
effective summary before starting the normal Evaluation Workflow.

Inspect the effective central settings with:

```bash
onnx-splitpoint-run-modes status
onnx-splitpoint-run-modes show final
onnx-splitpoint-run-modes validate
```

An existing schema-10/11 central registry is migrated to schema 12 when
loaded. Untouched historical Final defaults become Final Quality, while
customized leaves are preserved. To deliberately discard custom Final values
and restore the packaged default, run:

```bash
onnx-splitpoint-run-modes reset final
```

External YAML profiles are loaded from the GUI and validated against:

```text
onnx_splitpoint_tool/resources/schemas/evaluation_profile.schema.json
```

Use a saved Evaluation Profile path for a Final Quality run. The legacy
benchmark-profile alias named `final` identifies a built-in profile; it does
not by itself select the central `final` run mode.

## Data and evidence behavior

Final Quality selects up to 5,000 validation items for each task. In the
run-mode registry, a validation count of `0` means the complete manifest, not
"disabled"; it is intentionally not the 2.75.30 default. The selected 5,000
items are still materialized and content-bound as ordinary quality evidence.

Final Quality does not add a separate `prepare-evaluated-matrix` step, does
not require campaign preflight or sealing, does not repeat a full-dataset hash
scan, and has no mandatory canary. Dataset availability, calibration/validation
separation and the normal Standard execution gates still apply.

## Optional strict campaign profiles

The following packaged inputs remain available for explicit advanced or
historical use; they are not the normal Final Quality path:

- `final_splitpoint_evaluation_v1`
- `thesis_final_evaluated_matrix_v1`
- `thesis_final_campaign_v1`

`thesis_final_evaluated_matrix_v1` retains the strict three-model
`evaluated_matrix` campaign contract. `thesis_final_campaign_v1` additionally
retains the stronger `ranking_generalization` hold-out/protocol-freeze/ranking
contract. Their `TODO` values, referenced `campaign_inputs/...` artifacts and
campaign preflight remain binding only when one of those strict templates is
intentionally selected.

## Historical claim scopes (2.75.27)

Release 2.75.27 introduced exact Final claim scopes. `evaluated_matrix` limits
conclusions to declared model/hardware cells. `ranking_generalization`
additionally requires enabled ranking comparison, canonical confirmatory
hold-outs, prospective protocol freeze, registry, attestations, prediction
approval and a development-only fitted ranking bundle. Unknown or explicitly
empty scopes fail closed; older profiles infer their former intent. These
strict scope semantics remain available without being activated merely by the
2.75.30 Final Quality run mode.
