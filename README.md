## v47 note

v47 keeps the v46 YOLO26 Full-Hailo baseline behavior, but fixes the next runtime/evaluation issues: mixed Hailo↔TensorRT run labels are no longer passed to DFC as hw_arch values; lean result bundles synthesize/include v42/v47 pipeline summaries and status matrices; and YOLO26 benchmark generation reserves early Hailo→TensorRT candidates for throughput-first 5-split runs while still keeping a verified Hailo/Hailo candidate when available.

## v47 note

v47 normalizes mixed Hailo/TensorRT benchmark labels before HEF generation. Pipeline labels such as `hailo8_to_trt` and `trt_to_hailo8` stay in the benchmark plan, but Hailo DFC builds now use the physical chip target (`hailo8`, `hailo8r`, `hailo8l`) and reuse those artifacts for mixed runs. Lean result bundles also keep the v42 pipeline/status summaries.

## v46 note

v46 forces the YOLO26 Hailo Full baseline attempt before prefiltering, global probing, or split-case builds. When Hailo is selected for YOLO26, the generator now treats Hailo Full as a required suite-level reference baseline for Hailo-only vs. TensorRT-only vs. Hailo→TensorRT comparison. The decoded Full graph is tried first and the one2one raw-head fallback is used if the end-to-end output tail blocks Hailo. If Full-Hailo fails, the failure is recorded as a structured suite-level result and split generation continues.

## v44 note

v44 hardens the YOLO26 path using the older YOLO26L generation logs. It keeps Hailo Full available for one2one raw-head fallback when decoded Full fails at the end-to-end `TopK/GatherElements/ReduceMax/Squeeze` tail, avoids treating `/model.23/Transpose*` parser hints as true Hailo-Part2 compatibility, keeps useful YOLO26 splits as `Hailo Part1 -> TensorRT/ORT Part2` when Hailo-Part2 is not safe, and adds a conservative static guard against known-bad late-head Part1 boundaries that repeatedly failed with `format_conversion* Agent infeasible`.


## v43 note

v43 adds a YOLO26-specific Hailo policy: it detects `one2one_cv2/cv3` detection-head endpoints, avoids late `/model.23/Transpose` Part2 fallbacks that trigger Hailo Concat/format-conversion allocation failures, and keeps YOLO26 Hailo-Part1 → TensorRT/ORT-Part2 cases clean when Hailo-Part2 is not truly available.

## v42 note

v42 keeps the v41l YOLO/Hailo raw-head split generation policy, but improves runtime
validation and reporting for thesis evaluation. New per-case reports now expose fair
Hailo raw-head full E2E timing when a host-tail decoder is available, composed-stage
phase timings, stricter semantic-validation gating, raw-head tensor drift diagnostics,
and a suite-level `v42_pipeline_summary.*` focused on heterogeneous streaming FPS.

## v41l note


v41l tightens the YOLO/Hailo Part2 fallback policy. When a raw detection-head
contract is active, Part2 no longer accepts Hailo parser-suggested `/model.23/Concat`
endpoints as a substitute for the raw cv2/cv3 head tensors. Those suggested endpoints
avoid the DFL parser error but can still fail during Hailo translation with concat layout
errors. The generator now rejects those late boundaries and backfills candidates whose
Part2 can terminate at the complete raw-head output set.


Fixes Hailo Part2 accelerator-prefix builds that failed with `Couldn't find inputs from ONNX proto` by pruning dangling ONNX graph inputs before Hailo parsing/building.

# ONNX Split-Point Tool

## v41j note

This build adds a stale-suite guard for YOLO/Hailo benchmarksets. If a remote run uses an old suite that lacks `part2_hailo_prefix` / `part2_host_tail` artifacts, the tool now warns that the suite must be regenerated instead of silently skipping Hailo Part2 host-tail runs. It also strengthens Hailo YOLO raw-head decoding by avoiding a second sigmoid on class heads that are already probabilities and by preventing valid six-head outputs from falling through to the old reshape-prone fallback decoder.


GUI and CLI tooling for analysing ONNX graphs, exporting split models, generating benchmark suites, and comparing full vs split deployments across CPU/CUDA/TensorRT/Hailo configurations.

This is a clean distribution build. It intentionally does **not** ship image-heavy validation datasets. Prepare them once after installation with the GUI button **Prepare validation sets…** or the CLI command shown below.

## Quick start

```bash
./start_gui.sh
```

The startup script creates/uses `.venv`, installs missing runtime helpers, and launches the GUI.

## Validation datasets

Detection validation uses a lightweight COCO-50 subset. Classification smoke validation can use a downloadable Imagenette-mini preset mapped to ImageNet class IDs. The clean ZIP contains only compact manifests, not images. Prepare the defaults once:

```bash
python -m onnx_splitpoint_tool.cli validation-assets prepare
```

or in the GUI:

```text
Benchmark tab → Prepare validation sets…
```

For final ImageNet-style classification evaluation, local ImageNet-mini presets can still be imported from an ImageNet validation folder:

```text
Benchmark tab → Classification preset → Import…
```

See `docs/VALIDATION_DATASETS.md` for details.

## YOLO model preparation

For Ultralytics YOLO detectors, normal benchmark-set generation now consumes prepared Hailo raw-head baselines automatically and can also retry the suite full-Hailo build with raw detection-head endpoints when the decoded YOLO tail fails. For Hailo Part2, the generator can now build a parser-safe accelerator prefix and keep the DFL/decode tail as a host-side ONNX Runtime tail on the Jetson. The **Prepare current model…** action remains useful as an advanced diagnostic/screening step, but it is no longer required for the normal benchmark workflow. See `docs/MODEL_PREPARATION.md`, `docs/RELEASE_v41h.md`, and `docs/RELEASE_v41i.md`.

## Final evaluation workflow

The tool supports versioned evaluation profiles, including the final thesis profile and a smoke/regression profile. Profiles can be selected in the Benchmark tab or inspected with:

```bash
python -m onnx_splitpoint_tool.cli benchmark-profile --list
python -m onnx_splitpoint_tool.cli benchmark-profile final --model /path/to/model.onnx
```

## Clean release contents

Included:

- source code and GUI
- benchmark runner templates
- Hailo helper scripts
- evaluation profiles and schemas
- compact validation manifests
- documentation

Not included:

- `__pycache__`, `.pytest_cache`, compiled Python files
- historical patch/changelog files
- image-heavy validation datasets
- generated benchmark/results folders


### v41i note

The benchmark runner now decodes Ultralytics YOLO decoded outputs (`[1, 84, 8400]`) and Hailo raw-head outputs (`80x80x64`, `80x80x80`, `40x40x64`, `40x40x80`, `20x20x64`, `20x20x80`) through the same proxy-detection path. This makes the Hailo raw-head full baseline usable for CPU-vs-Hailo backend-drift and semantic dataset checks. Validation aggregation now gates only end-to-end variants (`composed` or `full`); `part1` and `part2` remain diagnostic timing stages unless they are the primary requested variant.
