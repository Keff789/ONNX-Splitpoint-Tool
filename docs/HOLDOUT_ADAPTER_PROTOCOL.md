# Hold-out adapter protocol

This protocol separates compatibility engineering from confirmatory evidence.
Postprocessing support for a new architecture is allowed before a hold-out is
opened; outcome-informed tuning on the model that is still claimed as a
hold-out is not.

## 1. Qualify compatibility without outcomes

Before inspecting predictions, labels, numerical similarity, quality, FPS,
energy or rankings from a confirmatory model, adapter work may use only:

- graph/export metadata such as names, shapes, dtypes, opset and endpoint type;
- vendor/compiler metadata and public architecture/export specifications;
- synthetic tensors and deterministic contract fixtures;
- declared development models and calibration/development datasets.

Adapters must dispatch by a frozen endpoint-contract family, not by a literal
hold-out model ID.

## 2. Freeze the complete adapter contract

The pipeline contract manifest must content-hash the adapter implementation
sources and lock all behavior that can affect an outcome, including:

- preprocessing, input layout and normalization;
- layout conversion plus vendor/compiler provenance for quantization scale and
  zero point;
- decoder family, tensor schema, strides, anchors and DFL bins where relevant;
- confidence/NMS thresholds, label mapping and coordinate transformations;
- dataset/image identities, candidate universe, seed, ranking policy and gates.

Diagnostic sweeps are never production-contract selectors and must remain
`diagnostic_only=true` and `claim_eligible=false`.

## 3. Open the hold-out once

After the profile, adapter sources and policy manifests are frozen, execute the
hold-out. Pass, fail and unsupported are all valid confirmatory outcomes. Do
not tune the adapter using those outcomes.

An output-independent mechanical repair may be recorded as a protocol
amendment only when its independence can be demonstrated with synthetic or
development fixtures. Any change influenced by hold-out predictions, quality,
performance, energy or rankings consumes that exact model: reclassify it as
development, discard its confirmatory claim, freeze a new protocol release and
use a fresh untouched model.

## 4. Family interpretation

- Additional YOLO26 variants after YOLO26s development are within-family
  transfer hold-outs, not model-family hold-outs.
- YOLO11 can be a separate-family hold-out only if its endpoint adapter is
  specified and frozen before outcomes are viewed. A YOLO11 variant used to
  build or tune that adapter becomes development.
- MobileNet and RegNet can be separate-family classification hold-outs when
  the frozen generic preprocessing plus logits/top-k contract applies without
  adaptation. A model used to repair normalization, labels or output
  interpretation is consumed and must be replaced for confirmation.
