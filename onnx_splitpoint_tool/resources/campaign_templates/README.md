# Final campaign template inputs

Copy this directory outside the clean source tree and edit the four configuration files. Set `locked: true` only after each file exactly matches the code path used by the canonical Full-ONNX reference, full-backend baselines, Generic Runner splits, and Native FIFO rows.

Recommended dataset roles:

- classification calibration: a fixed subset of ImageNet training data;
- classification validation: ILSVRC2012 validation;
- detection calibration: a fixed subset of COCO 2017 train;
- detection validation: COCO 2017 val.

The tool does not redistribute these datasets. Generate content-addressed manifests with `onnx-splitpoint-campaign dataset-manifest`, then run `onnx-splitpoint-campaign preflight`.

For `evaluated_matrix`, exact model bytes, dataset manifests, locked pipeline
contracts, Native/Full baselines and energy calibration remain mandatory; no
hold-out registry or ranking bundle is consumed. For
`ranking_generalization`, hold-out entries must additionally resolve to exact
local model artefacts and the strict registry/protocol freeze verifies their
roles, hashes and prospective attestations. Ranking-fit input rows are used
only when `evaluation_role: development` is explicit.
