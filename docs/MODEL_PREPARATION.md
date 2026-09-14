# Model preparation and YOLO full-Hailo handling

Since 2.75.0, every Hailo image build seals the canonical task preprocessing
contract before cache lookup or DFC compilation. The generated
`hailo_hef_build_receipt.json` binds the source/compiler ONNX hashes, HEF hash
and size, architecture, SDK, calibration identity and preprocessing SHA. A
bare `compiled.hef`, a receipt with a different preprocessing contract, or a
legacy cache entry is not reusable. Detection calibration therefore uses the
same centered RGB letterbox with pad 114 as Central Quality; Classification
keeps direct resize plus ImageNet normalization.

The v2 receipt also seals requested and effective calibration counts, storage
mode and the memory cap. The effective count is determined before cache lookup
from the compiler input shape and any authoritative calibration dataset
ceiling, then checked again after translation and materialization. Source and
compiler ONNX remain separate physical files; runtime and BenchmarkSet binding
hash both and accept Hailo-10/Hailo-10H only as the documented legacy naming
alias. Hailo-8, Hailo-8L and Hailo-8R are never interchangeable.

For strict final evaluation, every selected model should have a comparable Hailo full-model baseline. Some YOLO ONNX exports cannot be compiled as a decoded detector graph because Hailo rejects output-tail operators such as DFL reshape, TopK, GatherElements, NonMaxSuppression, or dynamic indexing.

## Recommended workflow

For normal benchmark-set generation you no longer need to run a separate manual preparation step first. The benchmark generator handles the YOLO full-Hailo path in this order:

1. Reuse a previously prepared full-Hailo HEF when the selected model sidecar or `_prepared_models` cache records one.
2. Build the requested full-Hailo graph directly when no reusable HEF exists.
3. If the decoded YOLO full-Hailo build fails and raw head endpoints can be inferred, retry once with raw detection-head Conv endpoints.

This keeps the final workflow compact: load/analyse the model, select the benchmark profile or accelerator matrix, generate the benchmark set, then run/collect results.

## Manual preparation screen

The manual screen remains available as an advanced diagnostic tool:

```text
Benchmark → Model preparation → Prepare current model…
```

It probes the current ONNX, tries supported Ultralytics export variants, and writes endpoint/cached-HEF metadata into the selected model sidecar. If a different ONNX export variant is selected, rerun Analyse on that prepared ONNX before generating the benchmark set. If the selected variant is only a copied current ONNX with a raw-head endpoint contract, the graph itself is unchanged.

## YOLO raw detection-head endpoint fallback

For YOLO11/YOLO10-style detector exports, Hailo may reject the decoded output tail but still support the network up to the raw detection head. The raw-head deployment ends at the six final detection-head Conv nodes, for example in YOLO11:

```text
/model.23/cv2.0/cv2.0.2/Conv
/model.23/cv3.0/cv3.0.2/Conv
/model.23/cv2.1/cv2.1.2/Conv
/model.23/cv3.1/cv3.1.2/Conv
/model.23/cv2.2/cv2.2.2/Conv
/model.23/cv3.2/cv3.2.2/Conv
```

A raw-head full-Hailo baseline means: Hailo executes the detector backbone/neck/head up to raw outputs; decode/NMS remains a host-side post-processing contract. The benchmark metadata records this explicitly as `full_endpoint_mode: raw_detection_head` and `full_output_contract.mode: raw_detection_head`.

The parser-suggested decode-tail probe can still be forced for diagnostics with:

```text
OSP_YOLO_PARSER_SUGGESTED_TIER2=1
```

## v41g note

Benchmark-set generation now copies a cached prepared full-Hailo HEF into the suite when available. When no cached HEF exists, generation can perform the raw-head fallback itself and place the successful HEF at:

```text
hailo/<hw_arch>/full/compiled.hef
```

Case manifests also inherit the full-Hailo output contract so the runner does not treat raw-head tensors as decoded detector outputs.
