# Artifacts & Postprocessing Contract

This document describes **what artifacts** a benchmark run produces and the
**postprocess contract** between runner → harness → GUI.

The goal is to keep the tool robust when new networks/harnesses are added.

## Canonical image preprocessing (2.75.0)

Image geometry is selected by task, never inferred from the output tensor
format:

| Task | Geometry | Color | Pad | Numeric encoding |
|---|---|---|---:|---|
| Detection | centered letterbox to the declared input size | RGB | 114 | normalized model input |
| Classification | direct resize to the declared input size | RGB | 0 | ImageNet normalization |

The semantic contract is
`onnx-splitpoint/image-preprocessing-contract`, schema version 2. Central
Quality, Hailo calibration, Runtime input manifests and build receipts carry
its SHA-256. A caller-provided geometry or scaling flag that contradicts the
contract is rejected before inference or compilation.

Detection Completed-v2 records use original-image `xyxy`, probability score
and integer model class ID. Raw accelerator heads remain a distinct physical
endpoint; their frozen host decode/NMS tail produces the attested completed
endpoint. BN6 is an output representation and does not opt out of letterbox.

DeepX Full Performance and Energy use
`deepx-sealed-runtime-input-v3`: an untimed semantic execution persists the
exact numeric runtime tensor, and both measured paths verify and replay that
same file by shape, dtype, layout, byte count and SHA-256. Detection rows are
claimable only when the same hotloop also persists a separately verifiable
Completed-v2 JSON artifact with matching content and file hashes.

The Quality request and the later canonical Native-Full row additionally join
the same source image ID/hash to the exact prepared tensor name, shape, dtype,
layout, byte count and SHA-256. These identities may not drift across measured
repetitions. The semantic manifest/tensor, DXNN and hotloop helper are confined
to their canonical artifact-role roots, with every path component checked for
symlinks before resolution. Completed-v2 writers likewise reject symlinked
destinations or parents and use exclusive temporary files plus atomic replace.

## Directory layout

Each benchmark case is executed in its own directory (e.g. `b155/`).

Inside that case directory, each provider run writes into:

- `results_cpu/` (example)
- `results_cuda/` (example)
- …

## Required artifacts per provider

Every `results_<provider>/` directory must contain:

### Validation report

- `validation_report.json`
- `validation_report.png`
- `validation_report.pdf`

These are produced by the runner and include accuracy proxy metrics, timings,
and a `viz` block with harness postprocess output.

### Harness postprocess artifacts

Every run must produce at least one **postprocess JSON** (and may produce
overlays).

Naming convention is **task-prefixed**:

- Detection (YOLO):
  - `detections_full.json`
  - `detections_composed.json`
  - optional overlays: `detections_full.png`, `detections_composed.png`

- Classification (ImageNet style):
  - `classification_full.json`
  - `classification_composed.json`
  - optional overlays: `classification_full.png`, `classification_composed.png`

Other tasks should follow the same pattern:

```
<task>_<variant>.json
<task>_<variant>.png   (optional)
```

Where `<variant>` is one of:

- `full`
- `composed`

## PostprocessResult schema

The harness returns a lightweight result object that must be JSON-serializable.

Minimal schema (see `splitpoint_runners.harness.base`):

```json
{
  "schema_version": 1,
  "task": "classification" | "detection" | "...",
  "summary_text": "optional short summary",
  "json": { "any": "json-serializable payload" },
  "overlays": {
    "main": "classification_full.png"
  }
}
```

Important:

- `overlays` paths must be **relative** (portable across remote/local).
- The runner validates the contract and surfaces violations via `contract_error`
  in `validation_report.json`.

## Label assets

To avoid “magic” label lists living in the GUI, the runner library ships
versioned assets:

- `splitpoint_runners/assets/imagenet_labels.txt`
- `splitpoint_runners/assets/coco80_labels.txt`

Harnesses load these by default. Callers can override (e.g. via `--labels`).

## Log sanitization

Captured logs are sanitized to make them readable in the GUI and exports:

- `\r` (carriage returns) are normalized to `\n`
- ANSI escape sequences (colors/cursor moves) are stripped

Warnings like ORT memcpy warnings are preserved, but displayed consistently.
