# Artifacts & Postprocessing Contract

This document describes **what artifacts** a benchmark run produces and the
**postprocess contract** between runner → harness → GUI.

The goal is to keep the tool robust when new networks/harnesses are added.

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
