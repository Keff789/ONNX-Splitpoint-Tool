# Validation datasets

The clean tool ZIP does not contain validation images. This keeps releases small and avoids repeatedly shipping the same datasets.

## GUI

Prepare downloadable validation data from:

```text
Benchmark tab → Prepare validation sets…
```

The dialog can prepare:

- `COCO-50 detection`
- `Imagenette mini-200 classification`
- `Imagenette mini-500 classification` if explicitly selected
- lightweight runner test images

Prepared data is stored in the user cache:

```text
~/.onnx_splitpoint_tool/validation_datasets/
```

On Windows this resolves to:

```text
C:\Users\<user>\.onnx_splitpoint_tool\validation_datasets\
```

## CLI

Status only:

```bash
python -m onnx_splitpoint_tool.cli validation-assets status
```

Prepare the default small sets, currently COCO-50 and Imagenette mini-200:

```bash
python -m onnx_splitpoint_tool.cli validation-assets prepare
```

Prepare everything, including Imagenette mini-500:

```bash
python -m onnx_splitpoint_tool.cli validation-assets prepare --imagenette500
```

Legacy alias:

```bash
python scripts/prepare_validation_sets.py --all
```

## Detection: COCO-50

COCO-50 is a fixed 50-image subset of COCO val2017 used for lightweight detection semantic validation and Mini-COCO AP50 reporting. The tool ships only a compact manifest with image ids and annotations. Images are downloaded on demand into:

```text
~/.onnx_splitpoint_tool/validation_datasets/detection/coco_50_data/
```

The downloader uses the public COCO val2017 image host. Some Windows/Python installations report a certificate hostname mismatch for the HTTPS endpoint, so the tool now prefers the official plain HTTP image URL and keeps HTTPS as a fallback.

When a detection benchmark suite is generated and the semantic validation field is empty, the tool copies the prepared COCO-50 set into the suite under:

```text
resources/validation/coco_50_data/
```

If COCO-50 is not prepared, the suite can still be generated, but dataset-level semantic validation is disabled or reduced depending on the selected options.

## Classification: ImageNet mini and Imagenette mini

There are two different classification validation options:

### Local ImageNet-mini presets

These are preferred for final ImageNet-style evaluation, but they require a local ImageNet validation set and ground-truth file. The tool cannot legally ship or auto-download the original ImageNet validation images.

Prepare them with:

```text
Benchmark tab → Classification preset → Import…
```

Supported local ImageNet presets:

- `imagenet_val_mini_200`
- `imagenet_val_mini_500`

### Downloadable Imagenette mini presets

For a public, downloadable classification fallback, the tool can download fast.ai's Imagenette2-320 archive and build small manifests mapped to ImageNet-1k class ids.

Supported downloadable presets:

- `imagenette_val_mini_200`
- `imagenette_val_mini_500`

This is useful for smoke/regression validation of ImageNet-pretrained classifiers. It is not a replacement for a broad ImageNet-mini final evaluation because it covers only 10 ImageNet classes.

## Runner test images

Runner test images are also prepared outside the ZIP. If they are not available, the generated runner falls back to a tiny embedded placeholder image so basic smoke tests still work.
