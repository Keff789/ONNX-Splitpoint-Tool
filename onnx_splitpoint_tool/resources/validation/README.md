# Validation resources in clean releases

Image-heavy validation data is not bundled here. The clean release ships only a
compact COCO-50 manifest (`coco_50_manifest.json`). Use the GUI button
`Prepare validation sets…` or run:

```bash
python -m onnx_splitpoint_tool.cli prepare-validation-sets --coco50
```

The prepared images and per-image annotation JSONs are stored under:

```text
~/.onnx_splitpoint_tool/validation_datasets/detection/coco_50_data
```
