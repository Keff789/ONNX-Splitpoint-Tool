# Quick start

```bash
./start_gui.sh
```

The GUI is organized around three common flows:

1. **Analyse**: load an ONNX graph and inspect split candidates.
2. **Benchmark**: generate a suite, optionally prepare YOLO variants, and run local/remote benchmarks.
3. **Benchmark-Analyse**: export a compact decision summary with FPS/latency/quality/drift plots.

## First run after installing a clean release

Prepare the detection validation subset once:

```bash
python -m onnx_splitpoint_tool.cli prepare-validation-sets --coco50
```

Or use the GUI button:

```text
Benchmark → Accelerators to benchmark → Prepare validation sets…
```

Classification presets are imported from a local ImageNet validation folder using the `Import…` button next to the Classification preset selector.
