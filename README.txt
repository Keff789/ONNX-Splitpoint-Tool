ONNX Split-Point Analyser (GUI + CLI)

Files:
  - analyse_and_split_gui.py : Tkinter GUI
  - analyse_and_split.py     : Core library + optional CLI

Highlights:
  - Export split-context diagrams (full + cut-flow) with configurable context hops (0..3).
  - Suggest split boundaries based on activation-communication + compute balance
  - Skip-/Block-aware candidate pruning (heuristic: avoids splitting inside long skip/residual blocks)
  - Link-model plugin for latency/energy with optional constraints (bandwidth/latency/energy)
  - Pareto export + clean System/Workload separation (system_config.json + workload_profile.json)
  - Peak activation memory (approx, from value spans):
      * per boundary: L[b] live-set bytes (same basis as Comm(b))
      * per partition: peak_left[b]=max_{i<=b} L[i], peak_right[b]=max_{i>=b} L[i]
      * optional constraints: Max act mem left/right (fits SRAM/VRAM)
  - Split directly from the GUI (export part1/part2 ONNX)
  - Strict boundary option (rejects splits where part2 still depends on original inputs)
  - Optional onnxruntime validation (full(x) ~= part2(part1(x)))
  - Runner skeleton generator (run_split_onnxruntime.py)
      * supports common CV outputs:
          - YOLO-style detection: draws bounding boxes + exports detections.json
          - ImageNet-like classification: exports top-k plot + overlay
      * YOLO preprocessing supports --yolo-scale {auto,norm,raw}
        (auto probes norm vs raw using one inference each)
      * Robust YOLO output decoding:
          - supports both "already-decoded" flat outputs (xywh/xyxy)
          - and "raw head" flat outputs concatenated across strides (8/16/32)
        The runner tries both interpretations and picks the more plausible one.
  - Split-context export:
      * a compact visualization of the split boundary (graph neighborhood + cut tensors)
      * cut tensors are rendered as ellipse nodes between Left/Right clusters
      * crossing edges into the Right side are thick red; same-side consumers (if shown) are dashed gray
      * exported as: split_context_b<boundary>.svg / .pdf / .png (and .dot)
      * If Graphviz `dot` is installed, the .dot is rendered to SVG/PDF.
        Otherwise, a matplotlib fallback diagram is created.

Requirements:
  pip install onnx numpy matplotlib pillow

Optional (for validation / runner):
  pip install onnxruntime

Run:
  python analyse_and_split_gui.py

  - NEW: Benchmark set generator (GUI button: "Benchmark set…")
      * exports the current top-k splits into a single folder
      * includes one subfolder per split (models + runner)
      * includes benchmark_suite.py to run all cases and aggregate results/plots
      * ALSO exports paper assets into the benchmark folder root:
          - split_candidates.tex
          - analysis_plots_overview.{pdf,svg}
          - analysis_activation_bytes.{pdf,svg}
          - analysis_cumulative_compute.{pdf,svg}
          - analysis_pareto_comm_imbalance.{pdf,svg}
          - analysis_latency_model.{pdf,svg}
          - analysis_peak_activation_memory.{pdf,svg}
          - system_config.json
          - workload_profile.json
          - pareto_export.csv
          - candidate_pruning.json

Notes on split export artifacts:
- The split export folder contains:
    * <model>_part1_b<boundary>.onnx
    * <model>_part2_b<boundary>.onnx
    * split_manifest.json
    * split_context_b<boundary>.dot / .svg / .pdf / .png
    * run_split_onnxruntime.py (+ .bat / .sh wrappers)

- Runner outputs are written into a subfolder by default:
    results/validation_report.json
    results/validation_report.png
    results/validation_report.pdf
    results/detections_*.png + detections_*.json (for YOLO-style models)
    results/classification_*.png (for classification models)
  You can change this via:  --out-dir <folder>

- Important: on Windows, do NOT run .bat files with Python.
  Use either:
    * double click the .bat
    * or run it directly from PowerShell:  .\run_split_onnxruntime.bat
