# ONNX Split-Point Tool (GUI + CLI)

Single-split boundary analysis for ONNX graphs. The tool enumerates all topological boundaries and ranks them by
communication volume, compute balance, and (optionally) a simple link/latency model.

## Repo layout (refactored)

- `onnx_splitpoint_tool/` – Python package (core code)
  - `api.py` – public API / compatibility re-exports
  - `cli.py` – CLI implementation
  - `gui_app.py` – Tkinter GUI
  - `onnx_utils.py` – parsing + graph utilities
  - `metrics.py` – cost/FLOPs/memory estimation + ranking
  - `pruning.py` – skip-/block-aware pruning
  - `split_export.py` – split export + runner skeleton + context diagrams
- `analyse_and_split.py` – thin wrapper entrypoint (kept for backwards compatibility)
- `analyse_and_split_gui.py` – thin wrapper entrypoint (kept for backwards compatibility)
- `download_model_zoo_examples.py` – helper to download ONNX model zoo examples
- `model_zoo_manifest.json` – curated model list for the downloader

## Install

```bash
pip install onnx numpy matplotlib pillow
# optional (validation/runner):
pip install onnxruntime
```

## Optional: Hailo DFC integration (Linux or Windows via WSL2)

The GUI can optionally run a **Hailo feasibility check (parse-only)** during ranking.
This is useful to automatically prune split candidates where a partition cannot be translated by the Hailo toolchain.

### Linux (native)

- Install the Hailo Dataflow Compiler (DFC) Python wheel into the same Python environment as this tool.
- In the GUI enable **Hailo check** and set **Backend = local** (or keep **auto**).

### Windows + WSL2 (recommended)

The Hailo DFC wheel is Linux-only. Recommended setup:

1. Install WSL2 + an Ubuntu distro.
2. Create a WSL virtualenv (default expected path: `~/hailo_dfc_venv`) and install the Hailo DFC wheel there.
   A helper script is included:
   `./scripts/setup_hailo_dfc_wsl.sh /path/to/hailo_dataflow_compiler-*.whl`
3. Run the GUI on Windows, enable **Hailo check**, set **Backend = wsl** (or **auto**), and point it to:
   - **WSL venv**: `~/hailo_dfc_venv/bin/activate`
   - **WSL distro**: optional (leave empty to use the default WSL distro)

Notes:

- The tool calls `wsl.exe` and runs a helper script inside WSL; you do **not** need to activate the venv manually.
- If your model files are on `C:`/`D:`, WSL can access them via `/mnt/c/...` and `/mnt/d/...`.
- A small helper for interactive shells is included:
  `source ./scripts/activate_hailo_dfc_venv.sh`

## Run

### GUI

```bash
python analyse_and_split_gui.py
```

### CLI

```bash
python analyse_and_split.py path/to/model.onnx --topk 10
```

## Highlights

- Export split-context diagrams (full + cut-flow) with configurable context hops (0..3).
- Suggest split boundaries based on activation-communication + compute balance.
- Skip-/Block-aware candidate pruning (heuristic: avoids splitting inside long skip/residual blocks).
- Link-model plugin for latency/energy with optional constraints (bandwidth/latency/energy).
- Pareto export + clean **System / Workload** separation (`system_config.json` + `workload_profile.json`).
- **Peak activation memory (approx, from value spans):**
  - per boundary: live-set bytes (same basis as Comm(b))
  - per partition: `peak_left[b]=max_{i<=b} live(i)`, `peak_right[b]=max_{i>=b} live(i)`
  - optional constraints: max act mem left/right (fits SRAM/VRAM)
- Split directly from the GUI (export part1/part2 ONNX).
- Strict boundary option (rejects splits where part2 still depends on original inputs).
- Optional onnxruntime validation (`full(x) ~= part2(part1(x))`).
- Runner skeleton generator (`run_split_onnxruntime.py`)
  - supports classification-style top-k plots
  - supports YOLO-style detection visualization + JSON export
- **Benchmark set generator (GUI button: “Benchmark set…”)**
  - exports one subfolder per selected split (models + runner)
  - writes `benchmark_suite.py` to run all cases and aggregate results/plots
  - also exports *paper assets* into the benchmark folder root:
    - `split_candidates.tex`
    - plots (`analysis_*.pdf` / `analysis_*.svg`)
    - `system_config.json`, `workload_profile.json`
    - `pareto_export.csv`, `candidate_pruning.json`

## Notes

- If Graphviz `dot` is installed, `.dot` files are rendered to SVG/PDF automatically. Otherwise a matplotlib fallback diagram is created.
- Windows: do not run `.bat` files with Python. Double click, or run from PowerShell directly.
