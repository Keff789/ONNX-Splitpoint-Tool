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

### Managed DFC profiles (Hailo-8 + Hailo-10)

Hailo-8 and Hailo-10 require different DFC/SDK versions. This repo supports both via **managed profiles**:

- `hailo8`  → DFC 3.x (e.g. 3.33)
- `hailo10` → DFC 5.x (e.g. 5.2)

You place the vendor wheels into:

- `onnx_splitpoint_tool/resources/hailo/hailo8/`
- `onnx_splitpoint_tool/resources/hailo/hailo10/`

Then you run the provisioning helper (creates one venv per profile under `~/.onnx_splitpoint_tool/hailo/`):

```bash
# inside Linux (native) or inside WSL
./scripts/provision_hailo_dfcs_wsl.sh --all
```

Important: the venvs must be created with **Python 3.10+** (DFC deps like `jax` have no wheels for Python 3.8).
On Ubuntu/WSL you typically need:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv
```

### GUI usage

- Set **Backend = auto** (or **wsl** on Windows).
- Leave **WSL venv override = auto** (recommended).
- The hardware tab shows **status badges** (Hailo-8 / Hailo-10) at startup; use **Refresh status** if needed.

Notes:

- Parse-only results are cached across runs (by sub-model hash) to avoid re-running DFC translation during ranking.
  Clear via **Clear cache** or delete `~/.onnx_splitpoint_tool/hailo_parse_cache.json`.

## Run

### GUI

```bash
# New GUI (panel-based)
python -m onnx_splitpoint_tool.gui.app

# Legacy GUI (kept for compatibility)
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
- Strict boundary option (rejects splits where part2 needs additional *intermediate* activations beyond the cut tensors; original graph inputs are allowed).
- Optional onnxruntime validation (`full(x) ~= part2(part1(x))`).
- Runner skeleton generator (`run_split_onnxruntime.py`)
  - generic ORT benchmark runner for: full / part1 / part2 / composed
  - supports input feeds via NPZ (`--inputs-npz`) and saving generated inputs (`--save-inputs-npz`)
  - supports dumping standalone split interfaces as NPZ (`--dump-interface {right,left,min,either}`) with metadata
  - supports CPU/CUDA/TensorRT (engine cache + fast-build preset) and writes `validation_report.json`

**Example (dump interface for Part 2 / “right” side):**
```bash
./run_split_onnxruntime.sh --provider tensorrt --dump-interface right --dump-interface-out results/interface
# outputs: results/interface_right.npz (and metadata in __meta__)
```

**Example (LLM-style shapes via overrides):**
```bash
./run_split_onnxruntime.sh --provider cuda --shape-override "input_ids=1x128 attention_mask=1x128"
```

**Example (reproducible inputs to NPZ):**
```bash
./run_split_onnxruntime.sh --provider cpu --seed 0 --save-inputs-npz results/inputs_full.npz
./run_split_onnxruntime.sh --provider cpu --inputs-npz results/inputs_full.npz
```

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

### External-data ONNX models (`*.onnx` + `*.onnx.data`)

Large models exported with ONNX external data are supported.

When exporting splits/benchmark sets, the tool tries to make the output folder usable by:

- creating a **hardlink** to the referenced `*.data` file (fast, no extra disk use; requires same filesystem),
- or falling back to **symlink**/**copy**,
- and if none of the above is possible, it rewrites the ONNX external-data `location` to an **absolute path** (works locally, not portable).

### GUI logs

The GUI writes logs to both:

- `~/.onnx_splitpoint_tool/gui.log`
- `./gui.log` (current working directory)
