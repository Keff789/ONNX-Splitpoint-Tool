# DeepX DX-M1 resources

DeepX is modeled as an accelerator backend parallel to Hailo.

- **Build/compiler side:** `dx-all-suite/dx-compiler` / DX-COM on x86 Linux.
- **Runtime side:** DX-RT / `dx_engine` / `run_model` on the target machine.
- **Artifact:** ONNX → `.dxnn`, analogous to Hailo ONNX → `.hef`.

The Hardware tab manages environment diagnostics and provisioning. It may checkout
`dx-all-suite` and create/repair compiler/runtime Python venvs, but it does **not**
install kernel drivers or flash firmware automatically.

Default local paths:

```bash
~/dx-all-suite
~/dx-all-suite/dx-compiler/venv-dx-compiler-local
~/venvs/deepx-runtime
~/Models/BackendArtifacts/deepx
```

Override paths via environment variables or `~/.onnx_splitpoint_tool/build_environments.yaml`:

```bash
export DX_ALL_SUITE_ROOT=~/dx-all-suite
export DEEPX_DX_ALL_SUITE_ROOT=~/dx-all-suite
export DEEPX_COMPILER_VENV=~/dx-all-suite/dx-compiler/venv-dx-compiler-local
export DEEPX_RUNTIME_VENV=~/venvs/deepx-runtime
```

The package now includes small DeepX helper modules:

```text
onnx_splitpoint_tool/deepx/config.py      # DX-COM config generation
onnx_splitpoint_tool/deepx/compiler.py    # ONNX -> DXNN wrapper
onnx_splitpoint_tool/deepx/artifacts.py   # DXNN cache helpers
onnx_splitpoint_tool/deepx/env_status.py  # status/provisioning diagnostics
```

Full DeepX execution is integrated into Evaluation Workflow profiles. The
workflow can compile or reuse receipt-bound DXNN artifacts, run DeepX Full and
supported split rows on the configured target, and bind setup-local TensorRT
Full quality evidence. Release-specific profiles and preflights still decide
which DeepX axes are scientifically admissible; environment preparation alone
never authorizes a hardware run.
