# Hailo DFC wheels (optional)

This tool can run a Hailo DFC *parse/translate* feasibility check (and, later, HEF generation).

Because **Hailo-8 and Hailo-10 require different DFC versions**, the tool supports **managed DFC profiles**.

## Where to put wheels

Copy the Hailo DFC Linux wheel(s) into:

- `onnx_splitpoint_tool/resources/hailo/hailo8/`  (DFC 3.x, Hailo-8 family)
- `onnx_splitpoint_tool/resources/hailo/hailo10/` (DFC 5.x, Hailo-10 family)

## Provision venvs (WSL / Linux)

Inside Linux/WSL, run:

```bash
./scripts/provision_hailo_dfcs_wsl.sh --all
```

If you created a repo-local venv (e.g. `uv venv --python 3.10`), the provision script will auto-use `./.venv/bin/python` when you do not pass `--python` explicitly.


If your WSL distro is **Ubuntu 20.04** (python3=3.8) and you cannot install python3.10 via `apt`,
install Python 3.10 via **pyenv** or **uv** and pass it explicitly:

```bash
./scripts/provision_hailo_dfcs_wsl.sh --all --python /path/to/python3.10
```

This will create venvs under:

- `~/.onnx_splitpoint_tool/hailo/venv_hailo8/`
- `~/.onnx_splitpoint_tool/hailo/venv_hailo10/`

The GUI can then use `WSL venv: auto` to pick the correct venv based on the selected `HW arch`.

## Important: do not upgrade ONNX/protobuf inside the DFC venvs

The Hailo DFC wheels pin specific dependency versions (notably `onnx` and `protobuf`).

- Example (DFC 3.33 / Hailo-8): `onnx==1.16.0` and `protobuf==3.20.3`

Upgrading `onnx` typically upgrades `protobuf` too, which breaks `hailo_sdk_client` imports with errors like:

`TypeError: Descriptors cannot be created directly.`

The provision script therefore **does not upgrade onnx by default**.

To make this harder to break accidentally, provisioning also writes a small `pip_constraints.txt` inside each managed venv and patches `bin/activate` to export `PIP_CONSTRAINT=...`.
That way, even if you run `pip install onnx` inside the managed venv, pip should keep the pinned versions (or fail loudly).

If you *did* end up with a broken env, delete the managed venv and re-provision.

> Note: Hailo DFC wheels are built for a specific Python version (often **Python 3.10**).
> The provision script itself may run on an older Python, but the **venv** must be created
> with a compatible Python (use `--python` if needed).
