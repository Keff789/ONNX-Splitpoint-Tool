#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Tuple, Optional

import torch


def load_module_from_file(module_path: str):
    """
    Dynamically import a Python module from a file path.
    Adds the module directory to sys.path so relative imports inside work more often.
    """
    module_path = os.path.abspath(module_path)
    module_dir = os.path.dirname(module_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    """
    Parse input shape like "1,3,224,224" -> (1,3,224,224)
    """
    parts = [p.strip() for p in shape_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --input-shape. Example: 1,3,224,224")
    shape = tuple(int(p) for p in parts)
    if any(d <= 0 for d in shape):
        raise ValueError(f"All dimensions must be > 0. Got: {shape}")
    return shape


def clean_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove common prefixes like 'module.' from DDP-trained checkpoints.
    """
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        cleaned[nk] = v
    return cleaned


def load_checkpoint_state_dict(ckpt_path: str, state_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Load checkpoint and return a state_dict.
    Handles common formats:
      - pure state_dict
      - dict with keys like 'state_dict', 'model', 'model_state_dict'
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        # If user specified which key holds weights, use it.
        if state_key:
            if state_key not in ckpt:
                raise KeyError(f"--state-key '{state_key}' not found in checkpoint keys: {list(ckpt.keys())}")
            sd = ckpt[state_key]
            if not isinstance(sd, dict):
                raise TypeError(f"checkpoint['{state_key}'] is not a dict state_dict.")
            return clean_state_dict_keys(sd)

        # Try common keys automatically
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return clean_state_dict_keys(ckpt[key])

        # Maybe it's already a state_dict-shaped dict
        # Heuristic: if many values are tensors
        tensor_values = sum(1 for v in ckpt.values() if torch.is_tensor(v))
        if tensor_values > 0 and tensor_values / max(1, len(ckpt)) > 0.5:
            return clean_state_dict_keys(ckpt)

    raise ValueError(
        "Could not interpret checkpoint format as a state_dict. "
        "Try providing --state-key to indicate which key contains the weights."
    )


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    out_path: str,
    opset: int,
    dynamic_batch: bool,
    input_name: str = "input",
    output_name: str = "output",
):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            out_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
        )


def validate_onnx(onnx_path: str, model: torch.nn.Module, dummy_input: torch.Tensor):
    """
    Optional: compare PyTorch output vs ONNXRuntime output on the dummy input.
    Requires: onnx, onnxruntime
    """
    try:
        import onnx  # type: ignore
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        print(f"[WARN] Validation skipped (onnx/onnxruntime not installed): {e}")
        return

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    print("[OK] onnx.checker.check_model passed.")

    # PyTorch output
    model.eval()
    with torch.no_grad():
        pt_out = model(dummy_input).detach().cpu()

    # ONNXRuntime output
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    ort_outs = sess.run(None, {inp_name: dummy_input.detach().cpu().numpy()})
    ort_out = torch.from_numpy(ort_outs[0])

    # Compare
    max_abs = (pt_out - ort_out).abs().max().item()
    mean_abs = (pt_out - ort_out).abs().mean().item()
    print(f"[OK] Validation compare: max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}")


def main():
    ap = argparse.ArgumentParser(description="Export a PyTorch model defined in a .py file to ONNX.")
    ap.add_argument("--model-file", required=True, help="Path to the .py file containing the model class.")
    ap.add_argument("--class-name", required=True, help="Name of the torch.nn.Module class in that file.")
    ap.add_argument("--checkpoint", default=None, help="Optional .pth/.pt checkpoint to load weights from.")
    ap.add_argument("--state-key", default=None, help="Optional key in checkpoint dict that holds the state_dict.")
    ap.add_argument("--onnx-out", required=True, help="Output .onnx path.")
    ap.add_argument("--input-shape", required=True, help='Dummy input shape, e.g. "1,3,224,224".')
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"], help="Dummy input dtype.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for export.")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version (commonly 13-18).")
    ap.add_argument("--dynamic-batch", action="store_true", help="Make batch dimension dynamic.")
    ap.add_argument("--init-kwargs", default="{}", help='JSON dict of kwargs passed to model constructor.')
    ap.add_argument("--validate", action="store_true", help="Run ONNX validation with onnxruntime.")
    args = ap.parse_args()

    # Load module + class
    module = load_module_from_file(args.model_file)
    if not hasattr(module, args.class_name):
        raise AttributeError(f"Class '{args.class_name}' not found in {args.model_file}.")
    ModelCls = getattr(module, args.class_name)

    # Parse kwargs for model init
    try:
        init_kwargs = json.loads(args.init_kwargs)
        if not isinstance(init_kwargs, dict):
            raise ValueError("init-kwargs must be a JSON dict like: '{\"num_classes\": 1000}'")
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse --init-kwargs as JSON: {e}")

    # Instantiate
    model = ModelCls(**init_kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{args.class_name} did not produce a torch.nn.Module instance.")

    # Load weights
    if args.checkpoint:
        sd = load_checkpoint_state_dict(args.checkpoint, state_key=args.state_key)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INFO] Loaded checkpoint: {args.checkpoint}")
        if missing:
            print(f"[WARN] Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    # Prepare dummy input
    shape = parse_shape(args.input_shape)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    model = model.to(device).eval()
    dummy = torch.randn(*shape, dtype=dtype, device=device)

    # Export
    export_onnx(
        model=model,
        dummy_input=dummy,
        out_path=args.onnx_out,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
    )
    print(f"[OK] Exported ONNX to: {args.onnx_out}")

    # Optional validation
    if args.validate:
        validate_onnx(args.onnx_out, model, dummy)


if __name__ == "__main__":
    main()
