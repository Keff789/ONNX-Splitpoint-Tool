"""Command line interface for ONNX Split-Point Tool."""

from __future__ import annotations

import argparse
from typing import List, Optional

import onnx
from onnx import shape_inference

from .metrics import (
    boundary_costs,
    boundary_tensor_counts,
    compute_boundary_flops_prefix,
    compute_scores_for_candidates,
    compute_tensor_bytes_per_value,
    per_node_flops,
)
from .onnx_utils import (
    backfill_quant_shapes,
    build_producers_consumers,
    topo_sort,
    value_info_map,
    apply_dim_param_overrides,
    make_llm_symbolic_dim_overrides,
    llm_preset_to_lengths,
)
from .units import (
    BANDWIDTH_MULT,
    FLOP_UNITS,
    UNIT_MULT,
    bandwidth_to_bytes_per_s,
)


def _pick_unit_auto(total_flops: float) -> str:
    if total_flops >= 1e12:
        return "TFLOP"
    if total_flops >= 1e9:
        return "GFLOP"
    if total_flops >= 1e6:
        return "MFLOP"
    if total_flops >= 1e3:
        return "KFLOP"
    return "FLOP"


def _scale(value: float, unit_map: dict, unit: str) -> float:
    return float(value) / float(unit_map[unit])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Analyse ONNX models and suggest split points.")
    ap.add_argument("onnx_model", help="Path to ONNX model")

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min-gap", type=int, default=2)
    ap.add_argument("--min-compute-pct", type=float, default=0.0)

    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--assume-activation-bytes", type=int, default=None)

    ap.add_argument("--llm-preset", type=str, default="none",
                    help="Apply decoder-style LLM symbolic dim overrides (KV-cache models). \n                         Use one of: standard, latency, throughput, rag, custom, none.")
    ap.add_argument("--llm-mode", type=str, default="decode", choices=["decode", "prefill"],
                    help="Which scenario to apply: decode uses sequence_length=1 & past_sequence_length=decode_past_len; \n                         prefill uses sequence_length=prefill_len & past_sequence_length=0.")
    ap.add_argument("--llm-prefill-len", type=int, default=512, help="Prefill sequence length (tokens)")
    ap.add_argument("--llm-decode-past-len", type=int, default=2048, help="Decode past sequence length (KV cache length)")

    ap.add_argument("--exclude-trivial", action="store_true")
    ap.add_argument("--only-single-tensor", action="store_true")

    ap.add_argument("--rank", type=str, default="cut", choices=["cut", "score", "latency"])
    ap.add_argument("--log-comm", action="store_true")
    ap.add_argument("--w-comm", type=float, default=1.0)
    ap.add_argument("--w-imb", type=float, default=3.0)
    ap.add_argument("--w-tensors", type=float, default=0.2)

    ap.add_argument("--bw", type=float, default=None, help="Link bandwidth value")
    ap.add_argument("--bw-unit", type=str, default="MB/s", choices=sorted(BANDWIDTH_MULT.keys()))
    ap.add_argument("--gops-left", type=float, default=None)
    ap.add_argument("--gops-right", type=float, default=None)
    ap.add_argument("--overhead-ms", type=float, default=0.0)

    ap.add_argument("--unit", type=str, default="MiB", choices=sorted(UNIT_MULT.keys()))
    ap.add_argument("--precision", type=int, default=3)

    ap.add_argument("--tex-out", type=str, default=None, help="Write a LaTeX table of the top-k picks")

    ap.add_argument(
        "--load-external-data",
        action="store_true",
        help=(
            "Load external tensor data (weights) into memory when parsing the model. "
            "This is NOT required for split analysis and can consume multiple GB for large models."
        ),
    )

    args = ap.parse_args(argv)

    # For analysis we only need shapes/metadata; avoid pulling large external weights into RAM.
    try:
        model = onnx.load(args.onnx_model, load_external_data=bool(args.load_external_data))

        # Optional LLM symbolic dim preset overrides (helps with KV-cache decoder models)
        llm_key = str(getattr(args, 'llm_preset', 'none')).strip().lower()
        if llm_key and llm_key not in ('none', 'off', 'false', '0'):
            prefill_len = int(getattr(args, 'llm_prefill_len', 512))
            decode_past_len = int(getattr(args, 'llm_decode_past_len', 2048))
            p_pref, p_dec = llm_preset_to_lengths(llm_key)
            if p_pref is not None and p_dec is not None and llm_key != 'custom':
                prefill_len, decode_past_len = int(p_pref), int(p_dec)
            batch = int(args.batch) if args.batch is not None else 1
            llm_mode = str(getattr(args, 'llm_mode', 'decode'))
            overrides = make_llm_symbolic_dim_overrides(
                model,
                batch=batch,
                prefill_len=prefill_len,
                decode_past_len=decode_past_len,
                mode=llm_mode,
            )
            if overrides:
                apply_dim_param_overrides(model, overrides, only_inputs=True)
                k_preview = ', '.join(list(overrides.keys())[:6])
                suffix = '...' if len(overrides) > 6 else ''
                print(f'[llm] applied {len(overrides)} symbolic dim overrides ({llm_mode}): {k_preview}{suffix}')
            else:
                print('[llm] preset requested but no matching dim_param names were found; skipping')

    except TypeError:
        # Older onnx versions: parse protobuf directly.
        with open(args.onnx_model, "rb") as f:
            model = onnx.ModelProto.FromString(f.read())
    model = shape_inference.infer_shapes(model)

    vimap = value_info_map(model)
    backfill_quant_shapes(model, vimap, batch_override=args.batch)

    nodes, producer_of, consumers_of = build_producers_consumers(model)
    order = topo_sort(nodes, producer_of)

    value_bytes = compute_tensor_bytes_per_value(vimap, batch_override=args.batch, assume_activation_bytes=args.assume_activation_bytes)
    costs, val_span = boundary_costs(order, producer_of, consumers_of, value_bytes)

    if not costs:
        print("No internal boundaries.")
        return 0

    crossing_counts = boundary_tensor_counts(order, producer_of, consumers_of, {v: 1 for v in producer_of})

    node_flops_list = per_node_flops(model, vimap, batch_override=args.batch)
    flops_by_node = {idx: fl for (idx, _, __, fl) in node_flops_list}
    total_flops = float(sum(flops_by_node.values()))
    flops_left_prefix = compute_boundary_flops_prefix(order, flops_by_node)

    candidates = list(range(len(costs)))

    if args.exclude_trivial:
        TRIVIAL = {"Relu", "Reshape", "BatchNormalization", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Identity"}

        def drop(b: int) -> bool:
            return nodes[order[b]].op_type in TRIVIAL or nodes[order[b + 1]].op_type in TRIVIAL

        candidates = [b for b in candidates if not drop(b)]

    if args.only_single_tensor:
        candidates = [b for b in candidates if int(crossing_counts[b]) == 1]

    if args.min_compute_pct > 0 and total_flops > 0:
        thr = (args.min_compute_pct / 100.0) * total_flops
        candidates = [
            b
            for b in candidates
            if float(flops_left_prefix[b]) >= thr and float(total_flops - flops_left_prefix[b]) >= thr
        ]

    scores = None
    latency = None

    if args.rank == "cut":
        candidates.sort(key=lambda b: float(costs[b]))

    elif args.rank == "score":
        scores = compute_scores_for_candidates(
            candidates,
            costs,
            crossing_counts,
            flops_left_prefix,
            total_flops,
            w_comm=args.w_comm,
            w_imb=args.w_imb,
            w_tensors=args.w_tensors,
            linear_comm=not args.log_comm,
        )
        candidates.sort(key=lambda b: float(scores.get(b, 0.0)))

    elif args.rank == "latency":
        bw_bps = bandwidth_to_bytes_per_s(args.bw, args.bw_unit)
        if bw_bps is None or args.gops_left is None or args.gops_right is None:
            candidates.sort(key=lambda b: float(costs[b]))
        else:
            gl = float(args.gops_left)
            gr = float(args.gops_right)
            latency = {
                b: (
                    1000.0
                    * (
                        float(flops_left_prefix[b]) / (gl * 1e9)
                        + float(costs[b]) / float(bw_bps)
                        + float(total_flops - flops_left_prefix[b]) / (gr * 1e9)
                    )
                    + float(args.overhead_ms)
                )
                for b in candidates
            }
            candidates.sort(key=lambda b: float(latency.get(b, float("inf"))))

    # non-maximum suppression by boundary index
    picks: List[int] = []
    for b in candidates:
        if all(abs(b - s) > int(args.min_gap) for s in picks):
            picks.append(b)
        if len(picks) >= int(args.topk):
            break

    mul = float(UNIT_MULT[args.unit])

    flop_unit = _pick_unit_auto(total_flops)

    print(f"\nTotal compute: {_scale(total_flops, FLOP_UNITS, flop_unit):.3f} {flop_unit}s")
    print(f"Top {len(picks)} suggestions (rank={args.rank}, min-gap={args.min_gap}, unit={args.unit}):\n")

    for r, b in enumerate(picks, 1):
        lidx, ridx = order[b], order[b + 1]
        fl_l = float(flops_left_prefix[b])
        fl_r = float(total_flops - flops_left_prefix[b])

        extra = ""
        if latency is not None and b in latency:
            extra = f"  latency={latency[b]:.2f} ms"
        elif scores is not None and b in scores:
            extra = f"  score={scores[b]:.3f}"

        print(
            f"{r:>2} boundary {b:>4}  {nodes[lidx].op_type:16s} -> {nodes[ridx].op_type:16s}  "
            f"comm={float(costs[b]) / mul:.{args.precision}f} {args.unit}  "
            f"#tensors={int(crossing_counts[b])}  "
            f"F_L={fl_l/1e9:.3f} GFLOPs  F_R={fl_r/1e9:.3f} GFLOPs{extra}"
        )

    if args.tex_out:
        # Simple LaTeX table export for top-k picks
        import re

        def _label_safe(s: str) -> str:
            s = re.sub(r"[^A-Za-z0-9]+", "_", s)
            return s.strip("_").lower()

        def _escape(s: str) -> str:
            return s.replace("_", "\\_")

        def _semantic_group_for_node(n: onnx.NodeProto) -> str:
            """Return a stable semantic group identifier for a node.

            We try to infer groups such as 'layers.20' or 'blocks.5' based on
            node names / value names. This is mainly for Transformer/LLM graphs,
            but is safe as a best-effort for other models.
            """

            hay = (n.name or "").strip()
            if not hay:
                # Node names are often empty; value names are usually more informative.
                if n.output:
                    hay = str(n.output[0])
                elif n.input:
                    hay = str(n.input[0])

            patterns = [
                (r"/layers\.(\d+)\b", lambda m: f"layers.{m.group(1)}"),
                (r"/layer\.(\d+)\b", lambda m: f"layer.{m.group(1)}"),
                (r"/blocks\.(\d+)\b", lambda m: f"blocks.{m.group(1)}"),
                (r"/block\.(\d+)\b", lambda m: f"block.{m.group(1)}"),
                (r"/encoder/layer\.(\d+)\b", lambda m: f"enc_layer.{m.group(1)}"),
                (r"/decoder/layer\.(\d+)\b", lambda m: f"dec_layer.{m.group(1)}"),
                (r"/backbone/stage(\d+)\b", lambda m: f"backbone_stage{m.group(1)}"),
                (r"/neck/stage(\d+)\b", lambda m: f"neck_stage{m.group(1)}"),
                (r"/head/stage(\d+)\b", lambda m: f"head_stage{m.group(1)}"),
                (r"/stage(\d+)\b", lambda m: f"stage{m.group(1)}"),
            ]

            for pat, fn in patterns:
                m = re.search(pat, hay)
                if m:
                    try:
                        return fn(m)
                    except Exception:
                        return "other"

            hay_l = hay.lower()
            if "/stem/" in hay_l:
                return "stem"
            if "/embed" in hay_l or "/embedding" in hay_l:
                return "embed"
            return "other"

        def _semantic_labels_for_boundaries() -> List[str]:
            M_local = max(0, len(order) - 1)
            out: List[str] = [""] * M_local
            for b in range(M_local):
                left = nodes[order[b]]
                right = nodes[order[b + 1]]
                gl = _semantic_group_for_node(left)
                gr = _semantic_group_for_node(right)
                out[b] = gl if gl == gr else f"{gl}->{gr}"
            return out

        sem_labels = _semantic_labels_for_boundaries()

        model_name = _escape(args.onnx_model.split("/")[-1].split("\\")[-1])
        label = _label_safe(model_name)

        lines2 = []
        lines2.append("% Requires \\usepackage{booktabs}\n")
        lines2.append("\\begin{table}[t]\n")
        lines2.append("  \\centering\n")
        lines2.append("  \\small\n")
        lines2.append("  \\setlength{\\tabcolsep}{4pt}\n")
        lines2.append("  \\begin{tabular}{@{}r l r r r r@{}}\n")
        lines2.append("    \\toprule\n")
        lines2.append("    Boundary & Semantic & Comm (MiB) & \\#Tensors & $F_L$ (GFLOP) & $F_R$ (GFLOP) \\\\ \n")
        lines2.append("    \\midrule\n")
        for b in picks:
            comm_mib = float(costs[b]) / (1024.0**2)
            fl_l = float(flops_left_prefix[b]) / 1e9
            fl_r = float(total_flops - flops_left_prefix[b]) / 1e9
            sem = _escape(str(sem_labels[b]) if b < len(sem_labels) else "")
            lines2.append(
                f"    {b} & {sem} & {comm_mib:.3f} & {int(crossing_counts[b])} & {fl_l:.3f} & {fl_r:.3f} \\\\ \n"
            )
        lines2.append("    \\bottomrule\n")
        lines2.append("  \\end{tabular}\n")
        lines2.append(f"  \\caption{{Example split candidates for {model_name} (illustrative).}}\n")
        lines2.append(f"  \\label{{tab:split_candidates_{label}}}\n")
        lines2.append("\\end{table}\n")

        with open(args.tex_out, "w", encoding="utf-8") as f:
            f.writelines(lines2)
        print(f"\nWrote LaTeX table to: {args.tex_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
