"""ONNX graph parsing helpers.

Includes:
- dtype/shape helpers
- value_info map extraction
- producer/consumer maps
- topological sort
- quantized-shape backfilling
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set, Iterable

import onnx
from onnx import TensorProto, helper, shape_inference


def dtype_nbytes(elem_type: int, default: int = 4) -> int:
    """Return bytes per element for ONNX TensorProto elem_type."""
    T = TensorProto
    mapping = {
        T.FLOAT: 4,
        T.FLOAT16: 2,
        T.BFLOAT16: 2,
        T.DOUBLE: 8,
        T.UINT8: 1,
        T.INT8: 1,
        T.UINT16: 2,
        T.INT16: 2,
        T.UINT32: 4,
        T.INT32: 4,
        T.UINT64: 8,
        T.INT64: 8,
        T.BOOL: 1,
    }
    return int(mapping.get(int(elem_type), default))


# ---------------------------- ValueInfo helpers ----------------------------

def shape_from_vi(vi) -> Optional[List[Optional[int]]]:
    if vi is None or not vi.type.HasField("tensor_type"):
        return None
    shp: List[Optional[int]] = []
    for d in vi.type.tensor_type.shape.dim:
        shp.append(int(d.dim_value) if d.HasField("dim_value") else None)
    return shp


def elemtype_from_vi(vi) -> Optional[int]:
    if vi is None or not vi.type.HasField("tensor_type"):
        return None
    return int(vi.type.tensor_type.elem_type)


def numel_from_shape(shape: Optional[List[Optional[int]]], batch_override: Optional[int] = None) -> int:
    if not shape:
        return 0
    shp = list(shape)
    if batch_override is not None and shp:
        if shp[0] is None or int(shp[0] or 0) == 0:
            shp[0] = int(batch_override)
    n = 1
    for d in shp:
        n *= int(d) if (d is not None and int(d) > 0) else 1
    return int(n)


def value_info_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    """Map tensor name -> ValueInfo.

    Notes
    -----
    ONNX models do not always populate `graph.value_info` for every produced tensor.
    In practice (especially for transformer exports), a significant number of tensors
    are produced by `Constant` nodes and never get a ValueInfo from shape inference.
    
    For our use-cases (activation byte estimates, unknown-shape diagnostics), it is
    still very helpful to at least know dtype + static dims for these constants.
    The constant payloads are usually tiny (shape/control tensors), so this improves
    coverage without impacting memory.
    """

    vis = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    m: Dict[str, onnx.ValueInfoProto] = {vi.name: vi for vi in vis}

    # Backfill missing ValueInfo for Constant node outputs.
    try:
        from onnx import AttributeProto
    except Exception:  # pragma: no cover
        AttributeProto = None  # type: ignore

    if AttributeProto is not None:
        for n in model.graph.node:
            if n.op_type != "Constant" or not n.output:
                continue
            out = n.output[0]

            # If the exporter already emitted a ValueInfo for this Constant, keep it
            # only if the shape is fully known. Otherwise, we overwrite it with a
            # best-effort inferred shape to avoid treating small constants as
            # “unknown size”.
            existing = m.get(out)
            if existing is not None:
                try:
                    sh = shape_from_vi(existing)
                except Exception:
                    sh = None
                if sh is not None and all(d is not None for d in sh):
                    continue

            # Try to infer dtype+shape for Constant outputs.
            #
            # Many exporters use a tensor-valued "value" attribute, but others
            # use scalar/vector attributes (value_int, value_ints, ...). If we
            # don't backfill these shapes, they look "unknown" and can skew the
            # unknown-size penalty/diagnostics.
            inferred = None  # (dtype, shape)

            # 1) TensorProto-valued Constant
            for a in n.attribute:
                if a.name == "value" and a.type == AttributeProto.TENSOR:
                    t = a.t
                    inferred = (t.data_type, list(t.dims))
                    break

            # 2) Scalar / vector Constant fallbacks
            if inferred is None:
                for a in n.attribute:
                    if a.name == "value_int" and a.type == AttributeProto.INT:
                        inferred = (TensorProto.INT64, [])
                        break
                    if a.name == "value_float" and a.type == AttributeProto.FLOAT:
                        inferred = (TensorProto.FLOAT, [])
                        break
                    if a.name == "value_ints" and a.type == AttributeProto.INTS:
                        inferred = (TensorProto.INT64, [len(a.ints)])
                        break
                    if a.name == "value_floats" and a.type == AttributeProto.FLOATS:
                        inferred = (TensorProto.FLOAT, [len(a.floats)])
                        break
                    if a.name == "value_string" and a.type == AttributeProto.STRING:
                        inferred = (TensorProto.STRING, [])
                        break
                    if a.name == "value_strings" and a.type == AttributeProto.STRINGS:
                        inferred = (TensorProto.STRING, [len(a.strings)])
                        break

            if inferred is not None:
                try:
                    dtype, shape = inferred
                    m[out] = helper.make_tensor_value_info(out, dtype, shape)
                except Exception:
                    # If something goes wrong, just leave it unknown.
                    pass

    return m


# ---------------------------- Graph utilities ----------------------------

def build_producers_consumers(model: onnx.ModelProto):
    nodes = list(model.graph.node)
    producer_of: Dict[str, int] = {}
    consumers_of: Dict[str, List[int]] = defaultdict(list)

    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                producer_of[out] = idx

    for idx, node in enumerate(nodes):
        for inp in node.input:
            if inp in producer_of:
                consumers_of[inp].append(idx)

    return nodes, producer_of, consumers_of


def topo_sort(nodes: List[onnx.NodeProto], producer_of: Dict[str, int]) -> List[int]:
    """Kahn topological sort over ONNX node indices."""
    preds: List[set] = [set() for _ in nodes]
    succs: List[set] = [set() for _ in nodes]

    for j, node in enumerate(nodes):
        for inp in node.input:
            if inp in producer_of:
                p = producer_of[inp]
                preds[j].add(p)
                succs[p].add(j)

    indeg = [len(p) for p in preds]
    q = deque([i for i, d in enumerate(indeg) if d == 0])
    order: List[int] = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in list(succs[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # If graph has cycles/unconnected remnants, append remaining nodes
    if len(order) != len(nodes):
        remaining = [i for i in range(len(nodes)) if i not in set(order)]
        order.extend(remaining)

    return order


# ---------------------------- Quant shape backfilling ----------------------------

def _get_attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
        if a.type == onnx.AttributeProto.INT:
            return int(a.i)
        if a.type == onnx.AttributeProto.FLOAT:
            return float(a.f)
        if a.type == onnx.AttributeProto.FLOATS:
            return list(a.floats)
        if a.type == onnx.AttributeProto.STRING:
            try:
                return a.s.decode("utf-8")
            except Exception:
                return str(a.s)
    return default


def _conv_output_hw(h_in: int, w_in: int, k: List[int], s: List[int], p: List[int], d: List[int]) -> Tuple[int, int]:
    kh, kw = (k + [1, 1])[:2]
    sh, sw = (s + [1, 1])[:2]
    dh, dw = (d + [1, 1])[:2]

    if len(p) == 2:
        pt, pl, pb, pr = p[0], p[1], p[0], p[1]
    elif len(p) == 4:
        pt, pl, pb, pr = p
    else:
        pt = pl = pb = pr = 0

    def out_dim(Lin: int, k: int, s: int, p0: int, p1: int, dil: int) -> int:
        return (Lin + p0 + p1 - dil * (k - 1) - 1) // s + 1

    hout = out_dim(int(h_in), int(kh), int(sh), int(pt), int(pb), int(dh))
    wout = out_dim(int(w_in), int(kw), int(sw), int(pl), int(pr), int(dw))
    return max(1, int(hout)), max(1, int(wout))


def _matmul_out_shape(sa: Optional[List[Optional[int]]], sb: Optional[List[Optional[int]]]) -> List[int]:
    """Broadcast-like matmul output shape (approx.)."""

    def norm2(s):
        if not s:
            return [1, 1]
        s2 = [d or 1 for d in s]
        if len(s2) == 1:
            return [1, s2[0]]
        return s2

    A = norm2(sa)
    B = norm2(sb)

    m, k1 = A[-2], A[-1]
    k2, n = B[-2], B[-1]
    _k = min(k1, k2)

    lead_a = A[:-2]
    lead_b = B[:-2]
    L = max(len(lead_a), len(lead_b))
    lead_a = [1] * (L - len(lead_a)) + lead_a
    lead_b = [1] * (L - len(lead_b)) + lead_b
    lead = [max(a, b) for a, b in zip(lead_a, lead_b)]

    return [int(x) for x in (lead + [m, n])]


def backfill_quant_shapes(
    model: onnx.ModelProto,
    vimap: Dict[str, onnx.ValueInfoProto],
    batch_override: Optional[int] = None,
    batch: Optional[int] = None,
) -> None:
    """Backfill ValueInfo entries for common quant operators.

    Accepts both `batch_override` and `batch` for compatibility.
    """
    if batch_override is None and batch is not None:
        batch_override = batch

    # Only need initializer *shapes* for quant backfilling.
    # Avoid materializing large weights into memory (external data).
    init_map = {init.name: list(init.dims) for init in model.graph.initializer}

    def ensure_vi(name: str, shape: List[int], elem_type: int):
        if name in vimap:
            return
        vi = helper.make_tensor_value_info(name, elem_type, list(shape))
        vimap[name] = vi
        model.graph.value_info.extend([vi])

    def shp(name: str) -> Optional[List[Optional[int]]]:
        return shape_from_vi(vimap.get(name))

    for node in model.graph.node:
        op = node.op_type

        if op == "QLinearConv":
            # Spec: inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, B(optional)
            if len(node.input) < 4 or not node.output:
                continue
            x, w, y = node.input[0], node.input[3], node.output[0]
            if y in vimap:
                continue
            xs = shp(x)
            W_dims = init_map.get(w)
            if not xs or len(xs) != 4 or not W_dims or len(W_dims) != 4:
                continue
            N, C, H, Wd = [batch_override if (i == 0 and batch_override) else int(xs[i] or 1) for i in range(4)]
            Cout, _, kH, kW = [int(d or 1) for d in W_dims]
            Ho, Wo = _conv_output_hw(
                H,
                Wd,
                [int(kH), int(kW)],
                _get_attr(node, "strides", [1, 1]) or [1, 1],
                _get_attr(node, "pads", [0, 0, 0, 0]) or [0, 0, 0, 0],
                _get_attr(node, "dilations", [1, 1]) or [1, 1],
            )
            ensure_vi(y, [int(N), int(Cout), int(Ho), int(Wo)], TensorProto.UINT8)

        elif op == "QLinearAdd":
            # Typical: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
            if len(node.input) < 4 or not node.output:
                continue
            a, b, y = node.input[0], node.input[3], node.output[0]
            if y in vimap:
                continue
            sa, sb = shp(a), shp(b)
            if not sa and not sb:
                continue
            A = [int(d or 1) for d in (sa or [])]
            B = [int(d or 1) for d in (sb or [])]
            L = max(len(A), len(B))
            A = [1] * (L - len(A)) + A
            B = [1] * (L - len(B)) + B
            out = [max(x, y2) for x, y2 in zip(A, B)]
            ensure_vi(y, out, TensorProto.UINT8)

        elif op == "QLinearGlobalAveragePool":
            if not node.input or not node.output:
                continue
            x, y = node.input[0], node.output[0]
            if y in vimap:
                continue
            xs = shp(x)
            if not xs or len(xs) != 4:
                continue
            N, C, _, _ = [batch_override if (i == 0 and batch_override) else int(xs[i] or 1) for i in range(4)]
            ensure_vi(y, [int(N), int(C), 1, 1], TensorProto.UINT8)

        elif op == "QLinearMatMul":
            # Spec: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
            if len(node.input) < 4 or not node.output:
                continue
            a, b, y = node.input[0], node.input[3], node.output[0]
            if y in vimap:
                continue
            sa, sb = shp(a), shp(b)
            if not sa and not sb:
                continue
            out = _matmul_out_shape(sa, sb)
            ensure_vi(y, out, TensorProto.UINT8)

        elif op == "QuantizeLinear":
            if not node.input or not node.output:
                continue
            x, y = node.input[0], node.output[0]
            if y in vimap:
                continue
            sa = shp(x)
            if sa:
                ensure_vi(y, [int(d or 1) for d in sa], TensorProto.UINT8)

        elif op == "DequantizeLinear":
            if not node.input or not node.output:
                continue
            x, y = node.input[0], node.output[0]
            if y in vimap:
                continue
            sa = shp(x)
            if sa:
                ensure_vi(y, [int(d or 1) for d in sa], TensorProto.FLOAT)

# ---------------- LLM / symbolic shape helpers ----------------

def collect_dim_params(model: ModelProto) -> Set[str]:
    """Collect all dim_param strings used in graph inputs/outputs/value_info."""
    params: Set[str] = set()

    def _scan_vi(vi: ValueInfoProto) -> None:
        try:
            shp = vi.type.tensor_type.shape
        except Exception:
            return
        for d in shp.dim:
            if getattr(d, "dim_param", ""):
                params.add(d.dim_param)

    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        _scan_vi(vi)

    return params


def apply_dim_param_overrides(
    model: ModelProto,
    overrides: Dict[str, int],
    *,
    only_inputs: bool = True,
    clear_dim_param: bool = True,
) -> List[Tuple[str, str, Optional[int], int]]:
    """Apply dim_param -> dim_value overrides in-place.

    This is useful for LLM graphs where shapes use symbolic names like
    'batch_size', 'sequence_length', 'past_sequence_length', etc.

    Returns a list of applied changes as tuples:
        (tensor_name, dim_param, old_dim_value, new_dim_value)
    """
    changes: List[Tuple[str, str, Optional[int], int]] = []

    def _apply_to_vi(vi: ValueInfoProto) -> None:
        try:
            shp = vi.type.tensor_type.shape
        except Exception:
            return
        for d in shp.dim:
            dp = getattr(d, "dim_param", "")
            if not dp:
                continue
            if dp not in overrides:
                continue
            new_v = int(overrides[dp])
            old_v: Optional[int] = None
            if d.HasField("dim_value"):
                old_v = int(d.dim_value)
            d.dim_value = new_v
            if clear_dim_param:
                d.dim_param = ""
            changes.append((vi.name, dp, old_v, new_v))

    if only_inputs:
        for vi in list(model.graph.input):
            _apply_to_vi(vi)
    else:
        for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
            _apply_to_vi(vi)

    return changes


def make_llm_dim_overrides(
    model: ModelProto,
    *,
    batch: int,
    seq_len: int,
    past_len: int,
    total_len: int,
) -> Dict[str, int]:
    """Heuristically map dim_param strings to concrete values for LLM graphs.

    We *don't* assume a fixed exporter; instead we look at existing dim_param
    strings and try to infer which ones represent batch/seq/past/total.

    The returned dict maps the *actual* dim_param strings present in the model
    to integer values.
    """
    params = collect_dim_params(model)
    out: Dict[str, int] = {}

    for p in params:
        pl = p.lower()

        # Batch-like
        if "batch" in pl or pl in {"b", "bs", "batch_size"}:
            out[p] = int(batch)
            continue

        # Past / KV cache
        if (
            "past" in pl
            or "kv" in pl
            or "cache" in pl
            or "history" in pl
            or "prev" in pl
        ):
            out[p] = int(past_len)
            continue

        # Total/context length (often used by attention mask)
        if (
            "total" in pl
            or "context" in pl
            or "attn" in pl
            or "attention" in pl
            or "mask" in pl
        ) and ("head" not in pl) and ("hidden" not in pl):
            out[p] = int(total_len)
            continue

        # Sequence length
        if (
            "seq" in pl
            or "sequence" in pl
            or (pl.endswith("_len") and ("head" not in pl) and ("hidden" not in pl))
        ):
            out[p] = int(seq_len)
            continue

    # ---------------------------------------------------------------------
    # Extra seeding: some exporters (incl. ORT's symbolic shape infer) emit
    # very generic dim_param names like "unk__0" / "dim_0". The heuristics
    # above can't classify those from the name alone.
    #
    # In that case, we can still map a good portion of dims by looking at
    # *where* they appear (standard LLM input/output names).
    # ---------------------------------------------------------------------
    try:
        vimap = value_info_map(model)

        def _seed_dim(value_name: str, axis: int, value: int) -> None:
            vi = vimap.get(value_name)
            if vi is None:
                return
            try:
                dims = vi.type.tensor_type.shape.dim
            except Exception:
                return
            if not dims:
                return
            if axis < 0:
                axis2 = len(dims) + axis
            else:
                axis2 = axis
            if axis2 < 0 or axis2 >= len(dims):
                return
            d = dims[axis2]
            if getattr(d, "dim_param", "") and d.dim_param not in out:
                out[d.dim_param] = int(value)

        # Common text model inputs.
        _seed_dim("input_ids", 0, batch)
        _seed_dim("input_ids", 1, seq_len)
        _seed_dim("position_ids", 0, batch)
        _seed_dim("position_ids", 1, seq_len)
        _seed_dim("attention_mask", 0, batch)
        _seed_dim("attention_mask", -1, total_len)
        if total_len != seq_len:
            _seed_dim("attention_mask", -2, seq_len)

        # KV cache inputs (past).
        for inp in model.graph.input:
            n = inp.name
            if not (n.startswith("past_key_values.") and (n.endswith(".key") or n.endswith(".value"))):
                continue
            _seed_dim(n, 0, batch)
            vi = vimap.get(n)
            if vi is None:
                continue
            try:
                dims = vi.type.tensor_type.shape.dim
            except Exception:
                continue
            rank = len(dims)
            if rank < 2:
                continue

            # Prefer the "middle" axis for past length (often axis 2 in [B, H, T, D]).
            cand_axes = [i for i in range(rank) if i not in (0, rank - 1)]
            pref_order = []
            if rank >= 3:
                pref_order.append(2)
                pref_order.append(1)
                pref_order.append(rank - 2)
            pref_order += cand_axes
            past_axis = None
            for ax in pref_order:
                if ax < 0 or ax >= rank:
                    continue
                dp = getattr(dims[ax], "dim_param", "")
                if dp and dp not in out:
                    past_axis = ax
                    break
            if past_axis is not None:
                out[dims[past_axis].dim_param] = int(past_len)

        # KV cache outputs (present): length is usually total_len (past+seq).
        for out_vi in model.graph.output:
            n = out_vi.name
            if not (n.startswith("present.") and (n.endswith(".key") or n.endswith(".value"))):
                continue
            _seed_dim(n, 0, batch)
            vi = vimap.get(n)
            if vi is None:
                continue
            try:
                dims = vi.type.tensor_type.shape.dim
            except Exception:
                continue
            rank = len(dims)
            if rank < 2:
                continue

            cand_axes = [i for i in range(rank) if i not in (0, rank - 1)]
            pref_order = []
            if rank >= 3:
                pref_order.append(2)
                pref_order.append(1)
                pref_order.append(rank - 2)
            pref_order += cand_axes
            t_axis = None
            for ax in pref_order:
                if ax < 0 or ax >= rank:
                    continue
                dp = getattr(dims[ax], "dim_param", "")
                if dp and dp not in out:
                    t_axis = ax
                    break
            if t_axis is not None:
                out[dims[t_axis].dim_param] = int(total_len)

        # Logits output often carries (batch, seq_len, vocab). Map the first two.
        _seed_dim("logits", 0, batch)
        _seed_dim("logits", 1, seq_len)

    except Exception:
        # Seeding is best-effort; ignore failures.
        pass

    return out


# Backwards-compatible alias (public API expects this name)
def make_llm_symbolic_dim_overrides(
    model: ModelProto,
    *,
    # Generic / explicit form
    batch: int = 1,
    seq_len: Optional[int] = None,
    past_len: Optional[int] = None,
    total_len: Optional[int] = None,
    # Convenience form used by the GUI presets
    prefill_len: Optional[int] = None,
    decode_past_len: Optional[int] = None,
    # Back-compat alias (older experiments)
    decode_past: Optional[int] = None,
    mode: str = "decode",
) -> Dict[str, int]:
    """Build a mapping from symbolic LLM dimensions to integers.

    This helper is used to *override symbolic dims* (``dim_param``) for LLM graphs
    that expose KV-cache inputs/outputs.

    Two ways to call it:

    1) Explicit (engineer mode): ``seq_len``, ``past_len``, ``total_len``.
    2) Preset-friendly (GUI): ``prefill_len`` + ``decode_past_len`` and ``mode``:
       - ``mode='prefill'`` -> ``seq_len=prefill_len``, ``past_len=0``
       - ``mode='decode'``  -> ``seq_len=1``, ``past_len=decode_past_len``

    If ``total_len`` is omitted, it defaults to ``seq_len + past_len``.
    """

    # Resolve alias
    if decode_past_len is None and decode_past is not None:
        decode_past_len = decode_past

    m = (mode or "").strip().lower()

    # If preset values are provided, compute explicit lengths from them.
    if (prefill_len is not None) or (decode_past_len is not None):
        if m in {"prefill", "prompt"}:
            seq_len = int(prefill_len if prefill_len is not None else 0)
            past_len = 0
            total_len = int(seq_len)
        else:  # decode (default)
            seq_len = 1
            past_len = int(decode_past_len if decode_past_len is not None else 0)
            total_len = int(seq_len + past_len)

    # Fall back to explicit lengths (or safe defaults).
    if seq_len is None:
        seq_len = 1
    if past_len is None:
        past_len = 0
    if total_len is None:
        total_len = int(seq_len + past_len)

    return make_llm_dim_overrides(
        model,
        batch=int(batch),
        seq_len=int(seq_len),
        past_len=int(past_len),
        total_len=int(total_len),
    )


def apply_llm_io_shape_overrides(
    model,
    *,
    batch: int,
    seq_len: int,
    past_len: int,
    total_len: int,
) -> None:
    """Best-effort: write concrete dim_value into common LLM I/O ValueInfo.

    This is intentionally conservative:
    - We only touch graph inputs/outputs (not internal value_info).
    - We only set dim_value when it is currently unset.

    Why we need this:
    - Many transformer ONNX graphs keep `dim_param` empty and leave dims
      unknown, which makes ONNX shape inference unable to propagate shapes.
    - By forcing concrete input sizes (decode/prefill presets), we improve
      downstream shape inference and reduce "unknown size" noise in comm
      estimation.
    """

    def _set_if_unset(dim, value: int) -> None:
        # Do not override fixed dims.
        if dim is None:
            return
        try:
            if dim.HasField("dim_value"):
                return
        except Exception:
            # proto2 HasField should exist, but be safe.
            if getattr(dim, "dim_value", None):
                return
        dim.dim_value = int(value)

    def _patch_vi(vi, *, is_output: bool) -> None:
        try:
            tt = vi.type.tensor_type
            if not tt.HasField("shape"):
                return
            dims = tt.shape.dim
        except Exception:
            return

        if not dims:
            return

        name = getattr(vi, "name", "") or ""
        lname = name.lower()
        rank = len(dims)

        # Batch is almost always dim0.
        _set_if_unset(dims[0], batch)

        # Token ids / positions
        if lname == "input_ids" or "input_ids" in lname or lname.endswith("/input_ids"):
            if rank >= 2:
                _set_if_unset(dims[1], seq_len)
            return

        if lname == "position_ids" or "position_ids" in lname:
            if rank >= 2:
                _set_if_unset(dims[1], seq_len)
            return

        # Logits
        if lname == "logits" or lname.endswith("/logits"):
            # Typical: [batch, seq, vocab]
            if rank >= 2:
                _set_if_unset(dims[1], seq_len)
            return

        # Attention mask (shapes vary across exporters)
        if "attention_mask" in lname or "attn_mask" in lname:
            if rank >= 1:
                _set_if_unset(dims[-1], total_len)
            if rank >= 2:
                # Often [B, 1, Q, K] or [B, Q, K]
                _set_if_unset(dims[-2], seq_len)
            if rank >= 4:
                # Common: [B, 1, Q, K]
                _set_if_unset(dims[1], 1)
            return

        # KV-cache inputs/outputs.
        #  - past_key_values.*: input caches (length = past_len)
        #  - present.*: output caches (length = total_len)
        is_past = ("past_key_values" in lname) or lname.startswith("past_key_values.")
        is_present = lname.startswith("present.")
        if is_past or is_present:
            cache_len = total_len if is_present else past_len

            # Heuristic for cache axis:
            #  - rank 4 common: [B, H, T, D] (T at axis2)
            #  - for other layouts, choose the sole remaining unknown dim
            #    (excluding batch + last dim) if possible.
            if rank >= 3:
                unknown_axes = []
                for i, d in enumerate(dims):
                    if i == 0:
                        continue
                    # avoid overriding head_dim (often last dim)
                    if i == rank - 1:
                        continue
                    try:
                        if d.HasField("dim_value"):
                            continue
                    except Exception:
                        if getattr(d, "dim_value", None):
                            continue
                    # if dim_param exists, let symbolic override logic handle it
                    if getattr(d, "dim_param", ""):
                        continue
                    unknown_axes.append(i)

                if len(unknown_axes) == 1:
                    _set_if_unset(dims[unknown_axes[0]], cache_len)
                elif rank >= 4 and 2 in unknown_axes:
                    _set_if_unset(dims[2], cache_len)
                elif rank >= 4 and 1 in unknown_axes:
                    _set_if_unset(dims[1], cache_len)
                elif unknown_axes:
                    _set_if_unset(dims[unknown_axes[0]], cache_len)

            return

    # Inputs
    for vi in getattr(model.graph, "input", []):
        _patch_vi(vi, is_output=False)

    # Outputs
    for vi in getattr(model.graph, "output", []):
        _patch_vi(vi, is_output=True)


def llm_preset_to_lengths(preset: str) -> Tuple[int, int]:
    """Return (prefill_len, decode_past_len) for known presets."""
    p = (preset or "").strip().lower()
    if p in {"latency", "latency_critical", "chat"}:
        return (128, 512)
    if p in {"standard", "std", "default"}:
        return (512, 2048)
    if p in {"throughput", "rag", "throughput_rag"}:
        return (2048, 128)
    # Fallback
    return (512, 2048)
