"""Candidate pruning heuristics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import onnx

def detect_skip_blocks(
    order: List[int],
    nodes: List[onnx.NodeProto],
    producer_of: Dict[str, int],
    *,
    min_skip_len: int = 8,
    merge_ops: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    """Detect residual/skip-like blocks by scanning merge ops (Add/Concat/... ).

    Returns a list of *merged* blocks. Each block is a dict with:
      - start_pos, end_pos (node positions in topo order)
      - members: list of detected skip inputs contributing to this merged span

    A detected skip input is an input tensor to a merge op whose producer is
    at least min_skip_len nodes earlier in topological order.

    The intended use is candidate pruning: do not split *inside* such blocks
    (i.e., forbid boundary indices b with start_pos <= b < end_pos).
    """

    if merge_ops is None:
        merge_ops = (
            'Add',
            'Concat',
            'Sum',
            'Mul',
        )
    merge_ops = {str(x) for x in merge_ops}

    pos_of = {node_idx: pos for pos, node_idx in enumerate(order)}

    raw: List[Dict[str, object]] = []
    for m_pos, node_idx in enumerate(order):
        n = nodes[node_idx]
        if str(n.op_type) not in merge_ops:
            continue
        for inp in list(n.input):
            if not inp or inp not in producer_of:
                continue
            p_node = int(producer_of[inp])
            if p_node not in pos_of:
                continue
            p_pos = int(pos_of[p_node])
            span = int(m_pos - p_pos)
            if span >= int(min_skip_len):
                raw.append(
                    {
                        'start_pos': int(p_pos),
                        'end_pos': int(m_pos),
                        'merge_pos': int(m_pos),
                        'merge_op': str(n.op_type),
                        'value': str(inp),
                        'span_len': int(span),
                    }
                )

    raw.sort(key=lambda d: (int(d['start_pos']), int(d['end_pos'])))

    merged: List[Dict[str, object]] = []
    for r in raw:
        s = int(r['start_pos'])
        e = int(r['end_pos'])
        if not merged or s > int(merged[-1]['end_pos']):
            merged.append({'start_pos': s, 'end_pos': e, 'members': [r]})
        else:
            merged[-1]['end_pos'] = max(int(merged[-1]['end_pos']), e)
            merged[-1].setdefault('members', [])
            merged[-1]['members'].append(r)

    return merged


def prune_candidates_skip_block(
    candidates: List[int],
    blocks: List[Dict[str, object]],
    *,
    allow_last_n_inside: int = 0,
) -> Tuple[List[int], List[int]]:
    """Prune candidate boundary indices that fall inside detected skip blocks."""

    forbid: set = set()
    for blk in blocks or []:
        s = int(blk.get('start_pos', 0) or 0)
        e = int(blk.get('end_pos', 0) or 0)
        e2 = max(s, int(e) - int(max(0, allow_last_n_inside)))
        for b in range(s, e2):
            forbid.add(int(b))

    kept = [b for b in candidates if int(b) not in forbid]
    return kept, sorted(forbid)

