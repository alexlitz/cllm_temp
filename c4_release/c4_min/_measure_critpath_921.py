"""#921 MEASURE — the TRUE width-free critical path of the log-sink divmod, derived
from the ACTUAL assembled block data-dependencies (which residual band each block
reads vs writes).  Collapse into ONE block any maximal antichain of blocks that write
INDEPENDENT places and read only already-computed places (the "width is free"
reduction).  What remains is the genuine sequential data-dependency chain.

This measures — NOT projects — whether the 127 stored log-sink blocks collapse toward
the menu doc's ~11-block width-free critical path, and how the fp32 chunking constraint
(parallel decompose overflows fp32) changes it.

Golden 174ece66 untouched (off build path).
Run: PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._measure_critpath_921
"""
from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

import torch


def _block_rw(spec) -> Tuple[Set[int], Set[int]]:
    """Residual dims a block READS (nonzero W_up/W_gate cols) and WRITES (nonzero
    W_down rows).  The bias lands on ONE, filtered out as a read (constant)."""
    Wup, Wg, Wd = spec["W_up"], spec["W_gate"], spec["W_down"]
    reads: Set[int] = set()
    for W in (Wup, Wg):
        cols = torch.nonzero(W.abs().sum(0) > 0).flatten().tolist()
        reads.update(cols)
    writes: Set[int] = set(torch.nonzero(Wd.abs().sum(1) > 0).flatten().tolist())
    return reads, writes


def critical_path(specs, one_dim: int) -> Dict[str, object]:
    """Greedy antichain-collapse: process blocks in order; a block joins the CURRENT
    parallel layer iff it neither reads a place written in the current layer nor writes
    a place read/written in the current layer (true independence); else it starts a new
    layer.  This lower-bounds the sequential depth with width free (a block can be
    reordered EARLIER only within its dependency constraints; the greedy same-order
    packing is a conservative UPPER bound on the true min-depth, i.e. the honest
    'at least this shallow' is what a real scheduler achieves)."""
    layers: List[Dict[str, object]] = []
    cur_reads: Set[int] = set()
    cur_writes: Set[int] = set()
    cur_names: List[str] = []
    # track when each place was last written, to enforce RAW across collapsed layers
    for name, spec in specs:
        r, w = _block_rw(spec)
        r.discard(one_dim); w.discard(one_dim)  # ONE is a shared constant, not a dep
        # independence within the current layer:
        #  - block must not READ anything the current layer WRITES (RAW)
        #  - block must not WRITE anything the current layer READS (WAR)
        #  - block must not WRITE anything the current layer WRITES (WAW)
        conflict = bool((r & cur_writes) or (w & cur_reads) or (w & cur_writes))
        if conflict and cur_names:
            layers.append({"names": cur_names, "n": len(cur_names)})
            cur_reads, cur_writes, cur_names = set(), set(), []
        cur_reads |= r
        cur_writes |= w
        cur_names.append(name)
    if cur_names:
        layers.append({"names": cur_names, "n": len(cur_names)})
    return {"depth": len(layers), "layers": layers,
            "stored": sum(len(s) for s in [specs])}


def run(div_logsink: bool):
    import c4_min.qwen_full_vm as Q
    saved = {}
    env = {"C4_LOGSINK_DIV": "1" if div_logsink else "0"}
    if not div_logsink:
        env["C4_DIV_LEAN"] = "1"
    for k, v in env.items():
        saved[k] = os.environ.get(k); os.environ[k] = v
    try:
        subset = Q.SUBSET_FULL
        QL = Q.QwenFullLayout(24, subset, efficient_alu=True, recurrent_divmod=False,
                              code_from_memory=True, shift_via_mul=True,
                              div_logsink=div_logsink)
        L = QL.L
        specs = Q._block_specs(L, 24, subset, efficient_alu=True, recurrent_divmod=False,
                               code_from_memory=True, shift_via_mul=QL.shift_via_mul,
                               div_logsink=QL.div_logsink)
        names = [n for n, _ in specs]
        mux = names.index("alu-ax-mux"); expand = names.index("alu-expand")
        if div_logsink:
            div = [(n, s) for n, s in specs if n.startswith("ls")]
        else:
            aluspan = specs[expand + 1:mux]
            div = [(n, s) for n, s in aluspan
                   if not (n.startswith("mul") or n == "alu-carry")]
        base = [(n, s) for n, s in specs if (n, s) not in div]
        cp_div = critical_path(div, L.ONE)
        cp_base = critical_path(base, L.ONE)
        cp_whole = critical_path(specs, L.ONE)
        return dict(div_logsink=div_logsink,
                    stored_whole=len(specs), stored_divmod=len(div),
                    stored_base=len(base),
                    critpath_divmod=cp_div["depth"],
                    critpath_base=cp_base["depth"],
                    critpath_whole=cp_whole["depth"],
                    divmod_layers=[(l["n"], l["names"][0]) for l in cp_div["layers"]])
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    import json
    for dl in (True, False):
        res = run(dl)
        label = "logsink-fp64" if dl else "radix16-lean-fp32"
        print(f"\n===== {label} =====")
        print(json.dumps({k: v for k, v in res.items() if k != "divmod_layers"}, indent=2))
        print("  divmod collapsed layers (n_blocks_in_layer, first_name):")
        for n, nm in res["divmod_layers"]:
            print(f"    {n:3d}  {nm}")
