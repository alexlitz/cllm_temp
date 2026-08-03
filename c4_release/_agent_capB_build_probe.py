#!/usr/bin/env python3
"""CAPSTONE PHASE B — build-feasibility probe of the CFM Doom model.

Builds the c4_min pure-forward VM with the Doom flag stack ON (C4_PC_WIDE,
C4_IMM_NIBS=6, C4_SHIFT32, C4_CMP32, C4_DIVMOD_SIGNED, C4_PF_CFM,
C4_CODE_ADDR_BITS=20) and reports the layout/dims/head_dim + whether the
code-CAM head_dim budget (CODE_ADDR_BITS+4+1+IMM_NIBS) fits, WITHOUT loading
the 552K code frames yet (the residual dim is code_size-INDEPENDENT under CFM).
"""
from __future__ import annotations
import os, sys, time, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402


def main():
    from c4_min.compact_alloc import build_compact_sparse_streaming
    from c4_min.nibble_pure_forward_complete import (
        CODE_ADDR_BITS, IMM_NIBS, _pf_cfm_enabled)
    print(f"flags: CFM={_pf_cfm_enabled()} CODE_ADDR_BITS={CODE_ADDR_BITS} "
          f"IMM_NIBS={IMM_NIBS} PC_WIDE={os.environ.get('C4_PC_WIDE')} "
          f"SHIFT32={os.environ.get('C4_SHIFT32')} CMP32={os.environ.get('C4_CMP32')} "
          f"DIVMOD_SIGNED={os.environ.get('C4_DIVMOD_SIGNED')}")
    # code_size is a build-time validation ceiling; under CFM the residual is
    # code_size-INDEPENDENT so we can pass a SMALL code_size and still get the
    # true Doom dims (the code lives in KV, not baked bands).  Use 64 for the
    # cheap layout, then report that the SAME dims hold for the 552K program.
    t0 = time.time()
    model, L, stats = build_compact_sparse_streaming(
        code_size=64, recurrent_divmod=True, compute_mode="dense_kernel")
    build_s = time.time() - t0

    d_model = model.blocks[0].ffn.W_up.shape[1] if hasattr(
        model.blocks[0].ffn.W_up, "shape") else None
    # robust d_model: read the embed
    d_model = model.embed.shape[1]
    n_blocks = len(model.blocks)
    # head_dim of the code-select block
    names = list(getattr(L, "_block_names", []))
    hd_info = {}
    csel_bi = names.index("code-select") if "code-select" in names else None
    if csel_bi is not None and csel_bi < len(model.blocks):
        attn = getattr(model.blocks[csel_bi], "attn", None)
        if attn is not None:
            hd_info = {"block": csel_bi, "n_heads": attn.n_heads,
                       "head_dim": attn.head_dim,
                       "budget": CODE_ADDR_BITS + 4 + 1 + IMM_NIBS}
    n_params = 0
    for blk in model.blocks:
        for comp in (getattr(blk, "attn", None), getattr(blk, "ffn", None)):
            if comp is None:
                continue
            for wn in ("W_q", "W_k", "W_v", "W_o", "W_up", "W_gate", "W_down"):
                w = getattr(comp, wn, None)
                if w is None:
                    continue
                d = None
                for attr in ("dense", "dense_resident"):
                    if getattr(w, attr, None) is not None:
                        d = getattr(w, attr); break
                if d is None and getattr(w, "csr", None) is not None:
                    d = w.csr
                if d is not None and hasattr(d, "numel"):
                    n_params += d.numel() if hasattr(d, "numel") else 0

    out = {
        "d_model": int(d_model),
        "n_blocks": int(n_blocks),
        "D_used": int(getattr(L, "dim", d_model)),
        "code_select": hd_info,
        "block_names": names,
        "build_s": round(build_s, 1),
        "cfm": _pf_cfm_enabled(),
        "code_addr_bits": CODE_ADDR_BITS,
        "imm_nibs": IMM_NIBS,
    }
    print("BUILD " + json.dumps(out))
    print(f"\nd_model={d_model}  n_blocks={n_blocks}  "
          f"code-select head_dim={hd_info.get('head_dim')} "
          f"(budget {hd_info.get('budget')})  "
          f"fits={hd_info.get('head_dim',0) >= hd_info.get('budget',1e9)}")
    print("block_names:", names)


if __name__ == "__main__":
    main()
