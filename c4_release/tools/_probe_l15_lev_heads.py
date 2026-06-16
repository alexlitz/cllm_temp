#!/usr/bin/env python3
"""Inspect the BUILT L15 attention block: num_heads + which heads have nonzero
W_q/W_o (i.e. are active). Confirms whether the LEV return_addr heads 8-11 exist.
spec_k=0. Usage: python tools/_probe_l15_lev_heads.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa


def main():
    probe = build_groundtruth_probe()
    blmap = probe.block_layer_map()
    l15_blocks = [e for e in blmap if e["logical"] == 15]
    print("logical-15 physical blocks:", [e["physical"] for e in l15_blocks])
    for e in blmap:
        if e["logical"] == 15:
            phys = e["physical"]
            blk = probe.model.blocks[phys]
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            nh = getattr(attn, "num_heads", None)
            hd = attn.W_q.shape[0] // nh if nh else None
            print(f"phys={phys} logical=15 num_heads={nh} head_dim={hd} "
                  f"W_q.shape={tuple(attn.W_q.shape)} W_o.shape={tuple(attn.W_o.shape)} "
                  f"attn_post_op={e.get('is_post_op_expansion')}")
            if nh is None:
                continue
            def _dense(t):
                t = t.data if hasattr(t, "data") else t
                return t.to_dense() if t.is_sparse or t.layout != torch.strided else t
            Wq = _dense(attn.W_q)
            Wo = _dense(attn.W_o)
            for h in range(nh):
                q_rows = Wq[h*hd:(h+1)*hd, :]
                o_cols = Wo[:, h*hd:(h+1)*hd]
                qn = q_rows.abs().sum().item()
                on = o_cols.abs().sum().item()
                o_out_dims = (o_cols.abs().sum(dim=1) > 1e-9).nonzero().flatten().tolist()
                print(f"  head {h:2d}: |W_q|={qn:12.2f} |W_o|={on:10.4f} "
                      f"O_out_dims(n={len(o_out_dims)})={o_out_dims[:10]}")


if __name__ == "__main__":
    main()
