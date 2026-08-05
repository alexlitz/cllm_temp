#!/usr/bin/env python3
"""_agent_fused_singleblock.py — tight isolated byte-exactness of the fused-hidden delta
kernel vs the 2-kernel delta path, on ONE synthetic sparse FFN block (no model build).

Builds a random sparse (W_up, W_gate, W_down) block with the c4-typical sparsity, runs
BOTH the 2-kernel FusedUpGateSiluDeltaFFN and the fused-hidden _MegaFFN.run_fused over the
SAME input residual, and reports the exact L-inf.  They must be bit-identical (same CSRs,
same accumulation order; the snapshot is a value-exact copy).

CPU-cheap build, GPU kernels only.  No model load -> memory-safe.
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch


class _FakeFFN:
    def __init__(self, Wu, Wg, Wd, bu, bg, bd):
        self.W_up = Wu; self.W_gate = Wg; self.W_down = Wd
        self.b_up = bu; self.b_gate = bg; self.b_down = bd


def make_block(D=1392, Dff=138, up_nnz_per=3, gt_nnz_per=1, dn_active=3, dn_nnz_per=46,
               seed=0, device="cuda:0"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Wu = torch.zeros(Dff, D)
    Wg = torch.zeros(Dff, D)
    Wd = torch.zeros(D, Dff)
    # W_up / W_gate: a few input cols per hidden row (the FFN reads a small residual band).
    in_band = torch.randperm(D, generator=g)[:60]   # inputs live in a 60-dim band
    for u in range(Dff):
        cols = in_band[torch.randperm(len(in_band), generator=g)[:up_nnz_per]]
        Wu[u, cols] = torch.randn(up_nnz_per, generator=g)
        colsg = in_band[torch.randperm(len(in_band), generator=g)[:gt_nnz_per]]
        Wg[u, colsg] = torch.randn(gt_nnz_per, generator=g)
    # W_down: a few active output rows, each reading many hidden cols.  Make the active
    # rows OVERLAP the input band (the RAW hazard the snapshot must dodge).
    out_rows = in_band[:dn_active]
    for d in out_rows:
        hcols = torch.randperm(Dff, generator=g)[:dn_nnz_per]
        Wd[d, hcols] = torch.randn(dn_nnz_per, generator=g)
    bu = torch.randn(Dff, generator=g) * 0.1
    bg = torch.randn(Dff, generator=g) * 0.1
    bd = torch.zeros(D)
    return _FakeFFN(Wu.to(device), Wg.to(device), Wd.to(device),
                    bu.to(device), bg.to(device), bd.to(device))


def main():
    dev = "cuda:0"
    from .fused_sparse_ffn import FusedUpGateSiluDeltaFFN
    from .fused_megablock import _MegaFFN

    for (Dff, dn_active, dn_nnz_per, seed) in [(138, 3, 46, 0), (502, 82, 12, 1),
                                               (1538, 2, 48, 2), (8, 2, 5, 3),
                                               (256, 8, 58, 4)]:
        ffn = make_block(Dff=Dff, dn_active=dn_active, dn_nnz_per=min(dn_nnz_per, Dff),
                         seed=seed, device=dev)
        D = ffn.W_up.shape[1]
        for K in (512, 2048):
            x = (torch.randn(1, K, D, device=dev) * 2.0)   # bigger magnitude -> real residue

            # 2-kernel reference
            two = FusedUpGateSiluDeltaFFN(ffn, dev, block_k=64)
            out2 = two.forward(x.clone())                  # [1,K,D]

            # fused-hidden path
            mf = _MegaFFN(ffn, dev, block_k=64)
            y = x.clone().reshape(K, D).transpose(0, 1).contiguous()  # [D,K]
            mf.run_fused(y, K, snap_scratch=None)
            outf = y.transpose(0, 1).reshape(1, K, D)

            linf = (out2 - outf).abs().max().item()
            # also vs a dense reference
            hid = torch.nn.functional.silu(x @ ffn.W_up.t() + ffn.b_up) \
                * (x @ ffn.W_gate.t() + ffn.b_gate)
            dense = x + hid @ ffn.W_down.t()
            linf_2_dense = (out2 - dense).abs().max().item()
            linf_f_dense = (outf - dense).abs().max().item()
            print(f"Dff={Dff:5d} n_act={dn_active:3d} K={K:5d}  "
                  f"Linf(fused vs 2kernel)={linf:.3e}  "
                  f"Linf(2k vs dense)={linf_2_dense:.3e}  "
                  f"Linf(fused vs dense)={linf_f_dense:.3e}", flush=True)


if __name__ == "__main__":
    main()
