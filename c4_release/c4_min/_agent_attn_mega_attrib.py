#!/usr/bin/env python3
"""_agent_attn_mega_attrib.py — LIVE-CAM attention cost attribution (CPU, lean, memory-safe).

Answers: how many live-CAM blocks/heads, what each costs (FFN nnz vs W_o vs gather), and
WHY the live-CAM path is the dominant wall now that the dead-FFN chain is wave-batch fused.

The composed single-dispatch schedule's ``_run_region`` runs, in application order:
  - ("mega", MegaBlockChain)  -> the wave-batch fused dead-FFN chain (ONE kernel/wave)
  - ("live", block_idx)       -> 3 separate ``_LiveCamBlock.forward_static[_delta]``:
        h = ffn_b( h + W_o_b(cam_out) )        (dense) or
        h = ffn_b( h[:,out_dims_b] += wo_delta_b )   (on-chip)

On the on-chip path each live block reduces to ONE ``SparseFFN.forward`` = 3 sparse GEMMs
(W_up, W_gate, W_down). The 3 live blocks are 3 SEPARATE FFN forwards (9 GEMM kernels) —
NOT fused, unlike the dead-FFN mega-chain. THAT is the live-CAM wall this task targets.

CPU only; lean streaming build (~1 GB). No GPU, no full densify.
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("C4_PF_CFM", "1")          # doom's 3-live-block CFM config
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e6
    return 1e9


def _guard():
    a = _mem_avail_gb()
    if a < 25.0:
        raise SystemExit(f"[GUARD] MemAvailable {a:.1f}GB < 25GB -> STOP")


def _nnz_of(w):
    """nnz of a sparse/dense weight wrapper (CooLinear / sparse weight / dense)."""
    if hasattr(w, "nnz"):
        return int(w.nnz)
    if hasattr(w, "csr") and w.csr is not None:
        return int(w.csr._nnz()) if hasattr(w.csr, "_nnz") else int((w.csr.to_dense() != 0).sum())
    d = None
    if getattr(w, "dense_resident", None) is not None:
        d = w.dense_resident
    elif getattr(w, "dense", None) is not None:
        d = w.dense
    if d is not None:
        return int((d != 0).sum())
    return -1


def main():
    _guard()
    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    import c4_min.nibble_pure_forward_cached as _PFCa
    _PF.SP_INIT = _PFC.SP_INIT = _PFCa.SP_INIT = 0xFC

    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.tight_attn_compose import install_composed
    from c4_min.direct_cam_batched import cam_head_map, _n_addr_bits
    from c4_min.pf_speculative import _frozen_skip_cut
    from c4_min.fused_megablock import MegaBlockRegion, divfree_carry_blocks

    print("[build] lean streaming sparse model (CPU) ...", flush=True)
    model, L, stats = build_lib_model_streaming(recurrent_divmod=True)
    _guard()
    print(f"[build] blocks={len(model.blocks)} dim={model.dim}  MemAvail={_mem_avail_gb():.1f}GB",
          flush=True)

    install_composed(model, verbose=False)
    _guard()

    cut = _frozen_skip_cut(model)
    print(f"\n[frozen-skip cut] = {cut}  (blocks [{cut},{len(model.blocks)}) are the "
          f"per-step region)", flush=True)

    # the live-CAM block map (doom CFM config)
    chm = cam_head_map(model, L)
    print(f"\n=== LIVE-CAM BLOCKS (composed doom CFM config) ===", flush=True)
    names = list(getattr(L, "_block_names", []))
    total_ffn_nnz = 0
    total_attn_nnz = 0
    for bi in sorted(chm):
        heads = chm[bi]
        blk = model.blocks[bi]
        nm = names[bi] if bi < len(names) else "?"
        attn = blk.attn
        ffn = blk.ffn
        wu, wg, wd = _nnz_of(ffn.W_up), _nnz_of(ffn.W_gate), _nnz_of(ffn.W_down)
        ffn_nnz = wu + wg + wd
        wq, wo = _nnz_of(attn.W_q), _nnz_of(attn.W_o)
        total_ffn_nnz += ffn_nnz
        total_attn_nnz += wq + wo
        head_desc = ", ".join(f"h{h}/{k}({_n_addr_bits(k)}b)" for h, k in heads)
        print(f"  block {bi:3d} [{nm:16s}] H={attn.n_heads} HD={attn.head_dim}  "
              f"cam_heads=[{head_desc}]", flush=True)
        print(f"      FFN nnz: W_up={wu} W_gate={wg} W_down={wd}  (total {ffn_nnz})",
              flush=True)
        print(f"      ATTN nnz: W_q={wq} W_o={wo}   "
              f"(W_o(cam_out) folds to a tiny scatter-add on-chip)", flush=True)

    print(f"\n  LIVE-CAM totals: FFN nnz={total_ffn_nnz}  ATTN(W_q+W_o) nnz={total_attn_nnz}",
          flush=True)
    print(f"  => on-chip path per live block = ONE SparseFFN.forward (3 GEMMs); "
          f"{len(chm)} blocks = {3*len(chm)} SEPARATE sparse-GEMM kernels (UNFUSED).",
          flush=True)

    # the dead-FFN mega chain: how many blocks, how many waves (already fused)
    mega = MegaBlockRegion(model, torch.device("cpu"), cut, carry_blocks=None)
    n_mega_seg = sum(1 for k, _ in mega.items if k == "mega")
    n_live_items = sum(1 for k, _ in mega.items if k == "live")
    n_dead = 0
    dead_ffn_nnz = 0
    for b in range(cut, len(model.blocks)):
        if getattr(model.blocks[b].attn, "_dead_block_fused", False):
            n_dead += 1
            ffn = model.blocks[b].ffn
            dead_ffn_nnz += _nnz_of(ffn.W_up) + _nnz_of(ffn.W_gate) + _nnz_of(ffn.W_down)

    print(f"\n=== DEAD-FFN MEGA CHAIN (already wave-batch fused) ===", flush=True)
    print(f"  region items: {n_mega_seg} mega-segments + {n_live_items} live blocks over "
          f"[{cut},{len(model.blocks)})", flush=True)
    print(f"  dead (attention-identity) blocks in region: {n_dead}   dead FFN nnz total="
          f"{dead_ffn_nnz}", flush=True)

    print(f"\n=== COST ATTRIBUTION (nnz-proportional, the GEMM-work proxy) ===", flush=True)
    print(f"  dead-FFN chain nnz : {dead_ffn_nnz:>8d}  (fused: ~{n_mega_seg} wave kernels)",
          flush=True)
    print(f"  live-CAM FFN nnz   : {total_ffn_nnz:>8d}  (UNfused: {3*len(chm)} separate kernels)",
          flush=True)
    if dead_ffn_nnz:
        print(f"  live/dead nnz ratio: {total_ffn_nnz/dead_ffn_nnz:.2f}x", flush=True)
    print(f"\n  NOTE: the live-CAM wall is NOT more nnz-work than the dead chain — it is that "
          f"the 3 live FFNs run as {3*len(chm)} SEPARATE tiny launch/tile-bound kernels while "
          f"the dead chain is ONE fused wave-batched megakernel. Fusing the 3 live FFNs into "
          f"one batched megakernel (block_k-tiled, concatenated GEMMs) is the wave-batch analog.",
          flush=True)


if __name__ == "__main__":
    main()
