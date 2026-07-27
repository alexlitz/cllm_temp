"""Follow-up: MQA (memory-safe), gelu divergence-magnitude probe, RMSNorm in-place.

Complements ``_agent_transform_verify`` (which covered baseline/BOS-sink/RoPE/GeLU
byte-exact but OOM'd on MQA and skipped GQA because n_heads=23 is prime).  This run:
  * MQA (n_kv=1): tie all KV heads to their mean; save originals on CPU (no GPU OOM).
  * GeLU divergence PROBE: report the raw residual max-abs divergence at the AX nibble
    decode dims (silu vs swish_b vs gelu) — quantifies HOW close the activation swap
    comes to flipping a nibble argmax (the byte-exactness margin), not just pass/fail.
  * RMSNorm (T1): wrap each block in RMSNorm with the compensator-lane identity gamma
    (qwen_embed.rmsnorm_identity_gamma) using a free residual dim, and re-run the
    battery.  Tests whether the VM rides RMSNorm byte-exact (the compensator fold).

Run:  C4_XFORM_DEVICE=cuda:0 python -m c4_release.c4_min._agent_transform_verify2
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from . import nibble_pure_forward_complete as N
from . import isa
from . import blogspec_model as BM
from ._agent_transform_verify import (
    run_on_device, battery, run_battery, patch_ffn_activation, set_attn_config,
    _gelu, _swish_b, BETA,
)

DEVICE = os.environ.get("C4_XFORM_DEVICE", "cuda:0")


def apply_kv_grouping_cpu(model, n_kv):
    """MHA -> GQA(n_kv)/MQA group-average, saving originals on CPU (GPU-mem-safe)."""
    saved = []
    H = model.blocks[0].attn.n_heads
    hd = model.blocks[0].attn.head_dim
    assert H % n_kv == 0, f"H={H} not divisible by n_kv={n_kv}"
    hpg = H // n_kv
    seen = set()
    for blk in model.blocks:
        at = blk.attn
        if id(at) in seen:            # recurrent build shares Attn modules; do once
            continue
        seen.add(id(at))
        saved.append((at, at.W_k.data.detach().cpu().clone(),
                      at.W_v.data.detach().cpu().clone()))
        for W in (at.W_k, at.W_v):
            Wv = W.data.view(H, hd, at.dim)
            for g in range(n_kv):
                sl = slice(g * hpg, (g + 1) * hpg)
                Wv[sl] = Wv[sl].mean(dim=0, keepdim=True)

    def restore():
        for at, wk, wv in saved:
            at.W_k.data.copy_(wk.to(at.W_k.device))
            at.W_v.data.copy_(wv.to(at.W_v.device))
    return restore


def gelu_divergence_probe(model, L, dev):
    """For a couple programs, run the full step ONCE under silu and under each swap,
    and report the max-abs divergence of the RAW residual at the AX nibble decode
    dims (before the argmax snap).  This is the byte-exactness margin: if it stays
    well under 0.5 the nibble argmax is safe; near 8 (half a nibble step) it flips."""
    progs = {
        "add": isa.assemble([('IMM', 5), ('PSH', 0), ('IMM', 3), ('ADD', 0), ('HALT', 0)]),
        "mul": isa.assemble([('IMM', 6), ('PSH', 0), ('IMM', 7), ('MUL', 0), ('HALT', 0)]),
        "div": isa.assemble([('IMM', 40), ('PSH', 0), ('IMM', 6), ('DIV', 0), ('HALT', 0)]),
    }

    def last_state(code):
        # run to the step BEFORE halt, capture the final-row residual at AX dims.
        stream = [__import__('c4_release.c4_min.blogspec_vocab', fromlist=['BOS']).BOS] \
            + N._build_frame(0, 0, N.SP_INIT, N.SP_INIT, 0)
        # run a few steps to reach the ALU-result step; reuse run_on_device path but
        # capture the residual of the LAST forward.
        overlay = N.make_overlay_complete(code, L, store_log={})
        toks = torch.tensor([stream], device=dev)
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            for blk in model.blocks:
                x = blk(x)
        return x[0, -1].detach().float().cpu()

    print("\n[gelu PROBE] max-abs residual divergence at AX nibble dims "
          "(step-0 forward; nibble flips at ~8):", flush=True)
    for name, code in progs.items():
        base = last_state(code)
        r = patch_ffn_activation(lambda x: _swish_b(x, BETA))
        sw = last_state(code); r()
        r = patch_ffn_activation(_gelu)
        ge = last_state(code); r()
        ax_dims = [L.AX + k for k in range(8)]
        d_sw = (sw[ax_dims] - base[ax_dims]).abs().max().item()
        d_ge = (ge[ax_dims] - base[ax_dims]).abs().max().item()
        d_sw_all = (sw - base).abs().max().item()
        d_ge_all = (ge - base).abs().max().item()
        print(f"    {name:5s} swish_b: AXdim={d_sw:.4f} allresid={d_sw_all:.2f} | "
              f"gelu: AXdim={d_ge:.4f} allresid={d_ge_all:.2f}", flush=True)


# ------------------------------------------------------------------
# T1: RMSNorm in-place via the compensator-lane identity (no rebuild).
# ------------------------------------------------------------------
def apply_rmsnorm_inplace(model, L, K):
    """Turn every block into pre-norm RMSNorm with compensator-identity gamma, using
    a free residual dim as the compensator lane.  Returns (restore, comp_dim).

    The compensator lane must hold K >> the largest real-lane magnitude so RMSNorm
    is an identity on the real dims.  We pick a free scratch dim and write K to it in
    EVERY embedding row (so every token's residual carries it), set gamma=K/sqrt(dim)
    on fresh RMSNorm modules, and flip each block's norm mode to 'rmsnorm'.
    """
    dim = model.dim
    # find a free dim: one that is zero across all embedding rows AND not a known
    # live band.  Use the last dim (padding) — verify it's all-zero in embed.
    comp = dim - 1
    col = model.embed.data[:, comp]
    assert float(col.abs().max()) == 0.0, f"dim {comp} not free (max={col.abs().max()})"
    saved_embed = model.embed.data[:, comp].clone()
    gamma = (K / math.sqrt(dim))

    seen = set()
    saved_blocks = []
    with torch.no_grad():
        model.embed.data[:, comp] = K
        for blk in model.blocks:
            if id(blk) in seen:
                continue
            seen.add(id(blk))
            saved_blocks.append(blk)
            an = BM.RMSNorm(dim).to(model.embed.device)
            fn = BM.RMSNorm(dim).to(model.embed.device)
            an.weight.data.fill_(gamma)
            fn.weight.data.fill_(gamma)
            blk.attn_norm = an
            blk.ffn_norm = fn
            blk.norm = "rmsnorm"
            blk.attn.dim  # touch

    def restore():
        with torch.no_grad():
            model.embed.data[:, comp] = saved_embed
            for blk in saved_blocks:
                blk.norm = "none"
                if hasattr(blk, "attn_norm"):
                    del blk.attn_norm
                if hasattr(blk, "ffn_norm"):
                    del blk.ffn_norm
    return restore, comp


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print(f"c4_min VM x model_subsets transform verification (part 2)  device={DEVICE}")
    print("=" * 78)
    t0 = time.time()
    model, L = N.build_pure_forward_complete_model(code_size=24, recurrent_divmod=True)
    model.eval()
    dev = torch.device(DEVICE)
    model.to(dev)
    print(f"  built+moved {time.time()-t0:.1f}s | n_blocks={len(model.blocks)} "
          f"dim={model.dim} n_heads={model.blocks[0].attn.n_heads} "
          f"head_dim={model.blocks[0].attn.head_dim}", flush=True)

    results = {}

    def stage(key, label):
        p, t, f = run_battery(model, L, key, dev, verbose=True)
        results[key] = (p, t, f)
        print(f"  => [{label}] {p}/{t} byte-exact ({time.time()-t0:.0f}s)", flush=True)

    # sanity re-baseline (cheap confidence the part-2 build matches part-1)
    print("\n[BASELINE re-check]", flush=True)
    stage("baseline", "baseline")

    # -- gelu divergence probe (quantify the byte-exactness margin) --
    gelu_divergence_probe(model, L, dev)

    # -- T3 MQA (n_kv=1), memory-safe --
    print("\n[T3] MHA -> MQA n_kv=1 (tie all 23 KV heads to mean); predict CAM BREAK",
          flush=True)
    r = apply_kv_grouping_cpu(model, 1); stage("mqa", "T3 mqa"); r()

    # -- T1 RMSNorm in-place (compensator identity) --
    for K in (4000.0, 5.0e5, 5.0e7):
        print(f"\n[T1] norm-free -> RMSNorm (compensator K={K:g}); predict EXACT if K "
              f"dominates register lanes (SP_INIT=65536)", flush=True)
        r, comp = apply_rmsnorm_inplace(model, L, K)
        stage(f"rmsnorm_K{K:g}", f"T1 rmsnorm K={K:g}")
        r()

    print("\n" + "=" * 78)
    print("PART-2 MATRIX")
    print("=" * 78)

    def maxdiv_of(fails):
        return max([d for (_, _, _, d) in fails if d is not None], default=0)

    print(f"{'transform':16s} {'pass':>7s} {'maxdiv':>7s}  verdict")
    for key in results:
        p, t, f = results[key]
        md = maxdiv_of(f)
        if p == t:
            verd = "byte-exact"
        elif p == 0:
            verd = "BREAKS-all"
        else:
            verd = f"partial({t-p} fail: " + ",".join(n for (n, _, _, _) in f) + ")"
        print(f"{key:16s} {p:>3d}/{t:<3d} {md:>7d}  {verd}")

    print(f"\ntotal wall: {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
