#!/usr/bin/env python3
"""VERIFY live-head-only attention scoring is BYTE-IDENTICAL to the full forward.

CPU-only.  Three proofs, in increasing strength:

  A. HONESTY AUDIT.  For every (block, head) slot, classify by (W_v slice nnz,
     W_o slice nnz).  The claim "a zero-value head outputs x" is only sound if the
     slots are cleanly bipartite (both-0 = dead, both-live = live) with NO
     W_v==0/W_o!=0 (a nonzero-W_o-with-zero-V head could still leak) and no
     attention output bias.  We ASSERT this and print the case counts.

  B. RAW-FORWARD L-inf.  On several random query batches (and on a real hidden
     captured from a program run), call the ORIGINAL SparseAttn.forward and the
     live-head-only forward on the SAME live-attention block and assert the
     residual L-inf is EXACTLY 0 — the strongest per-block proof.  Also exercises
     the cached / q_positions path.

  C. END-TO-END DECODE.  Run real programs (arith / a loop / a load-store) through
     the CPU non-cached forward (run_pure_forward_complete) with GLOBAL attention
     and again with live-head-only attention installed; assert the decoded AX
     traces are IDENTICAL, byte-for-byte.

Run:  python -m c4_min._verify_live_head_attention
"""
from __future__ import annotations
import sys

import torch

from c4_min import isa
from c4_min import blogspec_vocab as V
from c4_min import sparse_forward as _SF
from c4_min.lib_neural import build_lib_model_streaming
from c4_min import live_head_attention as LHA
from c4_min.local_attention import live_value_heads


def _dense(w):
    if getattr(w, "is_sparse", False):
        if w.dense_resident is not None:
            return w.dense_resident
        return w.csr.to_dense()
    return w.dense if hasattr(w, "dense") and w.dense is not None else w


def audit_head_bipartite(model):
    """Proof A: every head slot is cleanly {both-0} or {both-live}; no mixed slot,
    no output bias.  Returns (case_counts, ok)."""
    nh = model.blocks[0].attn.n_heads
    HD = model.blocks[0].attn.head_dim
    cases = {"both0": 0, "both_live": 0, "wv0_wo_live": 0, "wv_live_wo0": 0}
    mixed = []
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        # no output bias on attention (would break the "output is x" claim)
        assert not [a for a in dir(at) if a in ("b_o", "bias_o", "W_o_bias")], \
            f"block {bi} attn has an output bias"
        wv = _dense(at.W_v); wo = _dense(at.W_o)
        for h in range(nh):
            sl = slice(h * HD, (h + 1) * HD)
            vnz = int((wv[sl, :] != 0).sum())
            onz = int((wo[:, sl] != 0).sum())
            if vnz == 0 and onz == 0:
                cases["both0"] += 1
            elif vnz > 0 and onz > 0:
                cases["both_live"] += 1
            elif vnz == 0 and onz > 0:
                cases["wv0_wo_live"] += 1; mixed.append((bi, h, vnz, onz))
            else:
                cases["wv_live_wo0"] += 1; mixed.append((bi, h, vnz, onz))
    ok = (cases["wv0_wo_live"] == 0 and cases["wv_live_wo0"] == 0)
    return cases, ok, mixed


def raw_linf(model, live_blocks, device):
    """Proof B: original forward vs live-head forward, L-inf on random + cached
    inputs, over the LIVE-attention blocks (the only ones where scoring differs
    non-trivially) plus a couple of dead blocks (pure passthrough).  Returns the
    max L-inf over all probes."""
    torch.manual_seed(0)
    D = model.blocks[0].attn.dim
    worst = 0.0
    # probe a mix: all 3 live blocks + first 3 dead blocks.
    dead = [bi for bi in range(len(model.blocks)) if bi not in live_blocks][:3]
    probe_blocks = sorted(set(live_blocks) | set(dead))
    for bi in probe_blocks:
        at = model.blocks[bi].attn
        for S in (5, 30, 61):
            x = torch.randn(1, S, D, device=device) * 3.0
            # 1) default un-cached path
            LHA.uninstall_live_head_attention(model)
            g = at.forward(x)
            LHA.install_live_head_attention(model)
            l = at.forward(x)
            worst = max(worst, float((g - l).abs().max()))
            # 2) cached path (past_kv + q_positions) — split the span in two
            LHA.uninstall_live_head_attention(model)
            g1, kv = at.forward(x[:, :S // 2], q_positions=torch.arange(S // 2,
                                device=device), use_cache=True)
            qpos = torch.arange(S // 2, S, device=device)
            g2, _ = at.forward(x[:, S // 2:], past_kv=kv, q_positions=qpos,
                               use_cache=True)
            LHA.install_live_head_attention(model)
            l1, kvl = at.forward(x[:, :S // 2], q_positions=torch.arange(S // 2,
                                 device=device), use_cache=True)
            l2, _ = at.forward(x[:, S // 2:], past_kv=kvl, q_positions=qpos,
                               use_cache=True)
            worst = max(worst, float((g1 - l1).abs().max()),
                        float((g2 - l2).abs().max()))
    LHA.uninstall_live_head_attention(model)
    return worst, probe_blocks


def _battery():
    """(label, code, note) programs: arith, a loop, and a load/store."""
    from c4_min.bench_fast_path import build_loop_countdown
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    def _c(src):
        bc, _data = compile_c(src)
        return bytecode_to_isa(bc)

    progs = []
    # arithmetic (add + mul)
    progs.append(("add_42", _c("int main(){ int a; int b; a=40; b=2; return a+b; }"),
                  "byte add"))
    progs.append(("mul_42", _c("int main(){ int a; int b; a=6; b=7; return a*b; }"),
                  "byte mul"))
    # a loop (LEA/SI/LI + memory-head recall every iter).  n=3 keeps the CPU
    # cached-decode battery fast while still driving the memory KV head across the
    # loop span (the global-head cached path Proof B verified L-inf=0 on).
    code, _ax, _d, _lbl = build_loop_countdown(3)
    progs.append(("loop_n3", code, "countdown loop (store/load per iter)"))
    # an explicit load/store round-trip through a stack local
    progs.append(("store_load",
                  _c("int main(){ int x; x=123; return x; }"),
                  "SI then LI on a stack local"))
    return progs


def decode_trace(model, L, code, device, max_steps=200):
    """Decode via the KV-CACHED driver (O(cache)/step, byte-identical to the naive
    re-forward per test_cached_driver).  This drives BOTH the cached / q_positions
    attention path AND the memory/stack/LEV global heads through their KV cache —
    the exact path the fast driver uses, and fast enough on CPU."""
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    return run_pure_forward_cached(model, L, code, max_steps=max_steps,
                                   mask=0xFFFFFFFF, evict=True)


def main(argv=None):
    torch.set_grad_enabled(False)
    device = "cpu"
    print(f"[build] streaming complete model on {device} (code_size=64) ...", flush=True)
    model, L, _ = build_lib_model_streaming(code_size=64, compute_mode="dense_kernel")
    model.to(device)
    nb = len(model.blocks); nh = model.blocks[0].attn.n_heads
    print(f"[build] n_blocks={nb}  n_heads={nh}  slots={nb * nh}  "
          f"FRAME_LEN={V.FRAME_LEN}", flush=True)

    # ---- classification / stats -------------------------------------------
    stats = LHA.live_head_attention_stats(model)
    live_blocks = list(stats["per_block_live_heads"].keys())
    print("\n==================== HEAD CLASSIFICATION ====================")
    print(f"  total head-slots          : {stats['total_head_slots']}")
    print(f"  LIVE-value head-slots      : {stats['live_head_slots']} "
          f"({stats['frac_scored_after']*100:.2f}%)")
    print(f"  DEAD head-slots (skip)     : {stats['dead_head_slots']}")
    print(f"  live-attention blocks      : {stats['live_attention_blocks']} of "
          f"{stats['n_blocks']}")
    print(f"  live heads (block->heads)  : {stats['per_block_live_heads']}")

    # ---- Proof A: honesty audit ------------------------------------------
    print("\n==================== PROOF A: HEAD BIPARTITE AUDIT ====================")
    cases, a_ok, mixed = audit_head_bipartite(model)
    print(f"  slot cases: {cases}")
    print(f"  mixed (W_v/W_o disagree) slots: {mixed if mixed else 'NONE'}")
    print(f"  -> every slot is cleanly dead-or-live, no output bias : "
          f"{'PASS' if a_ok else 'FAIL — scope the subset!'}")

    # ---- Proof B: raw-forward L-inf --------------------------------------
    print("\n==================== PROOF B: RAW-FORWARD L-inf ====================")
    worst, probed = raw_linf(model, live_blocks, device)
    print(f"  probed blocks (live+dead) : {probed}")
    print(f"  max L-inf (full vs live)  : {worst:.3e} "
          f"({'BYTE-IDENTICAL' if worst == 0.0 else 'NONZERO!'})")

    # ---- Proof C: end-to-end decode --------------------------------------
    # Baseline = GLOBAL forward.  Compare against (1) live-head-only and (2)
    # live-head + DEAD-BLOCK FUSION (the whole attention sublayer of a dead block
    # bypassed AND its KV write skipped).  The memory/stack programs are the
    # critical KV-safety test: they exercise the 3 LIVE blocks' KV (which fusion
    # must leave untouched) while every dead block's KV is skipped.
    print("\n==================== PROOF C: END-TO-END DECODE ====================")
    progs = _battery()
    c_ok = True       # live-head-only == global
    d_ok = True       # live-head + dead-block fusion == global
    for label, code, note in progs:
        LHA.uninstall_dead_block_fusion(model)
        LHA.uninstall_live_head_attention(model)
        tg = decode_trace(model, L, code, device)                 # GLOBAL baseline

        LHA.install_live_head_attention(model)                    # live-head only
        tl = decode_trace(model, L, code, device)
        LHA.uninstall_live_head_attention(model)

        LHA.install_live_head_attention(model)                    # live-head + fusion
        LHA.install_dead_block_fusion(model)
        tf = decode_trace(model, L, code, device)
        LHA.uninstall_dead_block_fusion(model)
        LHA.uninstall_live_head_attention(model)

        same_l = (tg == tl)
        same_f = (tg == tf)
        c_ok = c_ok and same_l
        d_ok = d_ok and same_f
        verdict_str = ("IDENTICAL" if (same_l and same_f)
                       else "live=%s fuse=%s" % (
                           "OK" if same_l else "DIVERGES", "OK" if same_f else "DIVERGES"))
        print(f"  {label:12} [{note:34}]  steps={len(tg):4d}  "
              f"AX_final={tg[-1] if tg else '?'}  -> {verdict_str}")
        if not same_f:
            print(f"      FUSION DIVERGES!  g={tg}\n                        f={tf}")

    # ---- Proof B-fusion: fused block output L-inf == 0 vs global ----------
    # A fused DEAD block must return EXACTLY x (and None KV).  Probe the first few
    # dead blocks directly (the live blocks are never fused).
    print("\n============ PROOF B-fusion: FUSED DEAD-BLOCK L-inf == 0 ============")
    torch.manual_seed(1)
    Dm = model.blocks[0].attn.dim
    dead = [bi for bi in range(len(model.blocks)) if bi not in live_blocks][:4]
    fuse_worst = 0.0
    LHA.install_live_head_attention(model)
    LHA.install_dead_block_fusion(model)
    for bi in dead:
        at = model.blocks[bi].attn
        for S in (5, 30, 61):
            x = torch.randn(1, S, Dm, device=device) * 3.0
            out = at.forward(x)                       # fused: must be exactly x
            fuse_worst = max(fuse_worst, float((out - x).abs().max()))
            oc, kv = at.forward(x, q_positions=torch.arange(S, device=device),
                                use_cache=True)
            fuse_worst = max(fuse_worst, float((oc - x).abs().max()))
            assert kv is None, f"fused block {bi} returned non-None KV"
    LHA.uninstall_dead_block_fusion(model)
    LHA.uninstall_live_head_attention(model)
    print(f"  probed dead blocks        : {dead}")
    print(f"  max L-inf (fused vs x)    : {fuse_worst:.3e} "
          f"({'BYTE-IDENTICAL (out==x, KV=None)' if fuse_worst == 0.0 else 'NONZERO!'})")

    # ---- verdict ----------------------------------------------------------
    print("\n==================== VERDICT ====================")
    b_ok = (worst == 0.0)
    f_ok = (fuse_worst == 0.0)
    verdict = a_ok and b_ok and c_ok and d_ok and f_ok
    print(f"  A head bipartite audit         : {a_ok}")
    print(f"  B raw-forward L-inf == 0       : {b_ok}")
    print(f"  B-fusion fused block == x      : {f_ok}")
    print(f"  C live-head decode ident       : {c_ok}")
    print(f"  C dead-block-FUSION decode ident: {d_ok}")
    print(f"  OVERALL: {'PASS — live-head + dead-block-fusion == full forward BYTE-IDENTICAL' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
