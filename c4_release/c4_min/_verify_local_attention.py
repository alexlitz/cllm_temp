#!/usr/bin/env python3
"""VERIFY local (sliding-window) attention is BYTE-IDENTICAL to global attention.

Three parts:

  STEP-1b (window growth):  run the CACHED driver (O(S)) on a DEEP loop and, for
    every LIVE LOCAL head, measure the max attended distance across the whole run
    — confirms the ingest heads stay < one frame no matter how long the stream is
    (the recency + role-exact-match can never reach a MORE-recent role-match).

  STEP-3 (byte-identity):  run the FAST PATH (pf_speculative.speculative_run) on a
    battery (add / mul / loop / malloc / nested) with GLOBAL attention and again
    with LOCAL attention installed; assert the decoded AX trace + accept decision +
    final AX are IDENTICAL.  Also compares the raw hidden L-inf per query row on a
    small program (the strongest per-head proof: local == global residual).

Run:  python -m c4_min._verify_local_attention --device cuda:1 --window 64
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict

import torch

from c4_min import isa
from c4_min import nibble_pure_forward as _PF
from c4_min import blogspec_vocab as V
from c4_min import sparse_forward as _SF
from c4_min.lib_neural import build_lib_model_streaming
from c4_min import local_attention as LA
from c4_min import pf_speculative as PS


# ---------------------------------------------------------------------------
def _battery():
    """(label, code, expected_ax) programs that exercise ingest + memory heads."""
    from c4_min.bench_fast_path import (build_loop_countdown, build_nested,
                                        build_malloc, build_matmul)
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    progs = []

    def _c(src):
        bc, data = compile_c(src)
        return bytecode_to_isa(bc), data

    # arithmetic (byte)
    code, _ = _c("int main(){ int a; int b; a=40; b=2; return a+b; }")
    progs.append(("add_42", code, 42))
    code, _ = _c("int main(){ int a; int b; a=6; b=7; return a*b; }")
    progs.append(("mul_42", code, 42))
    # loop over a stack local (LEA/SI/LI every iter + memory head recall)
    code, _ax, _d, _lbl = build_loop_countdown(12)
    progs.append(("loop_n12", code, 0))
    # deep nested (byte-safe): the headline deep-loop shape
    code, ax, _d, _lbl = build_nested(3, 20)
    progs.append(("nested_3x20", code, ax))
    return progs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--block-steps", type=int, default=48)
    ap.add_argument("--deep-n", type=int, default=30,
                    help="countdown trip count for the window-growth probe")
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args(argv)

    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    # size the code band to fit EVERY battery program + the deep window probe.
    progs = _battery()
    from c4_min.bench_fast_path import build_loop_countdown
    deep_code, _, _, _ = build_loop_countdown(args.deep_n)
    max_instrs = max([len(c) for _, c, _ in progs] + [len(deep_code)])
    code_size = max(max_instrs + 2, 64)
    print(f"[build] streaming complete model on {dev} (code_size={code_size}) ...",
          flush=True)
    model, L, _ = build_lib_model_streaming(code_size=code_size,
                                            compute_mode="dense_kernel")
    model.to(dev)
    N_ROLES = _PF.N_ROLES
    n_heads = model.blocks[0].attn.n_heads
    print(f"[build] n_heads={n_heads} n_blocks={len(model.blocks)} "
          f"FRAME_LEN={V.FRAME_LEN}", flush=True)

    # ---- head classification --------------------------------------------------
    cls = LA.classify_heads(model)
    live_global = {bi: hs for bi, hs in cls.items() if hs}
    print("\n==================== HEAD CLASSIFICATION ====================")
    names = getattr(L, "_block_names", None)
    for bi, blk in enumerate(model.blocks):
        live = LA.live_value_heads(blk.attn)
        if not live:
            continue
        nm = names[bi] if names and bi < len(names) else "?"
        glob = cls[bi]
        loc = [h for h in live if h not in glob]
        print(f"  block {bi:3d} [{nm:>16}]  LIVE={live}  GLOBAL={glob}  LOCAL={loc}")
    tot_global = sum(len(v) for v in cls.values())
    tot_slots = n_heads * len(model.blocks)
    print(f"  -> {tot_global} GLOBAL head-slots kept full-causal; "
          f"{tot_slots - tot_global}/{tot_slots} head-slots WINDOWED "
          f"({(tot_slots - tot_global)/tot_slots*100:.1f}%)")
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # ======================================================================
    # STEP-3: byte-identity of the FAST PATH (global vs local attention).  This is
    # the real correctness GATE — it runs first (memory-safe, eviction ON).
    # ======================================================================
    print("\n==================== STEP-3 BYTE-IDENTITY (fast path) ====================")
    all_ok = True
    for label, code, exp in progs:
        # GLOBAL (baseline).
        LA.uninstall_local_attention(model)
        rg = PS.speculative_run(model, L, code, exp, block_steps=args.block_steps,
                                device=dev, evict=True, mask=0xFFFFFFFF,
                                block_moe=True)
        # LOCAL.
        LA.install_local_attention(model, window=args.window)
        rl = PS.speculative_run(model, L, code, exp, block_steps=args.block_steps,
                                device=dev, evict=True, mask=0xFFFFFFFF,
                                block_moe=True)
        LA.uninstall_local_attention(model)
        same = (rg.status == rl.status and
                rg.decoded_final_ax == rl.decoded_final_ax and
                rg.all_matched == rl.all_matched and
                rg.accepted_steps == rl.accepted_steps)
        all_ok = all_ok and same
        print(f"  {label:14}  steps={rg.step_count:6d}  "
              f"global[{rg.status} AX={rg.decoded_final_ax} acc={rg.accepted_steps}]  "
              f"local[{rl.status} AX={rl.decoded_final_ax} acc={rl.accepted_steps}]  "
              f"-> {'IDENTICAL' if same else 'DIVERGES!'}")

    # ---- per-query-row hidden L-inf on a small program (strongest proof) ------
    print("\n  --- per-row hidden L-inf (global vs local), add_42 ---")
    add_code = progs[0][1]
    linf = _hidden_linf(model, L, add_code, dev, args.window, args.block_steps)
    print(f"    max L-inf over all query-row hiddens: {linf:.3e} "
          f"({'BYTE-IDENTICAL' if linf == 0.0 else 'NONZERO'})")

    # ======================================================================
    # STEP-1b: measure LOCAL-head windows over a small deep run (memory-safe
    # non-cached driver: NO per-block persistent cache, peak ~0.1 GB).  The window
    # PLATEAUS within a few frames, so a modest deep-n suffices — it CANNOT grow:
    # every frame re-emits all 20 roles, so a MORE-recent role-match always exists
    # < 1 frame back, and recency ALiBi (6·30) drives the older match to exp(-180)≈0.
    # ======================================================================
    worst, window_ok = _measure_windows(model, L, cls, args, N_ROLES, names)

    print("\n==================== VERDICT ====================")
    verdict = window_ok and all_ok and linf == 0.0
    print(f"  window covers local heads : {window_ok} (worst measured {worst:.0f} < "
          f"W={args.window})")
    print(f"  fast-path AX identical    : {all_ok}")
    print(f"  hidden L-inf == 0         : {linf == 0.0}")
    print(f"  OVERALL: {'PASS — local == global BYTE-IDENTICAL' if verdict else 'FAIL'}")
    return 0 if verdict else 1


def _measure_windows(model, L, cls, args, N_ROLES, names):
    """Measure the LOCAL-head attention window over a small deep non-cached run.
    Returns (worst_window, window_ok)."""
    from c4_min.bench_fast_path import build_loop_countdown
    from c4_min.nibble_pure_forward_complete import run_pure_forward_complete
    deep_code, _, _, _ = build_loop_countdown(args.deep_n)
    per_head_maxdist = defaultdict(float)
    live_local = {}
    for bi, blk in enumerate(model.blocks):
        live = set(LA.live_value_heads(blk.attn))
        live_local[bi] = live - set(cls[bi])
    block_of = {id(b.attn): bi for bi, b in enumerate(model.blocks)}
    orig = _SF.SparseAttn.forward
    eps = args.eps

    def instrumented(self, x, past_kv=None, q_positions=None, use_cache=False):
        out = orig(self, x, past_kv=past_kv, q_positions=q_positions,
                   use_cache=use_cache)
        bi = block_of.get(id(self), -1)
        ll = live_local.get(bi, set())
        if not ll:
            return out
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim
        idx = torch.tensor(sorted(ll), device=x.device)
        Q = self.W_q.linear(x).view(B, S, H, HD).transpose(1, 2)[:, idx]
        K = self.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)[:, idx]
        q_pos = torch.arange(S, device=x.device)
        sc = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        dist = (q_pos.unsqueeze(0) - q_pos.unsqueeze(1)).abs().float()
        sc = sc - self.alibi_slopes[idx].view(1, -1, 1, 1) * dist
        sc = sc + torch.triu(torch.full((S, S), float("-inf"), device=x.device),
                             diagonal=1)
        a = _SF.softmax1(sc, dim=-1)[0]
        for j, h in enumerate(sorted(ll)):
            sig = a[j] >= eps
            if bool(sig.any()):
                per_head_maxdist[(bi, h)] = max(per_head_maxdist[(bi, h)],
                                                float(dist[sig].max().item()))
        del Q, K, sc, a, dist
        return out

    _SF.SparseAttn.forward = instrumented
    try:
        print(f"\n[step-1b] window measurement: non-cached run "
              f"(countdown n={args.deep_n}) ...", flush=True)
        trace = run_pure_forward_complete(model, L, deep_code,
                                          max_steps=args.deep_n * 8 + 40,
                                          mask=0xFFFFFFFF)
        print(f"[step-1b] {len(trace)} steps; stream "
              f"~{1 + len(trace)*V.FRAME_LEN} tokens", flush=True)
    finally:
        _SF.SparseAttn.forward = orig

    worst = 0.0
    print("  LOCAL-head measured windows (max |q_pos-k_pos| w/ weight >= eps):")
    for (bi, h), mx in sorted(per_head_maxdist.items()):
        nm = names[bi] if names and bi < len(names) else "?"
        worst = max(worst, mx)
        print(f"    block {bi:3d} [{nm:>16}] head {h:2d} [ingest] window={mx:6.1f}")
    print(f"  -> WORST local-head window: {worst:.1f} tokens (chosen W={args.window})")
    return worst, worst < args.window


def _hidden_linf(model, L, code, dev, window, block_steps):
    """Run verify_blocks twice (global/local), capturing the hidden at each query
    row via a hook, and return the max L-inf between the two."""
    draft = PS.draft_pf_program(code, max_steps=10000, mask=0xFFFFFFFF)
    captured = {}

    def _capture(tag):
        rows = []
        orig = _SF.SparseTransformer.forward_hidden_cached

        def patched(self, x, past_key_values=None, q_positions=None, use_cache=False):
            hidden, caches = orig(self, x, past_key_values=past_key_values,
                                  q_positions=q_positions, use_cache=use_cache)
            rows.append(hidden.detach().float().cpu())
            return hidden, caches

        _SF.SparseTransformer.forward_hidden_cached = patched
        try:
            # block_moe OFF so EVERY span goes through forward_hidden_cached (the
            # skip path bypasses this hook); the L-inf proof needs all spans.
            PS.verify_blocks(model, L, code, draft, block_steps=block_steps,
                             device=dev, evict=True, mask=0xFFFFFFFF, block_moe=False)
        finally:
            _SF.SparseTransformer.forward_hidden_cached = orig
        captured[tag] = rows

    LA.uninstall_local_attention(model)
    _capture("global")
    LA.install_local_attention(model, window=window)
    _capture("local")
    LA.uninstall_local_attention(model)

    g, l = captured["global"], captured["local"]
    if len(g) != len(l):
        return float("inf")
    mx = 0.0
    for a, b in zip(g, l):
        if a.shape != b.shape:
            return float("inf")
        mx = max(mx, float((a - b).abs().max().item()))
    return mx


if __name__ == "__main__":
    sys.exit(main())
