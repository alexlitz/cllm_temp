#!/usr/bin/env python3
"""EMPIRICAL per-head attention-window measurement (STEP 1 of the local-attn work).

Builds the streaming complete pure-forward model (23 heads: 20 ingest + memory
+ stack-pop + LEV) and runs a representative program (a loop that re-emits many
frames + a store/load so the memory head is exercised).  For every block that has
NON-ZERO attention weights, we instrument ``SparseAttn.forward`` to capture the
post-softmax attention weight matrix ``a[H, Sq, Sk]`` and, per head, measure the
MAXIMUM key distance ``|q_pos - k_pos|`` that carries weight >= EPS at ANY query
row.  That max-distance is the head's empirical attention window.

Output: per (block, head) the measured window; classify local (small window) vs
global (reaches far).  Zeroed-attention blocks are reported separately (their
attention output is provably ``x`` regardless of window — no key carries weight).

Run:  python -m c4_min._probe_attn_windows --device cuda:1
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict

import torch

from c4_min import isa
from c4_min import nibble_pure_forward as _PF
from c4_min import nibble_pure_forward_complete as _PFC
from c4_min import blogspec_vocab as V
from c4_min.lib_neural import build_lib_model_streaming
from c4_min import sparse_forward as _SF


def _loop_store_prog(n: int = 8):
    """A C-compiled countdown loop over a stack LOCAL — every iteration does a
    LEA/SI store of the counter and an LI load-back, so BOTH the ingest heads
    (every step) AND the memory KV head (the load-back across the loop span) are
    exercised.  Returns the compiled ISA program."""
    from c4_min.bench_fast_path import build_loop_countdown
    code, _ax, _data, _label = build_loop_countdown(n)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n", type=int, default=8, help="loop trip count")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="attention-weight floor for 'attends non-trivially'")
    args = ap.parse_args(argv)

    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    print(f"[build] streaming complete model on {dev} ...", flush=True)
    model, L, _ = build_lib_model_streaming(compute_mode="dense_kernel")
    model.to(dev)
    n_heads = model.blocks[0].attn.n_heads
    N_ROLES = _PF.N_ROLES
    print(f"[build] n_heads={n_heads}  N_ROLES={N_ROLES}  FRAME_LEN={V.FRAME_LEN}  "
          f"n_blocks={len(model.blocks)}", flush=True)

    try:
        code = _loop_store_prog(args.n)
    except Exception as e:  # noqa
        print(f"[warn] loop_compiler unavailable ({e}); using a hand loop", flush=True)
        code = None
    if code is None:
        # AX=n; loop: AX-=1; BNZ loop; EXIT.  Hand ISA.
        I = isa.Instr
        code = [I(isa.IMM, args.n), I(isa.PSH, 0), I(isa.IMM, 1),
                I(isa.SUB, 0), I(isa.BNZ, 1), I(isa.EXIT, 0)]

    # ---- instrument SparseAttn.forward to capture post-softmax weights ---------
    # Per (block index, head) -> max |q_pos - k_pos| with weight >= eps.
    per_head_maxdist = defaultdict(float)
    per_head_nonzero = defaultdict(bool)   # head has ANY key with weight>=eps beyond self
    per_head_selfonly = defaultdict(bool)
    block_of_attn = {}                     # id(attn) -> block index
    for bi, blk in enumerate(model.blocks):
        block_of_attn[id(blk.attn)] = bi

    orig_forward = _SF.SparseAttn.forward
    eps = args.eps

    def instrumented(self, x, past_kv=None, q_positions=None, use_cache=False):
        B, S, D = x.shape
        H, HD = self.n_heads, self.head_dim
        Q = self.W_q.linear(x).view(B, S, H, HD).transpose(1, 2)
        Knew = self.W_k.linear(x).view(B, S, H, HD).transpose(1, 2)
        Vnew = self.W_v.linear(x).view(B, S, H, HD).transpose(1, 2)
        if q_positions is None:
            q_pos = torch.arange(S, device=x.device)
        else:
            q_pos = q_positions.to(device=x.device, dtype=torch.long)
        if past_kv is not None:
            K_cache, V_cache, pos_cache = past_kv
            K = torch.cat([K_cache, Knew], dim=2)
            Vv = torch.cat([V_cache, Vnew], dim=2)
            k_pos = torch.cat([pos_cache.to(x.device), q_pos], dim=0)
        else:
            K, Vv, k_pos = Knew, Vnew, q_pos
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if past_kv is None and q_positions is None:
            pos = torch.arange(S, device=x.device)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()
            scores = scores - self.alibi_slopes.view(1, H, 1, 1) * dist
            causal = torch.triu(torch.full((S, S), float("-inf"), device=x.device),
                                diagonal=1)
            scores = scores + causal
            k_pos_row = pos
            q_pos_row = pos
        else:
            dist = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs().float()
            scores = scores - self.alibi_slopes.view(1, H, 1, 1) * dist.unsqueeze(0)
            mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1))
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            k_pos_row = k_pos
            q_pos_row = q_pos
        a = _SF.softmax1(scores, dim=-1)   # [B,H,Sq,Sk]

        # ---- measure per-head max attended distance -----------------------
        bi = block_of_attn.get(id(self), -1)
        aw = a[0]  # [H,Sq,Sk]
        # distance matrix [Sq,Sk]
        dmat = (q_pos_row.view(-1, 1) - k_pos_row.view(1, -1)).abs().float()
        for h in range(H):
            wh = aw[h]                       # [Sq,Sk]
            mask_sig = wh >= eps
            if mask_sig.any():
                dsig = dmat[mask_sig]
                mx = float(dsig.max().item())
                per_head_maxdist[(bi, h)] = max(per_head_maxdist[(bi, h)], mx)
                # non-self attention (dist>0) that is significant?
                if float(dsig.max().item()) > 0.0:
                    per_head_nonzero[(bi, h)] = True
            # does head EVER output non-trivially (V nonzero)?  Track weight mass.

        out = torch.matmul(a, Vv).transpose(1, 2).contiguous().view(B, S, D)
        out = x + self.W_o.linear(out)
        if use_cache:
            return out, (K, Vv, k_pos)
        return out

    _SF.SparseAttn.forward = instrumented
    try:
        print(f"[run] {len(code)} instrs, countdown n={args.n} "
              f"(full non-cached forward per step) ...", flush=True)
        trace = _PFC.run_pure_forward_complete(model, L, code,
                                               max_steps=args.max_steps,
                                               mask=0xFFFFFFFF)
        print(f"[run] {len(trace)} steps executed; final AX={trace[-1] if trace else '?'}",
              flush=True)
    finally:
        _SF.SparseAttn.forward = orig_forward

    # ---- report which heads/blocks are non-zero and their windows -------------
    # Identify which blocks have non-zero attention weights baked.
    def _nnz(w):
        n = getattr(w, "nnz", None)
        if n is not None:
            return int(n)
        return int((w != 0).sum().item())

    nonzero_attn_blocks = {}
    for bi, blk in enumerate(model.blocks):
        at = blk.attn
        # attention output is x (trivial) iff W_v==0 OR W_o==0
        wv_nnz = _nnz(at.W_v)
        wo_nnz = _nnz(at.W_o)
        nonzero_attn_blocks[bi] = (wv_nnz, wo_nnz)

    print("\n==================== ATTENTION-ACTIVE BLOCKS ====================")
    names = getattr(L, "_block_names", None)
    active_blocks = []
    for bi in sorted(nonzero_attn_blocks):
        wv_nnz, wo_nnz = nonzero_attn_blocks[bi]
        if wv_nnz > 0 and wo_nnz > 0:
            nm = names[bi] if names and bi < len(names) else "?"
            active_blocks.append(bi)
            print(f"  block {bi:3d} [{nm:>16}]  W_v nnz={wv_nnz}  W_o nnz={wo_nnz}  "
                  f"(ATTENTION OUTPUT IS NON-TRIVIAL)")
    print(f"  -> {len(active_blocks)} of {len(model.blocks)} blocks have live attention; "
          f"the rest output x (pure passthrough).")

    # per-head LIVE flag: head h contributes iff its W_v rows (input-dim slice
    # h*HD..(h+1)*HD) are non-zero AND W_o reads those cols (h*HD..(h+1)*HD).
    HD = model.blocks[0].attn.head_dim

    def _dense(w):
        if getattr(w, "is_sparse", False):
            if w.dense_resident is not None:
                return w.dense_resident
            return w.csr.to_dense()
        return w.dense

    def _live_heads(at):
        wv = _dense(at.W_v)   # [out=D, in=D]; head h's value = rows [h*HD:(h+1)*HD]
        wo = _dense(at.W_o)   # [out=D, in=D]; head h read from cols [h*HD:(h+1)*HD]
        live = []
        for h in range(n_heads):
            sl = slice(h * HD, (h + 1) * HD)
            vnz = int((wv[sl, :] != 0).sum().item())
            onz = int((wo[:, sl] != 0).sum().item())
            if vnz > 0 and onz > 0:
                live.append(h)
        return live

    def _role(h):
        return ("ingest[%d]" % h if h < N_ROLES else
                ("MEMORY" if h == N_ROLES else
                 ("STACK-POP" if h == N_ROLES + 1 else "LEV")))

    print("\n==================== PER-HEAD MEASURED WINDOW ====================")
    print(f"  (max |q_pos - k_pos| carrying attn weight >= {eps} at any query row)")
    print("  *** ONLY heads with NON-ZERO value output (W_v!=0 & W_o!=0) affect the")
    print("      residual; a zero-V head outputs 0 regardless of its window. ***\n")
    classification = {}
    for bi in active_blocks:
        nm = names[bi] if names and bi < len(names) else "?"
        live = _live_heads(model.blocks[bi].attn)
        print(f"  --- block {bi} [{nm}]  LIVE heads: {live} ---")
        for h in range(n_heads):
            mx = per_head_maxdist.get((bi, h), None)
            if mx is None:
                continue
            slope = float(model.blocks[bi].attn.alibi_slopes[h])
            islive = h in live
            tag = "LIVE " if islive else "zeroV"
            print(f"    head {h:2d} [{_role(h):>10}] {tag} slope={slope:7.4f}  "
                  f"max_window={mx:8.1f} tokens")
            if islive:
                classification[(bi, h)] = mx
        print()

    print("==================== CLASSIFICATION (LIVE heads only) ====================")
    for (bi, h), mx in sorted(classification.items()):
        nm = names[bi] if names and bi < len(names) else "?"
        kind = "LOCAL " if mx <= 60 else "GLOBAL"
        print(f"  block {bi:3d} [{nm:>16}] head {h:2d} [{_role(h):>10}] "
              f"window={mx:8.1f}  -> {kind}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
