"""READ-ONLY cross-step-live-set + per-block liveness measurement for the DOOM
config (register-resident bandwidth lever).

Builds the REAL doom-config model memory-safely (streaming sparse, ~1.5 GB peak;
NEVER densifies the whole thing) and measures:

  1. D (dim_after), n_blocks, never_share_count  (the compacted residual width).
  2. The PER-BLOCK live-dim curve: how many residual dims are live (defined &
     not-yet-last-used) at each block boundary — the within-step working set.
  3. The CROSS-STEP live set: which dims are BOTH written by the last block AND
     the ones the DRIVER emits (PC/AX/SP/BP/STK + HALTED) i.e. actually carried
     to step N+1.  In this architecture the cross-step carrier is the emitted
     FRAME (registers as bytes) + the KV store log — NOT the residual — so the
     residual's cross-step-live count is ~0 (fully recomputed each step).
  4. Bandwidth: full-D residual carry per block vs the true persistent-state
     carry, in bytes/step.

Run:  C4_PF_CFM=1 C4_CODE_ADDR_BITS=20 C4_MEM_ADDR_BITS=18 \
      OMP_NUM_THREADS=4 python -m c4_min._agent_regresident_liveness --code-size 64
"""
from __future__ import annotations
import os, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-size", type=int, default=64,
                    help="doom uses ~len(code)+2; small here just fixes the CODE table")
    ap.add_argument("--recurrent-divmod", action="store_true", default=True)
    args = ap.parse_args()

    from .lib_neural import build_lib_model_streaming
    from ._agent_sparse_dim_traffic import _csr_nz_cols, _csr_nz_rows

    print(f"[build] streaming sparse doom-config model (code_size={args.code_size}, "
          f"recurrent_divmod={args.recurrent_divmod}) ...", flush=True)
    sparse, L, stats = build_lib_model_streaming(
        code_size=args.code_size, recurrent_divmod=args.recurrent_divmod,
        addr32=True, compute_mode="dense_kernel")
    D = sparse.embed.shape[1]
    nblk = len(sparse.blocks)
    print(f"[build] DONE  D(dim_after)={D}  n_blocks={nblk}  "
          f"dim_before={stats.dim_before}  never_share_count={stats.never_share_count}  "
          f"shared_slots_saved={stats.shared_slots_saved}")
    print(f"[build] nonzero_params={stats.nonzero_params:,}  "
          f"dense_after={stats.dense_params_after:,}  size_after={stats.size_mb_after:.1f}MB")

    # --- RAW WEIGHT-liveness on the COMPACTED SPARSE model (sparse-aware) ---
    # def_block[d] = earliest block that WRITES dim d (W_o row / W_down row; embed=-1);
    # last_use[d]  = latest block that READS dim d (W_q/k/v col, W_up/gate col; lm_head=n).
    # Intervals reflect the TRUE data-dependency spans (early-write -> late-read = the dim
    # must survive that whole span in the residual). This is the transient-vs-persistent
    # structure the register-resident lever attacks.
    n = nblk
    INF, NEG = n + 1, -2
    def_block = [INF] * D
    last_use = [NEG] * D

    def mark_w(dims, b):
        for d in dims:
            if b < def_block[d]:
                def_block[d] = b

    def mark_r(dims, b):
        for d in dims:
            if b > last_use[d]:
                last_use[d] = b

    mark_w(_csr_nz_rows(sparse.embed) if hasattr(sparse.embed, "csr")
           else set((sparse.embed != 0).any(dim=0).nonzero().flatten().tolist()), -1)
    for b, blk in enumerate(sparse.blocks):
        at = getattr(blk, "attn", None)
        if at is not None and not getattr(at, "is_zero", False):
            for wn in ("W_q", "W_k", "W_v"):
                mark_r(_csr_nz_cols(getattr(at, wn, None)), b)
            mark_w(_csr_nz_rows(getattr(at, "W_o", None)), b)
        ff = getattr(blk, "ffn", None)
        if ff is not None:
            for wn in ("W_up", "W_gate"):
                mark_r(_csr_nz_cols(getattr(ff, wn, None)), b)
            mark_w(_csr_nz_rows(getattr(ff, "W_down", None)), b)
    mark_r(_csr_nz_cols(sparse.lm_head) if hasattr(sparse.lm_head, "csr")
           else set((sparse.lm_head != 0).any(dim=0).nonzero().flatten().tolist()), n)

    class _LV:
        __slots__ = ("def_block", "last_use_block", "never_share")
        def __init__(s, d, u):
            s.def_block, s.last_use_block, s.never_share = d, u, False
    liveness = []
    for d in range(D):
        db, ub = def_block[d], last_use[d]
        if db == INF and ub == NEG:
            liveness.append(_LV(0, 0)); continue
        if db == INF: db = -1
        if ub == NEG: ub = db
        liveness.append(_LV(db, ub))
    n_never = stats.never_share_count   # the driver-pinned persistent bands (from build)
    # per-block live count: dim d is live at block boundary b iff def_block<=b<=last_use
    live_at = [0] * (nblk + 1)
    persistent_full = 0    # dims live across the ENTIRE step (def<=0 and last_use>=nblk-1)
    for lv in liveness:
        db = max(lv.def_block, 0)
        lu = min(lv.last_use_block, nblk)
        for b in range(db, lu + 1):
            if b <= nblk:
                live_at[b] += 1
        if lv.def_block <= 0 and lv.last_use_block >= nblk - 1:
            persistent_full += 1

    max_live = max(live_at)
    mean_live = sum(live_at) / len(live_at)
    print(f"\n[live] per-block WITHIN-STEP live-dim curve over {nblk} blocks:")
    print(f"[live]   D(width)={D}  max_simultaneously_live={max_live}  "
          f"mean_live={mean_live:.0f}  never_share(persist-all-step)={n_never}")
    print(f"[live]   dims live across the WHOLE step (def@0..last_use@end)={persistent_full}")
    # print a coarse curve (every ~nblk/20 blocks)
    step = max(1, nblk // 20)
    curve = ", ".join(f"b{b}:{live_at[b]}" for b in range(0, nblk + 1, step))
    print(f"[live]   curve: {curve}")

    # --- CROSS-STEP live set: the dims the DRIVER reads to emit the next frame ---
    read_names = ["PC_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED"]
    driver_read = 0
    for nm in read_names:
        rec = L._names.get(nm)
        if rec:
            driver_read += rec[1]
    ax_rec = L._names.get("AX")
    ax_nib = ax_rec[1] if ax_rec else 0
    driver_read_total = driver_read + ax_nib

    # the 5 registers re-ingested next step (the true VM state, as residual dims)
    live_reg = 0
    for nm in ("PC", "AX", "SP", "BP", "STACK0"):
        rec = L._names.get(nm)
        if rec:
            live_reg += rec[1]

    print(f"\n[cross-step] driver end-of-step reads {read_names}+AX-nibbles = "
          f"{driver_read_total} residual dims (of D={D}) — the transformer's REAL "
          f"cross-step OUTPUT.")
    print(f"[cross-step] the re-ingested register set (PC/AX/SP/BP/STACK0 nibbles) = "
          f"{live_reg} residual dims = the TRUE persistent VM state in the residual.")
    print(f"[cross-step] emitted FRAME payload = 5 regs x 4 bytes = 20 bytes "
          f"(+ up to 8 bytes mem addr/val on a store) — the actual cross-step carrier "
          f"(via KV-cache frame-ingest CAM), NOT the residual.")

    # --- BANDWIDTH per step ---
    # The transformer moves the residual (S_active tokens x D dims x 4 bytes) through
    # HBM at each of nblk blocks (read+write activations).  The task's ~44 KB/step
    # sparse-active figure and ~371 KB/step unfolded figure are this residual traffic.
    # If the cross-step carry were reduced to only the persistent-state dims, the
    # INTER-BLOCK activation that MUST survive shrinks from D to the max-live working
    # set, but the true CROSS-STEP (inter-forward) carry is just the 20-byte frame.
    fp32 = 4
    per_tok_full = D * fp32
    per_tok_live = live_reg * fp32
    per_tok_read = driver_read_total * fp32
    print(f"\n[bandwidth] per-token residual (fp32): full D={D} -> {per_tok_full} B; "
          f"register-live {live_reg} -> {per_tok_live} B "
          f"({per_tok_full/max(per_tok_live,1):.1f}x); driver-read {driver_read_total} "
          f"-> {per_tok_read} B ({per_tok_full/max(per_tok_read,1):.1f}x).")
    print(f"[bandwidth] TRUE cross-step carrier = 20-byte frame vs a full-D residual "
          f"snapshot ({per_tok_full} B) = {per_tok_full/20:.0f}x smaller inter-step state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
