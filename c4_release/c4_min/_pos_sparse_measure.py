"""STEP 1 — measure the POSITION-LEVEL redundancy of the (op-level-live) blocks.

The op-level block-skip (``step_block_skip.py``) runs each LIVE block over ALL S
positions of the growing token stream, but the driver only ever DECODES the query
row (``x[0, -1]``).  A VM step's emitted frame is 30 tokens (and the stream holds
every prior frame too), yet the heavy ALU / divmod work is needed only at the ~few
positions the query row's decode actually depends on.

This module answers the user's exact question: for each block that runs, how many
token POSITIONS does its output actually AFFECT (change the decoded registers of
the query row), vs positions where it is a nil passthrough (masking it there leaves
the decode byte-identical)?

METHOD (decode-faithful, byte-exact truth): freeze the op's target-step input, run
the op's OP-LEVEL-LIVE blocks (from ``step_block_skip.build_live_index`` — the same
schedule that is already verified byte-exact) to get the base decoded registers.
Then for each live block bi (the FFN is POSITION-INDEPENDENT — each position's
SwiGLU depends only on that position's residual):

  build a BATCH of (S+1) copies of the residual leaving bi: row 0 = FFN fully on;
  rows 1..S = the FFN write reverted to its passthrough at exactly ONE position
  (SwiGLU ``FFN.forward`` returns ``a + W_down(...)``, so the passthrough IS the
  FFN input residual ``a`` — masking a position == 'FFN not computed there').  Run
  the op's remaining live blocks ONCE as a batch; a row whose query-row decode
  differs from base marks that position ACTIVE.

We report, per block:
  * active_positions  = how many of ALL S stream positions the FFN must compute at
  * redundancy = S / active_positions  (positions computed / positions needed)
and, for DIV/MOD, the aggregate over the divmod span.
"""
from __future__ import annotations

import time
from typing import List, Optional

import torch

from ._step_block_skip_fast import _freeze_target_input, _decode_regs
from ._step_block_skip_probe import _op_programs, _target_pc
from . import isa


def _run_live(model, x, live_after, start_after):
    """Run the op's live blocks with index > start_after (in order)."""
    with torch.no_grad():
        for bi in live_after:
            if bi > start_after:
                x = model.blocks[bi](x)
    return x


def measure_block_positions(model, L, x_in, base_regs, block_idx, live_list, S,
                            scan_lo=0):
    """Which stream positions [scan_lo,S) must block_idx's FFN compute at?

    BATCHED over positions: one suffix run of (P+1) rows."""
    blk = model.blocks[block_idx]
    live_after = [b for b in live_list if b > block_idx]
    with torch.no_grad():
        a = blk.attn.forward(x_in)
        full = blk.ffn.forward(a) if not blk._routed else blk.ffn(a)
        scan_pos = list(range(scan_lo, S))
        P = len(scan_pos)
        batch = full.expand(P + 1, S, full.shape[-1]).clone()
        for k, p in enumerate(scan_pos):
            batch[k + 1, p] = a[0, p]                 # FFN OFF at p (passthrough)
        with torch.no_grad():
            x = batch
            for bi in live_after:
                x = model.blocks[bi](x)
        active = []
        for k, p in enumerate(scan_pos):
            if _decode_regs(x[k + 1, -1], L) != base_regs:
                active.append(p)
    return active


def main(code_size: int = 32, device: str = "cpu",
         ops_subset: Optional[set] = None, frame_len: int = 30):
    from .compact_alloc import build_compact_sparse_streaming
    from .step_block_skip import build_live_index
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device); model.materialize_dense(device)
    names = list(getattr(L, "_block_names", []))
    nb = len(model.blocks)
    div_idx = [i for i, n in enumerate(names) if n.startswith("alu-div")]
    div_span = (min(div_idx), max(div_idx)) if div_idx else None
    live_index = build_live_index(model, L)
    print(f"[built] n_blocks={nb} dim={model.dim} dev={device} "
          f"div_span={div_span} build={time.time()-t0:.1f}s", flush=True)

    focus = ops_subset or {"ADD", "SUB", "MUL", "DIV", "MOD", "IMM", "PSH", "JMP"}
    for opname, prog, seed in _op_programs():
        if opname not in focus:
            continue
        op = isa.assemble(prog)[_target_pc(prog)].op
        live_list = live_index.get(op, live_index[None])
        t1 = time.time()
        x0 = _freeze_target_input(model, L, prog, seed)
        if x0 is None:
            print(f"  {opname:5s} (never reached)", flush=True)
            continue
        if device != "cpu":
            x0 = x0.to(device)
        S = x0.shape[1]
        frame_lo = max(0, S - frame_len)
        # base decode via the op's OP-LEVEL-LIVE schedule (verified byte-exact).
        with torch.no_grad():
            xb = x0
            for bi in live_list:
                xb = model.blocks[bi](xb)
        base_regs = _decode_regs(xb[0, -1], L)
        # sanity: full 238-block decode must match the op-level-live decode.
        with torch.no_grad():
            xf = x0
            for blk in model.blocks:
                xf = blk(xf)
        assert _decode_regs(xf[0, -1], L) == base_regs, \
            f"{opname}: op-level-live decode != full decode"
        print(f"\n=== {opname}  S={S} frame=[{frame_lo},{S}) "
              f"live_blocks={len(live_list)}  base={base_regs}", flush=True)
        # prefix cache: residual entering each live block.
        inputs = {}
        with torch.no_grad():
            x = x0
            prev = -1
            for bi in live_list:
                # advance is implicit: we always feed x0 through live blocks in order
                inputs[bi] = x
                x = model.blocks[bi](x)
        rows = []
        for bi in live_list:
            active = measure_block_positions(
                model, L, inputs[bi], base_regs, bi, live_list, S, scan_lo=0)
            in_frame = [p for p in active if p >= frame_lo]
            prior = [p for p in active if p < frame_lo]
            red = S / max(1, len(active))
            rows.append((bi, names[bi], len(active), len(in_frame),
                         len(prior), red, active))
        for bi, nm, na, nf, npr, red, active in rows:
            in_div = bool(div_span) and div_span[0] <= bi <= div_span[1]
            tag = "DIVMOD" if in_div else ""
            rel = [p - frame_lo for p in active if p >= frame_lo]
            print(f"    blk {bi:3d} {nm:24s} active={na:2d}/{S} "
                  f"(frame={nf} prior={npr}) redund={red:6.1f}x {tag}  "
                  f"frame_rel={rel}", flush=True)
        if div_span:
            div_live = [r for r in rows if div_span[0] <= r[0] <= div_span[1]]
            if div_live:
                tot_active = sum(r[2] for r in div_live)
                dense = len(div_live) * S
                print(f"  [DIVMOD span] {len(div_live)} live div-blocks over S={S}: "
                      f"dense_positions={dense}, active_positions={tot_active}, "
                      f"redundancy={dense/max(1,tot_active):.1f}x", flush=True)
        print(f"  ({time.time()-t1:.1f}s)", flush=True)


if __name__ == "__main__":
    import sys
    dev = "cuda" if "--cuda" in sys.argv else "cpu"
    subset = None
    for a in sys.argv:
        if a.startswith("--ops="):
            subset = set(a.split("=", 1)[1].split(","))
    main(device=dev, ops_subset=subset)
