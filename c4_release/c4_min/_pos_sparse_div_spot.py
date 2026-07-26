"""Fast spot-check: for DIV, confirm a SAMPLE of divmod-span blocks are active
ONLY at the query row (redundancy = S), without measuring all 186 blocks.

Picks the first / a few interior / the last blocks of the alu-div span and
measures each one's per-position active set (batched over all S positions).
"""
from __future__ import annotations

import time

import torch

from ._step_block_skip_fast import _freeze_target_input, _decode_regs
from ._step_block_skip_probe import _op_programs, _target_pc
from .step_block_skip import build_live_index
from ._pos_sparse_measure import measure_block_positions
from . import isa


def main(code_size: int = 32):
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    names = list(getattr(L, "_block_names", []))
    div_idx = [i for i, n in enumerate(names) if n.startswith("alu-div")]
    live_index = build_live_index(model, L)
    print(f"[built] div_span=({min(div_idx)},{max(div_idx)}) "
          f"n_div_blocks={len(div_idx)} build={time.time()-t0:.1f}s", flush=True)

    prog = [("IMM", 12), ("PSH", 0), ("IMM", 3), ("DIV", 0)]
    seed = {}
    op = isa.DIV
    live_list = live_index[op]
    x0 = _freeze_target_input(model, L, prog, seed)
    S = x0.shape[1]
    with torch.no_grad():
        xb = x0
        for bi in live_list:
            xb = model.blocks[bi](xb)
    base_regs = _decode_regs(xb[0, -1], L)
    print(f"DIV S={S} live_blocks={len(live_list)} base={base_regs}", flush=True)
    # prefix cache the residual entering each live block
    inputs = {}
    with torch.no_grad():
        x = x0
        for bi in live_list:
            inputs[bi] = x
            x = model.blocks[bi](x)
    # sample: first 4, 4 interior, last 4 divmod-span blocks (that are live)
    live_div = [b for b in live_list if min(div_idx) <= b <= max(div_idx)]
    n = len(live_div)
    sample = sorted(set(live_div[:4] + live_div[n//2-2:n//2+2] + live_div[-4:]))
    print(f"[sample] {len(sample)} divmod-span blocks of {n} live: {sample}",
          flush=True)
    for bi in sample:
        t1 = time.time()
        active = measure_block_positions(
            model, L, inputs[bi], base_regs, bi, live_list, S, scan_lo=0)
        frame_lo = S - 30
        rel = [p - frame_lo for p in active if p >= frame_lo]
        red = S / max(1, len(active))
        print(f"  blk {bi:3d} {names[bi]:24s} active={len(active):2d}/{S} "
              f"redund={red:6.1f}x frame_rel={rel} ({time.time()-t1:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
