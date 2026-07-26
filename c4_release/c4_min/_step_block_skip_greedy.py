"""STEP 1/2 (SOUND) — greedy minimal per-op live set via CUMULATIVE skip.

The single-block ablation is UNSOUND for a CHAIN: two blocks can each be
individually skippable (a later block masks the first's absence) yet NOT jointly
skippable.  IMM is the canonical case — ``ax-byte-nib`` and ``imm-ax-nib`` both
write AX nibbles, so dropping EITHER alone leaves AX=42, but dropping BOTH gives
AX=0.  The correct minimal live set is found by CUMULATIVE removal: keep a running
skip set, try to add each block to it, and KEEP it in the skip set only if the
op's decoded output is STILL byte-exact with ALL currently-skipped blocks removed
TOGETHER.  What remains un-skipped is the sound minimal live set.

This is the set STEP 2 must run; it is byte-exact BY CONSTRUCTION for the frozen
target step (and re-verified end-to-end through the real driver).
"""
from __future__ import annotations

import time
from typing import Dict, List

import torch

from . import isa
from ._step_block_skip_fast import _freeze_target_input, _decode_regs
from ._step_block_skip_probe import _op_programs


def _run_with_skip(model, x0, skip_set):
    x = x0
    with torch.no_grad():
        for bi, blk in enumerate(model.blocks):
            if bi in skip_set:
                continue
            x = blk(x)
    return x[0, -1]


def greedy_live(model, L, x0, base_regs):
    """Cumulative greedy: grow a skip set, keeping a block skipped only if the
    decode stays byte-exact with the WHOLE current skip set removed together.
    Returns the sound minimal LIVE set (blocks NOT in the final skip set)."""
    nb = len(model.blocks)
    skip = set()
    for bi in range(nb):
        trial = skip | {bi}
        st = _run_with_skip(model, x0, trial)
        if _decode_regs(st, L) == base_regs:
            skip = trial                     # safe to keep skipping (cumulatively)
    live = [i for i in range(nb) if i not in skip]
    return live


def main(code_size: int = 32, device: str = "cuda", ops_subset=None):
    from collections import Counter
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device); model.materialize_dense(device)
    names = list(getattr(L, "_block_names", []))
    nb = len(model.blocks)
    print(f"[built] n_blocks={nb} dev={device} build={time.time()-t0:.1f}s",
          flush=True)

    live_sets: Dict[str, List[int]] = {}
    for opname, prog, seed in _op_programs():
        if ops_subset is not None and opname not in ops_subset:
            continue
        t1 = time.time()
        x0 = _freeze_target_input(model, L, prog, seed)
        if x0 is None:
            print(f"  {opname:5s} (never reached)", flush=True)
            continue
        base = _run_with_skip(model, x0, set())
        base_regs = _decode_regs(base, L)
        live = greedy_live(model, L, x0, base_regs)
        live_sets[opname] = live
        print(f"  {opname:5s} SOUND-live={len(live):3d}/{nb} ({time.time()-t1:.1f}s)"
              f"  {[names[i] for i in live]}", flush=True)

    if live_sets:
        cs = sorted(len(v) for v in live_sets.values())
        print(f"\n[summary] SOUND minimal live count: min={cs[0]} max={cs[-1]} "
              f"mean={sum(cs)/len(cs):.1f} median={cs[len(cs)//2]}", flush=True)
        union = sorted(set().union(*[set(v) for v in live_sets.values()]))
        print(f"  union-live (some op) = {len(union)}/{nb}", flush=True)
        freq = Counter()
        for v in live_sets.values():
            freq.update(v)
        print(f"  shared blocks (live for many ops):", flush=True)
        for b, c in sorted(freq.items(), key=lambda x: -x[1])[:20]:
            print(f"    {b:3d} {names[b]:22s} {c}/{len(live_sets)}", flush=True)
    # emit a python dict of the NAME schedule for step_block_skip.py
    print("\n[schedule] per-op live NAMES:", flush=True)
    for opname, live in live_sets.items():
        print(f"    {opname}: {[names[i] for i in live]}", flush=True)
    return live_sets, names


if __name__ == "__main__":
    import sys
    dev = "cpu" if "--cpu" in sys.argv else "cuda"
    subset = None
    for a in sys.argv:
        if a.startswith("--ops="):
            subset = set(a.split("=", 1)[1].split(","))
    main(device=dev, ops_subset=subset)
