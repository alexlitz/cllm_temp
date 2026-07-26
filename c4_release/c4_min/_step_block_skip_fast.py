"""STEP 1/2 (fast, definitive) — DECODE-safe per-op block-skip map.

KEY INSIGHT (validated): the c4_min pure-forward driver runs each VM step as ONE
full model.forward over the growing token stream, starting FRESH from
``model.embed[toks]`` every step.  NOTHING of the residual survives a step except
the EMITTED TOKEN FRAME (the decoded PC/AX/SP/BP/STK the driver appends).  So a
block whose write does not change the DECODED registers of the step it runs in is
byte-exact-skippable for that opcode — the query-row scratch it scribbles is
thrown away when the next step re-embeds the stream.

This module FREEZES the target-step input once (embed+overlay at the op's PC),
then for each block runs the frozen forward WITH and WITHOUT that block and checks
whether the decoded registers change.  ~238 forwards/op instead of ~1000, and no
driver re-stepping between ablations -> fast enough for all 31 opcodes on GPU.

Reports the DECODE-safe live-block count per opcode (the true STEP-2 skip win) and
the per-op live SET (for building the static skip schedule).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from . import isa
from . import nibble_pure_forward_complete as pfc
from .nibble_pure_forward_complete import (
    make_overlay_complete, _build_frame, SP_INIT,
)
from ._step_block_skip_probe import _op_programs, _target_pc


def _freeze_target_input(model, L, prog_named, seed_mem):
    """Run the driver up to (not through) the op-under-test step and return the
    embedded+overlaid input x for THAT step (before any block runs)."""
    from .nibble_pure_forward_complete import _seed_frames, _mem_top
    code = isa.assemble(prog_named)
    target = _target_pc(prog_named)
    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    frame_idx = n_seed
    for _ in range(64):
        overlay = make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream], device=model.embed.device)
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            if cur_pc == target:
                return x            # frozen input for the target step
            for blk in model.blocks:
                x = blk(x)
        state = x[0, -1]
        pc = int(round(float(state[L.PC_VAL])))
        sp = int(round(float(state[L.SP_VAL])))
        bp = int(round(float(state[L.BP_VAL])))
        stk = int(round(float(state[L.STK_VAL])))
        ax = pfc._decode_reg_from_nibbles(state, L, L.AX)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        s_addr = s_val = 0; is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & 0xFF
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & 0xFF
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        cur_pc, cur_sp, cur_bp = pc, sp, bp
        if pc < 0 or pc >= len(code):
            break
    return None


def _run_frozen(model, x0, skip: Optional[int]):
    """Run all blocks on frozen input x0, optionally skipping ``skip``. Return the
    query-row state."""
    x = x0
    with torch.no_grad():
        for bi, blk in enumerate(model.blocks):
            if skip is not None and bi == skip:
                continue
            x = blk(x)
    return x[0, -1]


def _ablate_all(model, L, x0, base_regs):
    """Prefix-cached single-block ablation over ALL blocks.

    Run the full stack once, caching the INPUT to every block (the prefix), then
    for each block bi resume from its cached input, SKIP bi, and run bi+1..end.
    Each ablation costs only (nb-bi) blocks; the prefix is shared -> ~2x fewer
    block-applies than nb independent full forwards.  Returns the DECODE-live set
    (blocks whose skip changes the decoded registers)."""
    nb = len(model.blocks)
    inputs = [None] * nb            # inputs[bi] = residual entering block bi
    x = x0
    with torch.no_grad():
        for bi, blk in enumerate(model.blocks):
            inputs[bi] = x
            x = blk(x)
    live = []
    with torch.no_grad():
        for bi in range(nb):
            x = inputs[bi]          # residual entering block bi
            for bj in range(bi + 1, nb):   # skip bi, run the rest
                x = model.blocks[bj](x)
            if _decode_regs(x[0, -1], L) != base_regs:
                live.append(bi)
    return live


def _decode_regs(state, L):
    """The driver's observable output for a step: the 5 decoded registers + HALTED.

    The scalar lanes (PC/SP/BP/STK) hold EXACT integers up to O(1e-6) fp residue,
    so ``_snap_lane``'s argmax (== floor(x+½) for the value vocab) equals the
    integer round here — we use the fast round and separately CROSS-CHECK the base
    against the true ``_snap_lane`` in ``main`` so the equivalence is proven, not
    assumed.  AX is decoded via the SAME per-byte nibble argmax the driver uses
    (residue-immune).  Identical tuple => the emitted frame is byte-identical."""
    sc = state.cpu()
    return (
        int(round(float(sc[L.PC_VAL]))),
        pfc._decode_reg_from_nibbles(sc, L, L.AX),
        int(round(float(sc[L.SP_VAL]))),
        int(round(float(sc[L.BP_VAL]))),
        int(round(float(sc[L.STK_VAL]))),
        float(sc[L.HALTED]) > 0.5,
    )


def _decode_regs_exact(state, L):
    """The TRUE driver decode (``_snap_lane`` argmax) — used to cross-check that the
    fast ``_decode_regs`` agrees on the base state (proves the round equivalence)."""
    from .nibble_vm import _snap_lane
    sc = state.cpu()
    return (
        _snap_lane(sc[L.PC_VAL]), pfc._decode_reg_from_nibbles(sc, L, L.AX),
        _snap_lane(sc[L.SP_VAL]), _snap_lane(sc[L.BP_VAL]),
        _snap_lane(sc[L.STK_VAL]), float(sc[L.HALTED]) > 0.5,
    )


def main(code_size: int = 32, ops_subset=None, device: str = "cuda"):
    from collections import Counter
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device); model.materialize_dense(device)
    names = list(getattr(L, "_block_names", []))
    nb = len(model.blocks)
    print(f"[built] n_blocks={nb} dim={model.dim} dev={device} "
          f"build={time.time()-t0:.1f}s", flush=True)

    live_sets: Dict[str, List[int]] = {}
    for opname, prog, seed in _op_programs():
        if ops_subset is not None and opname not in ops_subset:
            continue
        t1 = time.time()
        x0 = _freeze_target_input(model, L, prog, seed)
        if x0 is None:
            print(f"  {opname:5s} (never reached)", flush=True)
            continue
        base = _run_frozen(model, x0, skip=None)
        base_regs = _decode_regs(base, L)
        assert base_regs == _decode_regs_exact(base, L), \
            f"{opname}: fast round != true _snap_lane on base {base_regs}"
        live = _ablate_all(model, L, x0, base_regs)
        live_sets[opname] = live
        print(f"  {opname:5s} DECODE-live={len(live):3d}/{nb} "
              f"({time.time()-t1:.1f}s)  {[names[i] for i in live]}", flush=True)

    if live_sets:
        counts = {k: len(v) for k, v in live_sets.items()}
        cs = sorted(counts.values())
        print(f"\n[summary] DECODE-safe live-block count (blocks a step MUST run):",
              flush=True)
        for k in sorted(counts, key=lambda z: counts[z]):
            print(f"    {k:5s} {counts[k]:3d}", flush=True)
        print(f"  min={cs[0]} max={cs[-1]} mean={sum(cs)/len(cs):.1f} "
              f"median={cs[len(cs)//2]}", flush=True)
        union = sorted(set().union(*[set(v) for v in live_sets.values()]))
        print(f"  union-live (some op needs) = {len(union)}/{nb}", flush=True)
        freq = Counter()
        for v in live_sets.values():
            freq.update(v)
        print(f"  blocks live for MANY ops (shared, small skip win):", flush=True)
        for b, c in sorted(freq.items(), key=lambda x: -x[1])[:30]:
            print(f"    block {b:3d} {names[b]:22s} live in {c}/{len(live_sets)}",
                  flush=True)
    return live_sets, names


if __name__ == "__main__":
    import sys
    dev = "cpu" if "--cpu" in sys.argv else "cuda"
    subset = None
    for a in sys.argv:
        if a.startswith("--ops="):
            subset = set(a.split("=", 1)[1].split(","))
    main(ops_subset=subset, device=dev)
