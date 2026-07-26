"""STEP 1 — opcode -> live-block map for the c4_min pure-forward VM.

For each opcode, run ONE model.forward of a single-op program (with the minimal
operand setup that op needs on the stack/AX), and diff the residual x BEFORE and
AFTER every block.  A block whose attn+FFN leave the residual BIT-UNCHANGED at
every position is a no-op (skippable) for that opcode; a block that writes any
dim is LIVE.

Reports the live-block-COUNT distribution per opcode and the union / per-op live
sets, so STEP 2 can drive a per-step block skip from the decoded opcode.

Run: PYTHONPATH=<c4_release> OMP_NUM_THREADS=4 python -m c4_min._step_block_skip_probe
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch

from . import isa
from . import nibble_pure_forward_complete as pfc
from .nibble_pure_forward_complete import (
    make_overlay_complete, _build_frame, SP_INIT,
)


def _op_programs() -> List[Tuple[str, List[Tuple[str, int]], Dict[int, int]]]:
    """A single-op battery: (opcode_name, program, seed_mem).

    Each program's LAST instruction is the op under test; the ones before it set
    up operand(s).  We probe ONLY the step whose PC lands on the op-under-test."""
    P: List[Tuple[str, List, Dict]] = []
    P.append(("IMM", [("IMM", 42)], {}))
    P.append(("LEA", [("LEA", 3)], {}))
    P.append(("JMP", [("JMP", 0)], {}))
    P.append(("PSH", [("IMM", 7), ("PSH", 0)], {}))
    P.append(("BZ", [("IMM", 0), ("BZ", 0)], {}))
    P.append(("BNZ", [("IMM", 1), ("BNZ", 0)], {}))
    for nm in ("ADD", "SUB", "MUL", "DIV", "MOD",
               "OR", "XOR", "AND", "SHL", "SHR",
               "EQ", "NE", "LT", "GT", "LE", "GE"):
        P.append((nm, [("IMM", 12), ("PSH", 0), ("IMM", 3), (nm, 0)], {}))
    P.append(("LI", [("IMM", 8), ("LI", 0)], {8: 99}))
    P.append(("LC", [("IMM", 8), ("LC", 0)], {8: 77}))
    P.append(("SI", [("IMM", 8), ("PSH", 0), ("IMM", 55), ("SI", 0)], {}))
    P.append(("SC", [("IMM", 8), ("PSH", 0), ("IMM", 55), ("SC", 0)], {}))
    P.append(("JSR", [("JSR", 0)], {}))
    P.append(("ENT", [("ENT", 1)], {}))
    P.append(("ADJ", [("ADJ", 1)], {}))
    P.append(("LEV", [("LEV", 0)], {}))
    P.append(("HALT", [("HALT", 0)], {}))
    return P


def _target_pc(prog):
    return len(prog) - 1


def _run_step_probe(model, L, prog_named, seed_mem):
    """Advance the driver to the target op's step, then run its forward capturing
    the residual before/after every block.  Returns per-block max-abs delta."""
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
        toks = torch.tensor([stream])
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            if cur_pc == target:
                d = []
                for blk in model.blocks:
                    x_in = x
                    x = blk(x)
                    d.append((x - x_in).abs().max().item())
                return d
            for blk in model.blocks:
                x = blk(x)
        state = x[0, -1]
        pc = int(round(float(state[L.PC_VAL])))
        sp = int(round(float(state[L.SP_VAL])))
        bp = int(round(float(state[L.BP_VAL])))
        stk = int(round(float(state[L.STK_VAL])))
        ax = pfc._decode_reg_from_nibbles(state, L, L.AX)
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        s_addr = s_val = 0
        is_store = False
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


def main(code_size: int = 32, eps: float = 0.0, verbose: bool = True):
    from collections import Counter
    from .compact_alloc import build_compact_sparse_streaming
    t0 = time.time()
    model, L, _ = build_compact_sparse_streaming(
        code_size=code_size, compute_mode="dense_kernel")
    names = list(getattr(L, "_block_names", []))
    nb = len(model.blocks)
    if verbose:
        print(f"[built] n_blocks={nb} dim={model.dim} "
              f"build={time.time()-t0:.1f}s", flush=True)

    live_sets: Dict[str, List[int]] = {}
    counts: Dict[str, int] = {}
    all_deltas: Dict[str, List[float]] = {}
    for opname, prog, seed in _op_programs():
        try:
            deltas = _run_step_probe(model, L, prog, seed)
        except Exception as e:
            print(f"  {opname:5s} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        if deltas is None:
            print(f"  {opname:5s} (target step never reached)", flush=True)
            continue
        live = [i for i, dd in enumerate(deltas) if dd > eps]
        live_sets[opname] = live
        counts[opname] = len(live)
        all_deltas[opname] = deltas
        if verbose:
            liven = [names[i] if i < len(names) else f"b{i}" for i in live[:24]]
            print(f"  {opname:5s} live={len(live):3d}/{nb}  {liven}"
                  + (" ..." if len(live) > 24 else ""), flush=True)

    union = sorted(set().union(*[set(v) for v in live_sets.values()])) if live_sets else []
    never_live = [i for i in range(nb) if i not in set(union)]
    if verbose:
        print(f"\n[summary] live-count distribution:", flush=True)
        for opname in sorted(counts, key=lambda k: counts[k]):
            print(f"    {opname:5s} {counts[opname]:3d}", flush=True)
        cs = sorted(counts.values())
        print(f"  min={cs[0]} max={cs[-1]} mean={sum(cs)/len(cs):.1f} "
              f"median={cs[len(cs)//2]}", flush=True)
        print(f"  union-live (some op) = {len(union)}/{nb}; "
              f"NEVER-live (dead all ops) = {len(never_live)}", flush=True)
        freq = Counter()
        for v in live_sets.values():
            freq.update(v)
        shared = sorted([(b, c) for b, c in freq.items()], key=lambda x: -x[1])
        print(f"  blocks live for MANY opcodes (top 30 by op-count):", flush=True)
        for b, c in shared[:30]:
            print(f"    block {b:3d} {names[b] if b < len(names) else '':22s} "
                  f"live in {c}/{len(live_sets)} ops", flush=True)
    return live_sets, counts, all_deltas, names, never_live


if __name__ == "__main__":
    main()
