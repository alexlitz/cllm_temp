"""STEP 1/2 (SOUND + OPERAND-UNION) — the byte-exact static per-op live set.

The single-operand greedy set is sound for THAT operand but not a general static
schedule: compares fire different blocks for TRUE vs FALSE, branches for taken vs
not-taken, bitwise/ALU for different bit patterns, DIV/MOD per dividend nibble.
The byte-exact STATIC schedule is the UNION of the cumulative-greedy sound set
over an OPERAND BATTERY per opcode.  This module computes it and emits the schedule.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch

from . import isa
from ._step_block_skip_fast import _freeze_target_input, _decode_regs
from ._step_block_skip_greedy import _run_with_skip, greedy_live


def _op_batteries() -> Dict[str, List[Tuple[list, dict]]]:
    """Per-opcode operand battery covering the result-shape corners."""
    B: Dict[str, List[Tuple[list, dict]]] = {}

    def bin_cases(nm, pairs):
        return [([("IMM", a), ("PSH", 0), ("IMM", b), (nm, 0)], {}) for a, b in pairs]

    B["IMM"] = [([("IMM", v)], {}) for v in (0, 42, 255, 10000, 0xFFFFFF)]
    B["LEA"] = [([("LEA", v)], {}) for v in (0, 3, 15, 100)]
    B["JMP"] = [([("JMP", 0)], {})]
    B["PSH"] = [([("IMM", v), ("PSH", 0)], {}) for v in (0, 7, 255)]
    # branch battery: taken/not-taken x several TARGETS (imm), since the branch
    # target (imm) flows into PC and can change which blocks the decode needs.
    B["BZ"]  = [([("IMM", z), ("BZ", t)], {}) for z in (0, 5) for t in (0, 4, 9)]
    B["BNZ"] = [([("IMM", z), ("BNZ", t)], {}) for z in (0, 5) for t in (0, 4, 9)]
    B["ADD"] = bin_cases("ADD", [(12, 3), (200, 100), (255, 255), (0, 0), (1, 254)])
    B["SUB"] = bin_cases("SUB", [(12, 3), (3, 12), (255, 0), (100, 100), (0, 1)])
    B["MUL"] = bin_cases("MUL", [(12, 3), (7, 0), (255, 255), (16, 16), (1, 200)])
    B["OR"]  = bin_cases("OR", [(0xF0, 0x0F), (0, 0), (0xFF, 0), (0xAA, 0x55),
                                (0x12, 0x34)])
    B["XOR"] = bin_cases("XOR", [(0xF0, 0x0F), (0xFF, 0xFF), (0, 0xFF), (0xAA, 0x55),
                                 (0x12, 0x34)])
    B["AND"] = bin_cases("AND", [(0xF0, 0x3C), (0xFF, 0xFF), (0, 0xFF), (0xAA, 0x55),
                                 (0x12, 0x34)])
    B["SHL"] = bin_cases("SHL", [(3, 4), (1, 0), (1, 7), (0xFF, 1), (5, 3)])
    B["SHR"] = bin_cases("SHR", [(240, 4), (1, 0), (0xFF, 7), (0x80, 1), (100, 3)])
    for nm in ("EQ", "NE", "LT", "GT", "LE", "GE"):
        B[nm] = bin_cases(nm, [(5, 5), (3, 5), (5, 3), (0, 0), (255, 0), (0, 255)])
    B["LI"] = [([("IMM", 8), ("LI", 0)], {8: v}) for v in (0, 99, 255, 1000)]
    B["LC"] = [([("IMM", 8), ("LC", 0)], {8: v}) for v in (0, 77, 255)]
    B["SI"] = [([("IMM", 8), ("PSH", 0), ("IMM", v), ("SI", 0)], {})
               for v in (0, 55, 255)]
    B["SC"] = [([("IMM", 8), ("PSH", 0), ("IMM", v), ("SC", 0)], {})
               for v in (0, 55, 255)]
    B["JSR"] = [([("JSR", 0)], {})]
    B["ENT"] = [([("ENT", v)], {}) for v in (0, 1, 4, 16)]
    B["ADJ"] = [([("ADJ", v)], {}) for v in (0, 1, 4)]
    B["LEV"] = [([("LEV", 0)], {})]
    B["HALT"] = [([("HALT", 0)], {})]
    B["DIV"] = bin_cases("DIV", [(100, 7), (12, 3), (255, 3), (10000, 13),
                                 (65535, 255), (16777215, 15), (0, 5), (7, 100)])
    B["MOD"] = bin_cases("MOD", [(100, 7), (12, 3), (255, 7), (10000, 13),
                                 (65535, 255), (16777215, 15), (0, 5), (7, 100)])
    return B


def main(code_size: int = 32, device: str = "cuda", ops_subset=None):
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

    bats = _op_batteries()
    live_sets: Dict[str, List[int]] = {}
    for opname, cases in bats.items():
        if ops_subset is not None and opname not in ops_subset:
            continue
        t1 = time.time()
        union = set()
        for prog, seed in cases:
            x0 = _freeze_target_input(model, L, prog, seed)
            if x0 is None:
                continue
            base = _run_with_skip(model, x0, set())
            union |= set(greedy_live(model, L, x0, _decode_regs(base, L)))
        live_sets[opname] = sorted(union)
        print(f"  {opname:5s} UNION-live={len(union):3d}/{nb} "
              f"({len(cases)} operands, {time.time()-t1:.1f}s)  "
              f"{[names[i] for i in sorted(union)]}", flush=True)

    if live_sets:
        cs = sorted(len(v) for v in live_sets.values())
        print(f"\n[summary] SOUND operand-UNION live count: min={cs[0]} max={cs[-1]} "
              f"mean={sum(cs)/len(cs):.1f} median={cs[len(cs)//2]}", flush=True)
        union = sorted(set().union(*[set(v) for v in live_sets.values()]))
        print(f"  union-live over ALL ops = {len(union)}/{nb}", flush=True)
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
