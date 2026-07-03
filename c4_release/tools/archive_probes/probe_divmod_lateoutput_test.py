#!/usr/bin/env python3
"""HYPOTHESIS TEST 2 (hook): is the final exit driven by OUTPUT at the divmod
row? Force the CORRECT quotient one-hot into OUTPUT_LO/HI at the divmod AX row
at a LATE block (after all corruptors) and see if exit becomes correct.

Also: find the EARLIEST block at which forcing the correct OUTPUT survives to
a correct exit (tells us whether a late re-emit in the divmod writeback could
work, and which corruptors matter).
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            o, i = op; bc.append(o | (i << 8))
        else:
            bc.append(op)
    return bc


def main():
    args = sys.argv[1:] or ["1162/37", "462/13", "1132/33"]
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    OLO = dp["OUTPUT_LO"]; OHI = dp["OUTPUT_HI"]
    OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]; MARK_AX = dp["MARK_AX"]

    state = {"on": False, "q": 0, "block": None, "row": None}

    def make_hook(blk):
        def hook(module, inp, out):
            if not state["on"] or state["block"] != blk:
                return out
            x = out[0] if isinstance(out, tuple) else out
            q = state["q"]
            lo_nib = q & 0xF
            hi_nib = (q >> 4) & 0xF
            row = state["row"]
            x[:, row, OLO:OLO + 16] = 0.0
            x[:, row, OHI:OHI + 16] = 0.0
            x[:, row, OLO + lo_nib] = 30.0
            x[:, row, OHI + hi_nib] = 30.0
            return (x,) + out[1:] if isinstance(out, tuple) else x
        return hook

    hooks = [model.blocks[b].register_forward_hook(make_hook(b)) for b in range(nblocks)]

    for arg in args:
        a, b = arg.split("/"); a = int(a); b = int(b)
        q = a // b
        PROG = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.DIV, Opcode.EXIT])
        # find the divmod row first (no hook)
        state["on"] = False
        ctx = probe._final_context(PROG, max_steps=6)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        rl = model.forward(padded, stop_after_block=nblocks - 1)[0]
        drs = [r for r in range(rl.shape[0])
               if float(rl[r, MARK_AX].item()) > 0.5
               and (float(rl[r, OP_DIV].item()) > 0.5 or float(rl[r, OP_MOD].item()) > 0.5)]
        row = drs[-1]
        _, base = probe.emitted_result(PROG, max_steps=6)
        print(f"\n=== {a}/{b} q={q} (row={row}) base_exit={base} ===")
        # try forcing correct OUTPUT at each candidate late block
        for blk in [28, 32, 34, 41, 42, nblocks - 2, nblocks - 1]:
            state.update(on=True, q=q, block=blk, row=row)
            _, fixed = probe.emitted_result(PROG, max_steps=6)
            state["on"] = False
            tag = "OK" if fixed == q else ""
            print(f"  force OUTPUT@blk{blk}: exit={fixed} {tag}")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
