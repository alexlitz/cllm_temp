#!/usr/bin/env python3
"""Find the EXACT block + magnitude of the OUTPUT corruptor at the divmod row.

For each fail, trace the RAW OUTPUT_LO/HI values (not just one-hot) at the
DIV compute row across blocks 28..end, and report the per-block DELTA to
OUTPUT so we can see which block writes the divisor nibbles and at what
magnitude (to size a dominant-write fix).
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
    args = sys.argv[1:] or ["1162/37", "462/13"]
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    OLO = dp["OUTPUT_LO"]; OHI = dp["OUTPUT_HI"]
    MARK_AX = dp["MARK_AX"]; OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]

    for arg in args:
        a, b = arg.split("/"); a = int(a); b = int(b)
        PROG = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.DIV, Opcode.EXIT])
        ctx = probe._final_context(PROG, max_steps=6)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        rl = model.forward(padded, stop_after_block=nblocks - 1)[0]
        drs = [r for r in range(rl.shape[0])
               if float(rl[r, MARK_AX].item()) > 0.5
               and (float(rl[r, OP_DIV].item()) > 0.5 or float(rl[r, OP_MOD].item()) > 0.5)]
        row = drs[-1]
        print(f"\n=== {a}/{b} (q={a//b}=0x{a//b:X}) div row={row} divisor=0x{b:X} ===")
        prev_lo = prev_hi = None
        for blk in range(27, nblocks):
            r = model.forward(padded, stop_after_block=blk)[0]
            lo = r[row, OLO:OLO + 16].cpu()
            hi = r[row, OHI:OHI + 16].cpu()
            if prev_lo is None:
                prev_lo, prev_hi = lo, hi
                continue
            dlo = lo - prev_lo
            dhi = hi - prev_hi
            chg_lo = [(i, round(float(dlo[i]), 1)) for i in range(16) if abs(float(dlo[i])) > 0.3]
            chg_hi = [(i, round(float(dhi[i]), 1)) for i in range(16) if abs(float(dhi[i])) > 0.3]
            if chg_lo or chg_hi:
                lay = getattr(model.blocks[blk], '_logical_layer', blk)
                # current argmax
                am_lo = int(lo.argmax()) if float(lo.max()) > 0.5 else -1
                am_hi = int(hi.argmax()) if float(hi.max()) > 0.5 else -1
                print(f"  blk{blk:2d} L{lay}: dLO={chg_lo} dHI={chg_hi}  -> now argmax lo={am_lo} hi={am_hi} "
                      f"(maxval lo={float(lo.max()):.1f} hi={float(hi.max()):.1f})")
            prev_lo, prev_hi = lo, hi


if __name__ == "__main__":
    main()
