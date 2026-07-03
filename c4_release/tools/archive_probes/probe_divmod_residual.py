#!/usr/bin/env python3
"""Probe the divmod residual fails: trace the operands the FlattenedDivMod
actually receives (NIB_A/NIB_B at GE positions 0..3) and the OUTPUT
trajectory at the DIV/MOD compute row across all blocks.

Run in the CAMPAIGN config:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 CUDA_VISIBLE_DEVICES=0 \
    python tools/probe_divmod_residual.py 1162/37 462/13 2009/43
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


def oh(row, base, w=16):
    return [i for i in range(w) if float(row[base + i].item()) > 0.5]


def oh_val(row, base, w=16):
    """Decode a one-hot nibble band into its integer value (argmax)."""
    vals = [float(row[base + i].item()) for i in range(w)]
    m = max(vals)
    if m <= 0.5:
        return None
    return vals.index(m)


def main():
    args = sys.argv[1:]
    if not args:
        args = ["1162/37", "462/13", "2009/43", "106/4", "1132/33"]
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    OLO = dp["OUTPUT_LO"]; OHI = dp["OUTPUT_HI"]
    MARK_AX = dp["MARK_AX"]; OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]
    ALU_LO = dp.get("ALU_LO"); ALU_HI = dp.get("ALU_HI")
    AXC_LO = dp.get("AX_CARRY_LO"); AXC_HI = dp.get("AX_CARRY_HI")
    SE_ALU_LO = dp.get("SE_ALU_LO"); SE_ALU_HI = dp.get("SE_ALU_HI")
    AXF_LO = dp.get("AX_FULL_LO"); AXF_HI = dp.get("AX_FULL_HI")
    SBV1_LO = dp.get("STACK0_BYTE_VAL_1_LO"); SBV1_HI = dp.get("STACK0_BYTE_VAL_1_HI")
    print(f"# nblocks={nblocks} campaign NO_STACK0_EMIT={os.environ.get('C4_NO_STACK0_EMIT')} "
          f"OPERAND_FROM_MEMSP={os.environ.get('C4_OPERAND_FROM_MEMSP')} DIV_MULTIBYTE={os.environ.get('C4_DIV_MULTIBYTE','1')}")

    for arg in args:
        a, b = arg.split("/")
        a = int(a); b = int(b)
        PROG = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.DIV, Opcode.EXIT])
        ctx = probe._final_context(PROG, max_steps=6)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        rl = model.forward(padded, stop_after_block=nblocks - 1)[0]
        drs = [r for r in range(rl.shape[0])
               if float(rl[r, MARK_AX].item()) > 0.5
               and (float(rl[r, OP_DIV].item()) > 0.5 or float(rl[r, OP_MOD].item()) > 0.5)]
        if not drs:
            print(f"\n=== {a}/{b} (q={a//b} r={a%b}): NO DIV compute row found ===")
            continue
        row = drs[-1]
        print(f"\n=== {a}/{b} (q={a//b} r={a%b}) DIV row={row} ===")
        # Find the block where divmod compute runs. Scan ALU/operand bands
        # and OUTPUT at the div row across blocks.
        prev_out = None
        for blk in range(nblocks):
            r = model.forward(padded, stop_after_block=blk)[0]
            rr = r[row]
            out = (oh(rr, OLO), oh(rr, OHI))
            # operand bands
            def v(base):
                return oh_val(rr, base) if base is not None else None
            alu = (v(ALU_LO), v(ALU_HI))
            axc = (v(AXC_LO), v(AXC_HI))
            se = (v(SE_ALU_LO), v(SE_ALU_HI))
            axf = (v(AXF_LO), v(AXF_HI))
            sbv1 = (v(SBV1_LO), v(SBV1_HI))
            tag = ""
            if out != prev_out:
                tag = f" OUTPUT lo={out[0]} hi={out[1]}"
                prev_out = out
            # only print when output changes or at the divmod-input/output region
            if tag:
                lay = getattr(model.blocks[blk], '_logical_layer', blk)
                print(f"  blk{blk:2d} L{lay}:{tag} | ALU={alu} AXC={axc} SE={se} AXF={axf} SBV1={sbv1}")
        _, code = probe.emitted_result(PROG, max_steps=6)
        print(f"  --> neural exit = {code} (expect {a//b})")


if __name__ == "__main__":
    main()
