#!/usr/bin/env python3
"""cmp_combine OUTPUT high-nibble leak probe (campaign config, CPU).

Observes the OUTPUT_LO / OUTPUT_HI band + CMP flags at the comparison
decode row across blocks for the failing survey-R5 cmp clusters
(if_gt / if_lt / if_eq step-3, bool_and, func_max/min tail step-13).

ROOT hypothesis: the cmp_combine result byte (0 or 1) written to OUTPUT_LO
by _layer10_alu_cmp_combine_rules / _l10_comparison_combine_rules is a
near-tie vs a leaked operand high-nibble that lands as (hi<<4) in the
OUTPUT band. This probe prints the winning OUTPUT index (argmax) so we
can see the margin between the correct result (0/1) and the leak.

Run:
  CUDA_VISIBLE_DEVICES="" python tools/_probe_cmp_combine_margin.py ALL 24 25 26 27 28
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_cmpmargin")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "6")))

from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


def cmp2(opc, a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), opc, Opcode.EXIT])


# Cases: operands with BOTH nibbles non-zero (the documented drift trigger).
# name -> (bytecode, expected exit code)
ALL = {
    # if_gt: default 1, hi_lt override -> 0
    "gt_18_35": (cmp2(Opcode.GT, 18, 35), 0),   # 0x12 vs 0x23 A<B -> GT 0
    "gt_35_18": (cmp2(Opcode.GT, 35, 18), 1),   # 0x23 vs 0x12 A>B -> GT 1
    "gt_23_23": (cmp2(Opcode.GT, 23, 23), 0),   # eq -> GT 0
    # if_lt
    "lt_18_35": (cmp2(Opcode.LT, 18, 35), 1),   # A<B -> LT 1
    "lt_35_18": (cmp2(Opcode.LT, 35, 18), 0),   # A>B -> LT 0
    # if_eq
    "eq_23_23": (cmp2(Opcode.EQ, 23, 23), 1),   # eq -> 1
    "eq_18_35": (cmp2(Opcode.EQ, 18, 35), 0),   # neq -> 0
    "eq_28_12": (cmp2(Opcode.EQ, 28, 12), 0),   # shared lo nibble C
    # ge/le
    "ge_35_18": (cmp2(Opcode.GE, 35, 18), 1),
    "le_18_35": (cmp2(Opcode.LE, 18, 35), 1),
}


def fmt(row, base, width=16, thr=0.3):
    vals = [float(row[base + i].item()) for i in range(width)]
    hot = [(i, v) for i, v in enumerate(vals) if abs(v) > thr]
    hot.sort(key=lambda t: -t[1])
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in hot) + "]"


def main(which, blocks):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    ohi_key = "OUTPUT_HI_THIS_STEP" if "OUTPUT_HI_THIS_STEP" in dp else "OUTPUT_HI"
    names = [which] if which in ALL else list(ALL.keys())
    for pname in names:
        bc, expect = ALL[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        # the compare step's AX row is the last-but-one AX (EXIT step is last)
        ax_row = ax_rows[-2] if len(ax_rows) >= 2 else ax_rows[-1]
        flag = "PASS" if got == expect else "**FAIL**"
        print(f"=== {pname} expect={expect} got={got} {flag} "
              f"ax_rows={ax_rows} using={ax_row} ===", flush=True)
        for b in blocks:
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=b)[0]
            row = resid[ax_row]
            cmp = [float(row[dp["CMP"] + i].item()) for i in range(4)]
            olo = fmt(row, dp["OUTPUT_LO"], thr=0.2)
            ohi = fmt(row, dp[ohi_key], thr=0.2)
            lo_arg = int(torch.tensor([row[dp["OUTPUT_LO"] + i]
                                       for i in range(16)]).argmax())
            print(f"  blk{b:2d} CMP[hilt,hieq,loeq,lolt]="
                  f"[{cmp[0]:.2f},{cmp[1]:.2f},{cmp[2]:.2f},{cmp[3]:.2f}] "
                  f"LO_arg={lo_arg} OUT_LO={olo} OUT_HI={ohi}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    blks = [int(x) for x in sys.argv[2:]] or [24, 25, 26, 27, 28]
    main(which, blks)
