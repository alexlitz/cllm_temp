#!/usr/bin/env python3
"""Equal-high-nibble GT lo-margin probe (campaign config, CPU).

Reads the four CMP flags + OUTPUT_LO at the AX decode row across blocks for
the failing if_gt cases (eq-hi, lo_gt) + contrast. Mirrors the proven
probe_cmp_se_recover.py path (build_groundtruth_probe._final_context) but
pinned to CPU and lean (few programs/blocks).
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_VM_CACHE_DIR", "/tmp/c4cache_cmpfront")
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


def gt(a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b),
                Opcode.GT, Opcode.EXIT])


def lt(a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b),
                Opcode.LT, Opcode.EXIT])


ALL = {
    "gt_54_53": (gt(54, 53), 1),   # FAIL eq-hi lo_gt -> GT=1
    "gt_53_54": (gt(53, 54), 0),   # eq-hi lo_lt -> GT=0 (override must fire)
    "gt_97_20": (gt(97, 20), 1),   # diff-hi A>B -> GT=1 (1087 step3)
    "gt_54_43": (gt(54, 43), 1),   # diff-hi A>B -> GT=1
    "lt_86_87": (lt(86, 87), 1),   # eq-hi lo_lt -> LT=1 (override must fire)
    "lt_50_44": (lt(50, 44), 0),   # eq-hi lo_gt -> LT=0 (lo_lt-alone must NOT fire)
}


def fmt(row, base, width=16, thr=0.3):
    vals = [float(row[base + i].item()) for i in range(width)]
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in enumerate(vals)
                           if abs(v) > thr) + "]"


def main(which, blocks):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
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
        ax_row = ax_rows[-1]
        flag = "PASS" if got == expect else "**FAIL**"
        print(f"=== {pname} expect={expect} got={got} {flag} ax_row={ax_row} ===",
              flush=True)
        # find the emit/decode row: the EXIT step's AX row (last AX before EXIT)
        for b in blocks:
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=b)[0]
            row = resid[ax_row]
            cmp = [float(row[dp["CMP"] + i].item()) for i in range(4)]
            olo = fmt(row, dp["OUTPUT_LO"], thr=0.1)
            ohi_key = "OUTPUT_HI_THIS_STEP" if "OUTPUT_HI_THIS_STEP" in dp else "OUTPUT_HI"
            ohi = fmt(row, dp[ohi_key], thr=0.1)
            print(f"  blk{b:2d} CMP[hilt,hieq,loeq,lolt]="
                  f"[{cmp[0]:.2f},{cmp[1]:.2f},{cmp[2]:.2f},{cmp[3]:.2f}] "
                  f"OUT_LO={olo} OUT_HI={ohi}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    blks = [int(x) for x in sys.argv[2:]] or [21, 22, 23]
    main(which, blks)
