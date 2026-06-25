#!/usr/bin/env python3
"""var_mul MUL-step ALU+15 / opcode-marker survival probe (campaign config).

Question: at the MUL step's MARK_AX row, which opcode marker dims fire (does
OP_ADD/OP_SUB/OP_EQ.. co-fire with OP_MUL?) and what is ALU_HI+15 / ALU_LO+15?
The dropped LoadedOperandHi15ClearFFN zeroed ALU_{HI,LO}+15 on rows where any of
{ADD,SUB,EQ,NE,LT,GT,LE,GE} fires AND the cell is in (0.5, 5.85). If that window
ALSO captures a value var_mul's MUL legitimately reads (e.g. operand whose true
high nibble is 0xF, or a marker co-fire), the wrap regresses var_mul.

  C4_VM_CACHE_DIR=/tmp/c4cache_varupd3 python tools/probe_varmul_alu15.py [blocks...]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

# var_mul programs: a={a}; b={b}; return a*b  (BOTH operands loaded)
PROGS = {
    "a6_b7":   ("int main() { int a; int b; a = 6; b = 7; return a * b; }", 6, 7, 42),
    "a12_b3":  ("int main() { int a; int b; a = 12; b = 3; return a * b; }", 12, 3, 36),
    "a15_b15": ("int main() { int a; int b; a = 15; b = 15; return a * b; }", 15, 15, 225),
}

OPCODE_NAMES = ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
                "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
                "OP_LI", "OP_SI", "OP_PSH", "OP_LEA", "OP_IMM"]


def fmt(row, base, width=16, thr=0.5):
    vals = [float(row[base + i].item()) for i in range(width)]
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in enumerate(vals) if abs(v) > thr) + "]"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)

    for pname, (src, a, b, exp) in PROGS.items():
        bc, data = compile_c(src)
        _, oracle_tokens = runner._oracle_pc_ax_steps(
            bc, data or b"", "", expected_steps=None, with_tokens=True)
        prompt = runner._build_element(bc, data or b"", [], "", spec_k=1,
                                       adaptive_start_k=0, expected_steps=None)
        prefix = list(prompt.context)
        tape = list(prefix)
        for stp in oracle_tokens:
            tape.extend(stp)
        padded = torch.tensor([tape], dtype=torch.long, device=dev)
        ax_base = dp["MARK_AX"]
        with torch.no_grad():
            emb = model.embed(padded)[0]
        # Markers are materialized at L5 (decode), so read them at the resid
        # of block 11 (the wrap's input). Compute resid once.
        with torch.no_grad():
            resid11 = model.forward(padded, stop_after_block=11)[0]
        # Dump opcode markers at EVERY MARK_AX row to find the MUL step.
        print(f"--- {pname}: per-step MARK_AX opcode markers @blk11 ---", flush=True)
        ax_rows = []
        for stp_idx in range(len(oracle_tokens)):
            ss = len(prefix) + stp_idx * STEP
            for off in range(STEP):
                r = ss + off
                if r < len(tape) and emb[r, ax_base].abs().item() > 0.5:
                    ax_rows.append((stp_idx, r))
                    oc = []
                    for nm in OPCODE_NAMES:
                        if nm in dp and abs(resid11[r, dp[nm]].item()) > 0.3:
                            oc.append(f"{nm}={resid11[r, dp[nm]].item():.1f}")
                    a15 = resid11[r, dp['ALU_HI'] + 15].item()
                    l15 = resid11[r, dp['ALU_LO'] + 15].item()
                    print(f"   step{stp_idx:2d} ax_off={off} "
                          f"ALU_HI+15={a15:.2f} ALU_LO+15={l15:.2f}: {oc}",
                          flush=True)
                    break
        # Find the MUL step.
        mul_step = None
        ax_row = None
        for stp_idx, r in ax_rows:
            if "OP_MUL" in dp and resid11[r, dp["OP_MUL"]].item() > 0.5:
                mul_step = stp_idx
                ax_row = r
                break
        print(f"=== {pname} a={a} b={b} exp={exp} nsteps={len(oracle_tokens)} "
              f"mul_step={mul_step} ax_row={ax_row} ===", flush=True)
        if ax_row is None:
            print("   NO MUL MARK_AX row found", flush=True)
            continue
        # opcode markers at the embed (input) of that row
        oc = []
        for nm in OPCODE_NAMES:
            if nm in dp:
                v = emb[ax_row, dp[nm]].item()
                if abs(v) > 0.3:
                    oc.append(f"{nm}={v:.1f}")
        print(f"   embed opcode markers: {oc}", flush=True)
        for blk in blocks:
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=blk)[0]
            row = resid[ax_row]
            oc_blk = []
            for nm in OPCODE_NAMES:
                if nm in dp:
                    v = row[dp[nm]].item()
                    if abs(v) > 0.3:
                        oc_blk.append(f"{nm}={v:.1f}")
            print(f"   blk{blk:2d} ALU_LO={fmt(row, dp['ALU_LO'])} "
                  f"ALU_HI={fmt(row, dp['ALU_HI'])} "
                  f"markers={oc_blk}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [11]
    main(blks)
