#!/usr/bin/env python3
"""Operand-CAM address-leak SURVEY across LEA/LI-addressed clusters.

For each target cluster's operand-delivery step (the ADD/CMP/SUB that reads a
LOADED operand-A out of ALU), read ALU_LO / ALU_HI at the AX marker row (post
block N) and flag the contaminant cells (nonzero cells that are NOT the true
operand nibble). This is the BUILT-layout evidence for whether the address leak
is a GENERALIZABLE address-high-nibble pattern (unified fix) or per-cluster
idiosyncratic (confirming the load-bearing wall).

Runs in the campaign default config (C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1).

  CUDA_VISIBLE_DEVICES=1 python tools/probe_operand_cam_leak_survey.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def cells(row, base, width=16, thr=0.4):
    return {i: round(float(row[base + i].item()), 2)
            for i in range(width) if abs(float(row[base + i].item())) > thr}


TARGETS = [
    (250, "var_simple_0 (x=990; LEA/LI ret)"),
    (325, "var_update_0 (x=x+k; LOADED ADD)"),
    (275, "var_mul_0 (a*b; LOADED MUL)"),
    (350, "if_gt_0 (35>43; IMM CMP)"),
    (400, "if_eq_0 (49==49; IMM CMP)"),
    (425, "if_var_0 (LOADED var CMP)"),
    (550, "func_identity_0 (identity; LEA/LI)"),
    (575, "func_add_0 (a+b; LOADED ADD)"),
    (1046, "absdiff_0 (|16-85|; LOADED SUB/CMP)"),
]

BLK = int(os.environ.get("PROBE_BLK", "13"))


def main():
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    progs = generate_test_programs()
    print(f"campaign NOSTK={os.environ.get('C4_NO_STACK0_EMIT','1')} "
          f"MEMSP={os.environ.get('C4_OPERAND_FROM_MEMSP','1')} "
          f"HI15CLR={os.environ.get('C4_LOADED_OPERAND_ADD_HI15_CLEAR','1')} "
          f"HI13CLR={os.environ.get('C4_FUNCADD_ALU_HI13_CLEAR','1')} blk={BLK}",
          flush=True)
    print(f"STEP_TOKENS={STEP}\n", flush=True)

    for (idx, label) in TARGETS:
        src, exp, desc = progs[idx]
        bc, data = compile_c(src)
        try:
            _, oracle_tokens = runner._oracle_pc_ax_steps(
                bc, data or b"", "", expected_steps=None, with_tokens=True)
        except Exception as e:
            print(f"id={idx} {label}: oracle FAIL {e}", flush=True)
            continue
        prompt = runner._build_element(bc, data or b"", [], "", spec_k=1,
                                       adaptive_start_k=0, expected_steps=None)
        prefix = list(prompt.context)
        tape = list(prefix)
        for stp in oracle_tokens:
            tape.extend(stp)
        padded = torch.tensor([tape], dtype=torch.long, device=dev)
        nsteps = len(oracle_tokens)
        with torch.no_grad():
            resid_all = model.forward(padded, stop_after_block=BLK)[0]
        print(f"=== id={idx} {label} exp={exp} nsteps={nsteps} ===", flush=True)
        for s in range(nsteps):
            start = len(prefix) + s * STEP
            toks = tape[start:start + STEP]
            ax_off = next((i for i, t in enumerate(toks)
                           if t == int(Token.REG_AX)), None)
            if ax_off is None:
                continue
            ax_row = start + ax_off
            r = resid_all[ax_row]
            a_lo = cells(r, dp['ALU_LO'])
            a_hi = cells(r, dp['ALU_HI'])
            if not a_lo and not a_hi:
                continue
            opname = "?"
            for nm in ("OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
                       "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
                       "OP_LI", "OP_LC", "OP_LEA", "OP_IMM", "OP_PSH",
                       "OP_ENT", "OP_ADJ"):
                b = dp.get(nm)
                if b is not None and float(r[b]) > 1.0:
                    opname = nm
                    break
            print(f"  step{s:2d} {opname:7s} ALU_LO={a_lo} ALU_HI={a_hi}",
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
