#!/usr/bin/env python3
"""Dump per-row MEM-section K-side signals at the L8 attn block input.

For a target VM step's AX-marker query, enumerate EVERY token row in the
emitted context up to that step and print the K-side dims the L8 head-5
mem-to-ALU CAM keys on:
  MARK_MEM, MEM_STORE, MEM_VAL_B0/B1/B2/B3, L2H0[MEM_I], H1[MEM_I],
  PSH_AT_SP, plus the CLEAN_EMBED byte-0 value held at that row.

This pinpoints which row carries the operand value (mem[SP] byte 0) and
whether any per-row store-commit bit exists to discriminate the real PSH
store's value row from a phantom IMM-step MEM value row.

Usage:
    CUDA_VISIBLE_DEVICES=0 C4_OPERAND_FROM_MEMSP=1 C4_NO_STACK0_EMIT=0 \
        python tools/probe_mem_store_rows.py 9 3   # id, target_step
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import torch  # noqa: E402

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return list(bc), exp, desc


def _nib_argmax(vec, base):
    sl = vec[base:base + 16]
    m = float(sl.max())
    if m < 0.5:
        return -1
    return int(sl.argmax())


@torch.no_grad()
def main(idx, target_step, max_steps, block):
    ST = Token.STEP_TOKENS
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers = {}
    mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)
    model = runner.model
    dev = next(model.parameters()).device
    dp = model.dim_positions
    CLO = dp["CLEAN_EMBED_LO"]
    MARK_MEM = dp["MARK_MEM"]
    MARK_AX = dp["MARK_AX"]
    MEM_STORE = dp["MEM_STORE"]
    PSH_AT_SP = dp["PSH_AT_SP"]
    MEM_VAL_B0 = dp["MEM_VAL_B0"]
    MEM_VAL_B1 = dp["MEM_VAL_B1"]
    MEM_VAL_B2 = dp["MEM_VAL_B2"]
    MEM_VAL_B3 = dp["MEM_VAL_B3"]
    L2H0 = dp["L2H0"]
    H1 = dp["H1"]
    MEM_I = 4

    bc, exp, desc = _corpus(idx)
    prefix = list(mr._build_context(bc, b"", [], ""))
    context = list(prefix)
    for _ in range(ST * max_steps):
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        nxt = int(model.forward(padded)[0][-1].argmax().item())
        context.append(nxt)
        if nxt == Token.HALT:
            break

    emitted = context[len(prefix):]
    step_slice = emitted[target_step * ST:(target_step + 1) * ST]
    ax_off = None
    for i, t in enumerate(step_slice):
        if t == Token.REG_AX:
            ax_off = i
            break
    qpos = len(prefix) + target_step * ST + (ax_off if ax_off is not None else 1)
    print(f"== id={idx} {desc[:36]} step{target_step} AXq@pos{qpos} "
          f"blk{block} OPERAND={os.environ.get('C4_OPERAND_FROM_MEMSP','0')} "
          f"NOSTK={os.environ.get('C4_NO_STACK0_EMIT','0')} ==")

    padded = torch.tensor([context], dtype=torch.long, device=dev)
    resid = model.forward(padded, stop_after_block=block)[0]

    print("  pos | tok | M_MEM M_AX | STORE PSHsp | VB0 VB1 VB2 VB3 | "
          "L2H0m H1m | CE0")
    for pos in range(min(qpos + 1, len(context))):
        v = resid[pos]
        tok = context[pos]
        mmem = float(v[MARK_MEM])
        max_ = float(v[MARK_AX])
        store = float(v[MEM_STORE])
        psh = float(v[PSH_AT_SP])
        vb0 = float(v[MEM_VAL_B0])
        vb1 = float(v[MEM_VAL_B1])
        vb2 = float(v[MEM_VAL_B2])
        vb3 = float(v[MEM_VAL_B3])
        l2h0m = float(v[L2H0 + MEM_I])
        h1m = float(v[H1 + MEM_I])
        ce0 = _nib_argmax(v, CLO)
        # Only print rows that look MEM-relevant or store-relevant or the query.
        relevant = (mmem > 0.5 or store > 0.5 or psh > 0.5 or vb0 > 0.5 or
                    vb1 > 0.5 or vb2 > 0.5 or vb3 > 0.5 or
                    (l2h0m > 0.5 and h1m < 0.5) or pos == qpos)
        if not relevant:
            continue
        mark = " <Q" if pos == qpos else ""
        print(f"  {pos:3d} | {tok:3d} | {mmem:5.1f} {max_:4.1f} | "
              f"{store:5.1f} {psh:5.1f} | "
              f"{vb0:3.0f} {vb1:3.0f} {vb2:3.0f} {vb3:3.0f} | "
              f"{l2h0m:5.1f} {h1m:4.1f} | {ce0:3d}{mark}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.lstrip('-').isdigit()]
    idx = args[0] if args else 9
    step = args[1] if len(args) > 1 else 3
    ms = args[2] if len(args) > 2 else 6
    blk = args[3] if len(args) > 3 else 11
    main(idx, step, ms, blk)
