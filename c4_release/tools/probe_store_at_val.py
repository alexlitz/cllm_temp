#!/usr/bin/env python3
"""Dump MEM_STORE_AT_VAL (the L7 relay output) + CLEAN_EMBED byte-0 + the K-side
store gate value at EVERY MEM value-byte-0 row, at the L8-attn block input.

This isolates Part A of the STACK0 campaign: under C4_NO_STACK0_EMIT=1 the L8
head-5 CAM attends to a PHANTOM value row instead of the real PSH store row.
The discriminator is MEM_STORE_AT_VAL (relayed by make_layer7_mem_store_relay_op
from the MARK_MEM marker row). If it is 0 on the real store's value row (or 1 on
phantom rows) under NOSTK=1, the relay broke in the 30-token layout.

Usage:
    CUDA_VISIBLE_DEVICES=0 C4_OPERAND_FROM_MEMSP=1 C4_NO_STACK0_EMIT=1 \
        python tools/probe_store_at_val.py 9 3   # id, target_step
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
    MEM_STORE = dp["MEM_STORE"]
    MEM_VAL_B0 = dp["MEM_VAL_B0"]
    MEM_VAL_B1 = dp["MEM_VAL_B1"]
    L2H0 = dp["L2H0"]
    H1 = dp["H1"]
    SAV = dp.get("MEM_STORE_AT_VAL", None)
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

    print(f"== id={idx} {desc[:36]} step{target_step} blk{block} "
          f"OPERAND={os.environ.get('C4_OPERAND_FROM_MEMSP','0')} "
          f"NOSTK={os.environ.get('C4_NO_STACK0_EMIT','0')} "
          f"ST={ST} SAV_dim={SAV} ==")

    padded = torch.tensor([context], dtype=torch.long, device=dev)
    resid = model.forward(padded, stop_after_block=block)[0]

    print("  pos | tok | role        | M_MEM STORE | VB0 VB1 | "
          "L2H0m H1m | STORE_AT_VAL | CE0")
    qlimit = len(prefix) + (target_step + 1) * ST
    for pos in range(min(qlimit, len(context))):
        v = resid[pos]
        tok = context[pos]
        mmem = float(v[MARK_MEM])
        store = float(v[MEM_STORE])
        vb0 = float(v[MEM_VAL_B0])
        vb1 = float(v[MEM_VAL_B1])
        l2h0m = float(v[L2H0 + MEM_I])
        h1m = float(v[H1 + MEM_I])
        sav = float(v[SAV]) if SAV is not None else float("nan")
        ce0 = _nib_argmax(v, CLO)
        is_marker = mmem > 0.5
        is_val0 = (l2h0m > 0.5 and h1m < 0.5)
        if not (is_marker or is_val0 or store > 0.5):
            continue
        role = "MARK_MEM" if is_marker else ("VAL0" if is_val0 else "store?")
        print(f"  {pos:3d} | {tok:3d} | {role:11s} | {mmem:5.1f} {store:5.1f} | "
              f"{vb0:3.0f} {vb1:3.0f} | {l2h0m:5.1f} {h1m:4.1f} | "
              f"{sav:12.3f} | {ce0:3d}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.lstrip('-').isdigit()]
    idx = args[0] if args else 9
    step = args[1] if len(args) > 1 else 3
    ms = args[2] if len(args) > 2 else 6
    blk = args[3] if len(args) > 3 else 11
    main(idx, step, ms, blk)
