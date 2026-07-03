#!/usr/bin/env python3
"""Multi-id teacher-forced POPPED self-check (amortizes the one CPU build).

For each program id: ONE forward over the byte-exact DraftVM oracle tape; per
step prints oracle STACK0[0], the model's STACK0[0] emission argmax, and the
STACK0_B0_POPPED latch value at the STACK0 row. Confirms the latch fires from
the consuming cmp/branch step onward and the dump emits 0 on post-pop steps.

Usage: C4_STACK0_B0_POPPED=1 python tools/_probe_popped_teacher_multi.py id1,id2,... [max_steps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ["C4_TEST_SPEC_K"] = "0"; os.environ["C4_SMOKE_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from tests.declarative_oracle import declarative_oracle_for_program
from neural_vm.batched_pure_neural import DraftVM
from neural_vm.run_vm import AutoregressiveVMRunner

STACK0_VAL_OFF = 21
STACK0_PRED_OFF = 20


def run_one(ctx, dp, pid, max_steps):
    popd = dp.get("STACK0_B0_POPPED")
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    oracle = declarative_oracle_for_program(bc, suite_expected=exp)
    cap = min(max_steps, (oracle.steps or max_steps))
    vm = DraftVM(list(bc))
    step_tokens = []
    for _ in range(cap):
        if vm.halted or not vm.step():
            break
        step_tokens.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break

    class _Shim:  # noqa: N801
        model = ctx.model
    prompt = AutoregressiveVMRunner._build_context(_Shim(), bc, [], [])
    tape = list(prompt); rows = []
    for st in step_tokens:
        rows.append(len(tape)); tape.extend(st)

    R = ctx.fwd._residual_pre_head(tape)
    W = ctx.model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = ctx.model.head.bias
    print(f"\n== id{pid} {desc!r} (exp={exp & 0xff}) ==")
    print("  step | oracle | emit | POPPED")
    for si, base in enumerate(step_tokens and rows):
        st = step_tokens[si]
        ob = st[STACK0_VAL_OFF] if len(st) > STACK0_VAL_OFF else -1
        pr = base + STACK0_PRED_OFF
        emit = int((W @ R[pr] + (b if b is not None else 0)).argmax())
        pv = float(R[base + STACK0_PRED_OFF][popd]) if popd is not None else float("nan")
        print(f"  {si:4d} | 0x{ob:02x} | 0x{emit:02x}={emit:<3d} | POPPED={pv:+.2f}")


def main():
    ids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [350]
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions
    print(f"flag_POPPED={os.environ.get('C4_STACK0_B0_POPPED','0')} POPPED_dim={dp.get('STACK0_B0_POPPED')}")
    for pid in ids:
        run_one(ctx, dp, pid, max_steps)


if __name__ == "__main__":
    main()
