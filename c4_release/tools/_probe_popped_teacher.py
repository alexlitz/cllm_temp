#!/usr/bin/env python3
"""CPU teacher-forced self-check for the STACK0_B0_POPPED dump blocker.

ONE forward over the byte-exact DraftVM oracle tape (no autoregressive loop, so
it is tractable under CPU contention). Per VM step it reads:
  * the model's STACK0[0] EMISSION (argmax at the offset-20 predictor row), and
  * the ``STACK0_B0_POPPED`` latch value at the STACK0-marker row.
A correctly-popped frame shows POPPED=0 on the ON-stack steps (the dump keeps
the operand byte-0) and POPPED=1 on the post-pop steps (the dump is blocked ->
STACK0[0] emits the oracle's 0x00, not the stale operand).

Because this is teacher-forced over the byte-exact oracle tape, the per-step
emission argmax == the model's own production emission for every step at/before
the first value-byte correction (interp_oracle_gate soundness guard).

Usage: C4_STACK0_B0_POPPED=1 python tools/_probe_popped_teacher.py [id] [max_steps]
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

STACK0_VAL_OFF = 21    # value byte 0 inside a 35-token step
STACK0_PRED_OFF = 20   # row that predicts the offset-21 token


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 350
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]

    with contextlib.redirect_stdout(io.StringIO()):
        from tools.interp_oracle_gate import build_gate_context
        ctx = build_gate_context(verbose=False)
    dp = ctx.dim_positions
    popd = dp.get("STACK0_B0_POPPED")
    ms = dp.get("MARK_STACK0")

    # Byte-exact oracle tape via the DraftVM (the speculation reference).
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner, DraftVM
    oracle = declarative_oracle_for_program(bc, suite_expected=exp)
    cap = min(max_steps, (oracle.steps or max_steps))
    vm = DraftVM(list(bc))
    step_tokens = []
    for _ in range(cap):
        if vm.halted:
            break
        if not vm.step():
            break
        step_tokens.append([int(t) for t in vm.draft_tokens()])
        if vm.halted:
            break

    # Build the prompt prefix + concatenate the per-step reference tokens.
    from neural_vm.run_vm import AutoregressiveVMRunner
    class _Shim:  # noqa: N801
        model = ctx.model
    prompt = AutoregressiveVMRunner._build_context(_Shim(), bc, [], [])
    tape = list(prompt)
    step_rows = []
    for st in step_tokens:
        base = len(tape)
        tape.extend(st)
        step_rows.append(base)

    print(f"\n===== id{pid} {desc!r} (exp={exp & 0xff}) flag_POPPED={os.environ.get('C4_STACK0_B0_POPPED','0')} =====")
    print(f"  STACK0_B0_POPPED dim={popd}  steps={len(step_tokens)}")

    # ONE teacher-forced forward over the whole oracle tape.
    R = ctx.fwd._residual_pre_head(tape)
    W = ctx.model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = ctx.model.head.bias

    print(f"\n  step | oracle_STACK0[0] | emit_argmax | POPPED@stack0_row")
    for si, base in enumerate(step_rows):
        st = step_tokens[si]
        oracle_b0 = st[STACK0_VAL_OFF] if len(st) > STACK0_VAL_OFF else None
        pred_row = base + STACK0_PRED_OFF
        emit = int((W @ R[pred_row] + (b if b is not None else 0)).argmax())
        stack0_row = base + STACK0_PRED_OFF  # marker row carries the band
        pv = float(R[stack0_row][popd]) if popd is not None else float("nan")
        ob = "?" if oracle_b0 is None else f"0x{oracle_b0:02x}={oracle_b0}"
        print(f"  {si:4d} | oracle={ob:11s} | emit=0x{emit:02x}={emit:<4d} | POPPED={pv:+.3f}")


if __name__ == "__main__":
    main()
