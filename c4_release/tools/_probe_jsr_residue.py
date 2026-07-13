#!/usr/bin/env python3
"""At the byte-1 emit row entering block 34, dump OP_JSR / OP_MUL / OP_SHL /
OP_ENT residue for idx104 (fail) vs idx110 (pass), and other muls, to confirm
the spurious-OP_JSR-leak root of the l14_jsr_ax_bytes_zero mis-fire."""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0"); os.environ["C4_TEST_SPEC_K"] = "0"
_H = os.path.dirname(os.path.abspath(__file__)); _P = os.path.dirname(_H)
if _P not in sys.path: sys.path.insert(0, _P)
import warnings; warnings.filterwarnings("ignore")
import torch
from src.compiler import compile_c
from neural_vm.verification.faithful_autoregressive import (
    FaithfulAutoregressiveRunner, _ContextShim,
)
from neural_vm.verification.faithful_interpreter import ModelExactForward
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs

BLK = 34


@torch.no_grad()
def prep(pid, model, mef):
    name, exp, src = generate_test_programs()[pid]
    bc, _ = compile_c(name)
    STEP = int(Token.STEP_TOKENS); RAX = int(Token.REG_AX)
    shim = _ContextShim(model)
    tape = list(shim._build_context(bc, b"", []))
    prefix = len(tape)
    for _ in range(4 * STEP + 6):
        lg = mef.forward(tape, return_logits=True)
        tape.append(int(lg[-1].argmax()))
    ax = [i for i, t in enumerate(tape) if t == RAX and i >= prefix]
    mk = None
    for c in reversed(ax):
        if c + 5 < len(tape): mk = c; break
    return tape[:mk + 2], mk + 1, exp


@torch.no_grad()
def main():
    runner = FaithfulAutoregressiveRunner(disk_cache=True)
    model = runner.model
    dp = dict(runner.dim_positions)
    mef = ModelExactForward(model)
    pids = [int(x) for x in (sys.argv[1:] or ["104", "110", "106", "100", "102", "103", "107", "111"])]
    fields = ["OP_JSR", "OP_MUL", "OP_SHL", "OP_ENT", "IS_BYTE", "H1", "MARK_AX"]
    print(f"{'pid':>4} {'0xval':>6} {'b0':>4}  " + "  ".join(f"{f:>7}" for f in fields) + "   b1_pass")
    for pid in pids:
        trunc, row, exp = prep(pid, model, mef)
        padded = torch.tensor([trunc], dtype=torch.long, device=mef.device)
        x_in = model.forward(padded, stop_after_block=BLK - 1)[0]
        if x_in.is_sparse: x_in = x_in.to_dense()
        r = x_in.float()[row]
        vals = []
        for f in fields:
            d = dp.get(f)
            if f == "H1":
                vals.append(float(r[d + 1]))  # H1+1 (AX byte)
            else:
                vals.append(float(r[d]))
        # decode byte-1
        want = (exp >> 8) & 0xFF
        # final decode of byte-1: argmax logits at row over full model
        lg = mef.forward(trunc, return_logits=True)
        emit = int(lg[-1].argmax())
        pf = "PASS" if emit == want else f"FAIL(got 0x{emit:02x} want 0x{want:02x})"
        print(f"{pid:>4} 0x{exp:04x} 0x{exp&0xff:02x}  "
              + "  ".join(f"{v:>7.2f}" for v in vals) + f"   {pf}", flush=True)


if __name__ == "__main__":
    main()
