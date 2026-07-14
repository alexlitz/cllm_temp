#!/usr/bin/env python3
"""Verify C4_MUL_B1_DELIVERY: decode byte-1 (and full exit) for the mul cluster
(idx 100..114) via faithful-autoregressive (spec_k=0). Reports per-program
whether the decoded AX equals the oracle exit. Set C4_MUL_B1_DELIVERY in env.
"""
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


@torch.no_grad()
def decode_exit(pid, model, mef):
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
    # decode the 4 emitted bytes of the mul-result AX marker (as full_trace does)
    b = [tape[mk + 1 + j] & 0xFF for j in range(4)]
    got = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
    return exp, got, src


@torch.no_grad()
def main():
    pids = [int(x) for x in sys.argv[1:]] or list(range(100, 115))
    flag = os.environ.get("C4_MUL_B1_DELIVERY", "0")
    print(f"C4_MUL_B1_DELIVERY={flag}")
    runner = FaithfulAutoregressiveRunner(disk_cache=True)
    model = runner.model
    mef = ModelExactForward(model)
    npass = 0; nfail = 0
    for pid in pids:
        exp, got, src = decode_exit(pid, model, mef)
        ok = (exp == got)
        npass += ok; nfail += (not ok)
        tag = "PASS" if ok else "FAIL"
        print(f" id{pid:>3} {tag}  got={got}(0x{got:x}) oracle={exp}(0x{exp:x})  {src.strip()[:40]}", flush=True)
    print(f"=== {npass} pass / {nfail} fail / {len(pids)} total ===")


if __name__ == "__main__":
    main()
