#!/usr/bin/env python3
"""Decode the per-step REG_AX / REG_SP emitted token values for an expr
program so we see the actual computed intermediate (MUL/ADD result) and where
the second op goes wrong. Compares against the declarative oracle trace.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/probe_expr_regtrace.py 852 800 875
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return bc, exp, desc


def _decode_regs(ctx):
    """Walk the context; each REG_AX marker is followed by 4 value bytes."""
    out = []
    i = 0
    while i < len(ctx):
        t = ctx[i]
        if t == Token.REG_AX and i + 4 < len(ctx):
            v = 0
            for j in range(4):
                v |= (ctx[i + 1 + j] & 0xFF) << (j * 8)
            out.append(("AX", i, v))
            i += 5
        else:
            i += 1
    return out


def main(ids):
    probe = build_groundtruth_probe()
    dev = next(probe.model.parameters()).device
    for idx in ids:
        bc, exp, desc = _corpus(idx)
        ctx = probe._final_context(bc, max_steps=12)
        _, code = probe.emitted_result(bc, max_steps=12)
        regs = _decode_regs(ctx)
        print(f"\n==== id={idx} {desc} exp={exp} neural={code} ====")
        print("  AX values per step:", [v for _, _, v in regs])
        # declarative oracle trace
        try:
            from tests.declarative_oracle import declarative_oracle_for_program
            o = declarative_oracle_for_program(bc, b"", label="probe")
            tr = getattr(o, "trace", None) or getattr(o, "steps_trace", None)
            print("  oracle steps:", getattr(o, "steps", "?"))
        except Exception as e:
            print("  oracle err:", e)


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [852, 800, 875]
    main(ids)
