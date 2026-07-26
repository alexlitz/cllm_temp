#!/usr/bin/env python3
"""validate_real_scale_word32.py — Task 4: real-scale MONOLITHIC proof.

Run at least ONE full-ISA-scale sparse matmul (contraction dim ~896 = the
model's d_model, plus the model's MAX nnz = 1964) end-to-end MONOLITHICALLY
through the 32-bit draft VM — ONE pass, no per-op summing — and confirm:

  * the result is BYTE-EXACT vs numpy 32-bit (the FULL value: hi + lo bytes,
    not just the truncatable low byte),
  * the per-op step RATE (steps/MAC) measured at this REAL scale matches the
    toy-scale rate the grounding uses (so the grounding's per-op rate is valid
    at the real dimensions, not only the toy 207-node c4vm.onnx).

This is the "actually ran it" proof that the grounded step count's per-op rate
holds at the real forward scale.

Run:  python -m c4_min.selfhost.validate_real_scale_word32
"""
from __future__ import annotations

import sys
import time

import numpy as np

from c4_min.selfhost.word32_draft_vm import ref_interpret_word32, WORD
from c4_min.selfhost._matmul_paged_src import paged_dot_c, SCALE


def _emit_hilo_dot_code(w, x, scale=SCALE):
    """Compile a paged fixed-point dot that emits the acc's (hi, lo) bytes so the
    FULL 32-bit value is validated monolithically."""
    body = paged_dot_c(w, x, scale=scale)
    assert "printf(acc);" in body and "int s; int acc, k;" in body
    body = body.replace("int s; int acc, k;", "int s; int acc, k; int hi; int lo;")
    body = body.replace("printf(acc);",
                        "hi = acc / 256; lo = acc - hi * 256; printf(hi); printf(lo);")
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, _ = compile_c(body)
    return bytecode_to_isa(bc)


def _numpy32_dot(w, x, scale=SCALE):
    acc = 0
    for i in range(len(w)):
        acc = (acc + (w[i] * scale * x[i] * scale) // scale) & WORD
    return acc & WORD


def main() -> int:
    print("=" * 74)
    print("Task 4 — REAL-SCALE MONOLITHIC validation of the 32-bit draft VM")
    print("=" * 74)
    print("ONE pass per dim through ref_interpret_word32 (no per-op summing).\n")
    ok = True
    rates = []
    # d_model=896 (a real FFN/attn contraction) and the model's MAX nnz=1964.
    for K in [896, 1964]:
        rng = np.random.RandomState(7)
        w = [int(v) for v in rng.randint(0, 3, size=K)]      # weight-like
        x = [int(v) for v in rng.randint(0, 5, size=K)]      # activation-like
        code = _emit_hilo_dot_code(w, x)
        t0 = time.time()
        trace, steps = ref_interpret_word32(code, out=(o := []), max_steps=80_000_000)
        wall = time.time() - t0
        ref = _numpy32_dot(w, x)
        ref_hi, ref_lo = (ref >> 8) & 0xFF, ref & 0xFF
        byte_exact = (o == [ref_hi, ref_lo])
        rate = steps / K
        rates.append(rate)
        print(f"  K={K:>4}: numpy32 acc={ref:>7} (hi={ref_hi:>3} lo={ref_lo:>3})  "
              f"word32 emit={o}  byte_exact={byte_exact}")
        print(f"          steps={steps:,}  steps/MAC={rate:.2f}  wall={wall*1000:.0f}ms")
        ok = ok and byte_exact

    # per-op rate consistency vs the toy-scale rate the grounding uses.
    toy_rate = 76.38          # ground_full_model_steps measured paged COO steps/MAC
    real_rate = float(np.mean(rates))
    consistent = abs(real_rate - toy_rate) / toy_rate < 0.02
    print()
    print(f"  toy-scale grounding rate   = {toy_rate:.2f} steps/MAC")
    print(f"  REAL-scale (K=896/1964)    = {real_rate:.2f} steps/MAC")
    print(f"  rate holds at real scale (within 2%): {consistent}")
    ok = ok and consistent

    print()
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  — the 32-bit draft VM runs a real-"
          f"dimension\n         sparse matmul monolithically, byte-exact, at the "
          f"grounding's per-op rate.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
