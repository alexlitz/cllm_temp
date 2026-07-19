#!/usr/bin/env python3
"""Standalone gadget-level check of the RECURRENT divmod block body: apply the
unique block specs in the recurrent apply-order to a residual vector, threading
R/IT/DIV_RES across the reused iteration applications, and verify DIV_RES/MOD_RES
match divmod32 for a battery of operands (incl. edge cases).  No full model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from c4_min import nibble_alu32 as A
from c4_min.nibble_pure_forward_complete import PureForwardCompleteLayout, N_ROLES
from c4_min.nibble_muldivmod import divmod32


def apply_spec(x, spec):
    up = F.linear(x, spec["W_up"]) + spec["b_up"]
    gate = F.linear(x, spec["W_gate"]) + spec["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, spec["W_down"], spec["b_down"])


def run(a_val, b_val, L, dim, unique, apply_names):
    al = L.ALU32
    spec_by_name = dict(unique)
    x = torch.zeros(dim, dtype=torch.float64)
    x[L.ONE] = 1.0
    # seed STACK0 nibbles = a, AX nibbles = b (the operand-B the KB precompute reads).
    for c in range(8):
        x[L.STACK0 + c] = (a_val >> (4 * c)) & 0xF
        x[L.AX + c] = (b_val >> (4 * c)) & 0xF
    for name in apply_names:
        x = apply_spec(x.to(torch.float64), spec_by_name[name].to(torch.float64) if False else spec_by_name[name])
    div = 0
    mod = 0
    for c in range(8):
        div |= (int(round(float(x[al.DIV_RES + c]))) & 0xF) << (4 * c)
        mod |= (int(round(float(x[al.MOD_RES + c]))) & 0xF) << (4 * c)
    return div & 0xFFFFFFFF, mod & 0xFFFFFFFF


def run_unrolled(a_val, b_val):
    """Reference: apply the UNROLLED divmod block stack (the 262-block version)."""
    L = PureForwardCompleteLayout(8, n_heads=N_ROLES + 3)
    A.extend_layout_for_alu32(L)
    dim = L.D
    A._ONE = L.ONE
    al = L.ALU32
    blocks = A.compile_divmod_blocks(L, dim)
    x = torch.zeros(dim, dtype=torch.float64)
    x[L.ONE] = 1.0
    for c in range(8):
        x[L.STACK0 + c] = (a_val >> (4 * c)) & 0xF
        x[L.AX + c] = (b_val >> (4 * c)) & 0xF
    for _n, s in blocks:
        x = apply_spec(x, {k: v.to(torch.float64) for k, v in s.items()})
    div = mod = 0
    for c in range(8):
        div |= (int(round(float(x[al.DIV_RES + c]))) & 0xF) << (4 * c)
        mod |= (int(round(float(x[al.MOD_RES + c]))) & 0xF) << (4 * c)
    return div & 0xFFFFFFFF, mod & 0xFFFFFFFF


def main():
    n_heads = N_ROLES + 3
    L = PureForwardCompleteLayout(8, n_heads=n_heads)
    A.extend_layout_for_alu32(L, recurrent_divmod=True)
    dim = L.D
    A._ONE = L.ONE
    unique, apply_names = A.compile_divmod_blocks_recurrent(L, dim, n_iters=8)
    unique = [(n, {k: v.to(torch.float64) for k, v in s.items()}) for n, s in unique]

    cases = [(0, 1), (1, 1), (7, 2), (100, 7), (999, 7), (1000, 13), (255, 16),
             (65535, 255), (0xFFFFFFFF, 3), (123456, 789), (5, 0), (0, 0),
             (2**31, 2), (2**32 - 1, 15), (42, 42), (41, 42), (1000000, 1),
             (720, 6), (84, 5), (10**9, 7), (0xFFFFFFFF, 0xFFFFFFFF),
             (0xDEADBEEF, 0x1234), (256, 256), (65536, 65537)]
    ok_ref = 0        # byte-identity vs the UNROLLED block stack (the real gate)
    ok_num = 0        # sanity vs divmod32 (informational; b==0 mod differs by design)
    for a_val, b_val in cases:
        div, mod = run(a_val, b_val, L, dim, unique, apply_names)
        udiv, umod = run_unrolled(a_val, b_val)
        eq, em = divmod32(a_val, b_val)
        ref = (div == udiv and mod == umod)
        num = (div == eq and mod == em)
        ok_ref += ref
        ok_num += num
        flag = "OK " if ref else "MISMATCH"
        print(f"[{flag}] {a_val} /% {b_val} -> rec(div={div},mod={mod}) "
              f"unrolled(div={udiv},mod={umod}) divmod32(div={eq},mod={em})")
    print(f"\n{ok_ref}/{len(cases)} recurrent == UNROLLED (byte-identity gate)")
    print(f"{ok_num}/{len(cases)} recurrent == divmod32 (b==0 mod=0 by ISA design)")
    sys.exit(0 if ok_ref == len(cases) else 1)


if __name__ == "__main__":
    main()
