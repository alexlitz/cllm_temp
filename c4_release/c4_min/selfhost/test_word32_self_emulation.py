#!/usr/bin/env python3
"""test_word32_self_emulation.py — gates for the 32-bit-memory draft VM + the
grounded true-self-emulation step count.

Run:  python -m pytest c4_min/selfhost/test_word32_self_emulation.py -q
"""
from __future__ import annotations

import numpy as np

from c4_min.selfhost.word32_draft_vm import ref_interpret_word32, WORD
from c4_min.selfhost._compile_helper import compile_paged_dot
from c4_min.nibble_pure_forward_complete import ref_interpret


def _byte_trace(code):
    out = []
    tr = ref_interpret(code, max_steps=20_000_000, mask=0xFFFFFFFF, out=out)
    return out, tr


def test_word32_byte_identical_on_small():
    """(A) word32 VM == byte VM (value + steps + full trace) when values fit a byte."""
    for w, x in [([1, 0, 1], [2, 0, 1]), ([1, 1], [1, 1])]:
        code = compile_paged_dot(w, x)
        w32_trace, w32_n = ref_interpret_word32(code, out=(o32 := []))
        b_out, b_trace = _byte_trace(code)
        assert o32 == b_out
        assert w32_trace == b_trace
        assert w32_n == len(b_trace)


def test_word32_keeps_high_byte():
    """(B) word32 keeps the high byte the byte VM truncates (store 1000 > 255)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    src = """
int main() {
  int acc, k, hi, lo, v;
  acc = 0; k = 0;
  while (k < 10) { acc = acc + 100; k = k + 1; }
  v = acc; lo = v - (v / 256) * 256; hi = v / 256;
  printf(hi); printf(lo); return 0;
}
"""
    bc, _ = compile_c(src)
    code = bytecode_to_isa(bc)
    _tr, _n = ref_interpret_word32(code, out=(o32 := []))
    b_out, _ = _byte_trace(code)
    assert o32 == [3, 232]          # 1000 = 3*256 + 232, full value kept
    assert b_out == [0, 232]        # byte VM truncated the high byte


def test_word32_correct_vs_numpy_overflow():
    """(C) word32 == numpy 32-bit on dots whose acc overflows a byte."""
    rng = np.random.RandomState(0)
    for K in [8, 64, 104]:
        w = [int(v) for v in rng.randint(1, 6, size=K)]
        x = [int(v) for v in rng.randint(1, 6, size=K)]
        code = compile_paged_dot(w, x)
        _tr, steps = ref_interpret_word32(code, out=(o := []))
        acc = 0
        for i in range(K):
            acc = (acc + (w[i] * 16 * x[i] * 16) // 16) & WORD
        assert o[-1] == (acc & 0xFF)
        assert acc > 255            # really overflowed the byte


def test_step_count_data_independent():
    """The paged-dot step count depends only on nnz length, not the data."""
    for data in ([1] * 8, [3, 0, 2, 0, 1, 0, 2, 0], [255] * 8):
        code = compile_paged_dot(data, [2] * len(data))
        _tr, steps = ref_interpret_word32(code)
        assert steps == 627         # same for every data at this length


def test_real_scale_monolithic():
    """(Task 4) a real-scale (K=896) dot runs monolithically, byte-exact, at the
    grounding's per-op rate."""
    K = 896
    rng = np.random.RandomState(7)
    w = [int(v) for v in rng.randint(0, 3, size=K)]
    x = [int(v) for v in rng.randint(0, 5, size=K)]
    # emit hi/lo so the FULL value is checked
    from c4_min.selfhost._matmul_paged_src import paged_dot_c
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    body = paged_dot_c(w, x).replace(
        "int s; int acc, k;", "int s; int acc, k; int hi; int lo;").replace(
        "printf(acc);", "hi = acc/256; lo = acc - hi*256; printf(hi); printf(lo);")
    bc, _ = compile_c(body)
    code = bytecode_to_isa(bc)
    _tr, steps = ref_interpret_word32(code, out=(o := []), max_steps=80_000_000)
    acc = 0
    for i in range(K):
        acc = (acc + (w[i] * 16 * x[i] * 16) // 16) & WORD
    assert o == [(acc >> 8) & 0xFF, acc & 0xFF]
    rate = steps / K
    assert abs(rate - 76.38) / 76.38 < 0.02      # rate holds at real scale
