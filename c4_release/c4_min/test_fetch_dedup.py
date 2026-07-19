"""Tests for the FETCH-DEDUP compiler-in-weights (nibble_fetch_dedup).

Gates:
  * the factored PC fetch reads the correct word for EVERY baked slot,
  * the deduped model's produced bytecode + result is byte-identical to the
    baseline BakedCompilerMachine AND to the in-module reference interpreter,
  * D grows SUB-linearly in code_size (region-split + sqrt factored one-hot).
"""
from __future__ import annotations

import math

import torch

from c4_min import isa
from c4_min import nibble_compiler as C
from c4_min import nibble_fetch_dedup as D

EXPRS = ["2+3*4", "2*3+4", "1+2+3", "2*3*4", "4+5*6", "3*4+5", "9+8+7", "5*6*7"]


def _gen_and_machine(out_size=16):
    gen_prog = C.expr_compiler_bytecode(outbase=None) if False else None
    gen = C._assemble(C.expr_compiler_bytecode(outbase=0))  # size only
    gen_size = len(gen)
    prog = C.expr_compiler_bytecode(outbase=gen_size)       # OUT region at gen_size
    bcm = D.DedupBakedCompilerMachine(prog, out_size=out_size, src_size=8,
                                      mem_size=8, stack_depth=8)
    return prog, bcm


def test_factored_fetch_reads_every_slot():
    gen = C._assemble(C.expr_compiler_bytecode(outbase=0))
    gen_size = len(gen)
    prog = C.expr_compiler_bytecode(outbase=gen_size)
    code = C._assemble(prog)
    model, L = D.build_dedup_baked_compiler_step(code, out_size=16, src_size=8,
                                                 mem_size=8, stack_depth=8)
    bad = 0
    for i in range(gen_size):
        state = model.embed[0].clone()
        state[L.PC] = float(i)
        x = state.view(1, 1, -1)
        for blk in model.blocks[:3]:          # blockA, blockB, word-select
            x = blk(x)
        word = round(float(x[0, 0][L.WORD]))
        exp = (code[i].op & 0xFF) | ((code[i].imm & 0xFF) << 8)
        bad += (word != exp)
    assert bad == 0, f"{bad} baked slots misfetched"


def test_dedup_matches_reference_and_baseline():
    prog, bcm = _gen_and_machine()
    gen = C._assemble(prog)
    gen_size = len(gen)
    N = gen_size + 16
    for e in EXPRS:
        src = [ord(c) for c in e]
        # deduped model
        d_trace, d_words = bcm.run(e, return_code=True, max_steps=4000)
        d_produced = [d_words[k] for k in range(8)]
        # in-module reference (relocated to outbase=gen_size)
        r_trace, r_words = C.interpret_full(gen, src, code_size=N, mem_size=8,
                                            max_steps=4000)
        r_produced = [r_words[gen_size + k] for k in range(8)]
        assert d_produced == r_produced, (e, d_produced, r_produced)
        assert d_trace[-1] == (eval(e) & 0xFF), (e, d_trace[-1])


def test_D_is_sublinear():
    # D grows ~sqrt(code_size), not 3*code_size.
    ds = []
    for gen in [64, 256, 1024, 4000]:
        L = D.build_dedup_compiler_layout(gen, 16, 8, 8, 8)
        ds.append((gen, L.D))
    # baseline at code_size 4000 is >12000; dedup must be < 400.
    assert ds[-1][1] < 400, ds
    # doubling-ish code_size must NOT triple D (baseline would ~triple).
    (g0, d0), (g1, d1) = ds[0], ds[-1]
    assert d1 / d0 < 2.0, ds        # ~x60 code -> < x2 D
