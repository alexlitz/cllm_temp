"""Tests for the FULL-native compact-ALU VM run via conditional block sparsity
(``full_native_fast``).

The full efficient-ALU recurrent build is deep (291 applied layers, ~14.7 GB dense
FFN) and its active-unit probe is minutes long, so the end-to-end byte-exact test
is OPT-IN (``C4_RUN_FULL_NATIVE=1``, GPU) — it builds the artifact, verifies the
ENTIRE ISA byte-exact through the conditional-block model vs ``isa.interpret``, and
is machine-safe (dense weights stay on CPU, only the gathered active block is on the
GPU; the conftest RSS watchdog guards the ~30 GB build peak).

The always-on tests are cheap: the coverage-corpus + log-sink reference are pure
Python / tiny, so they run in the default suite.
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa
from c4_min import full_native_fast as FNF


# ---------------------------------------------------------------------------
# ALWAYS-ON cheap checks (no model build).
# ---------------------------------------------------------------------------
def test_divide_integration_state_is_fp32():
    """The shipped fast VM's divide is the fp32 base-16 recurrent long division —
    zero fp64 params, byte-exact through the real forward (the fp64 log-sink divide
    has been retired from the production path)."""
    st = FNF.divide_integration_state()
    assert st["fp64_params"] == 0, st
    assert st["model_dtype"] == "torch.float32", st


def test_coverage_corpus_fits_code_size():
    """Every value-sweep coverage program fits the built VM's code table width so
    the active-unit probe can overlay it (the IndexError guard)."""
    for cs in (24, 32):
        corpus = FNF._coverage_corpus(code_size=cs)
        assert corpus, "empty corpus"
        assert all(len(prog) <= cs for prog in corpus), \
            f"a corpus program exceeds code_size={cs}"


def test_full_native_ops_cover_the_isa():
    """The native op set covers every ALU / mem / control / function opcode (no
    subroutine dispatch — all NATIVE)."""
    ops = set(FNF.FULL_NATIVE_OPS)
    for op in (isa.IMM, isa.PSH, isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
               isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
               isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
               isa.LI, isa.SI, isa.LC, isa.SC,
               isa.JMP, isa.BZ, isa.BNZ, isa.JSR, isa.ENT, isa.LEV, isa.HALT):
        assert op in ops, isa.NAMES[op]


# ---------------------------------------------------------------------------
# OPT-IN end-to-end byte-exact (heavy: full build + probe on GPU).
# ---------------------------------------------------------------------------
_RUN = os.environ.get("C4_RUN_FULL_NATIVE") == "1"
_DEVICE = os.environ.get("C4_FULL_NATIVE_DEVICE", "cuda:1")


@pytest.mark.skipif(not _RUN,
                    reason="opt-in: set C4_RUN_FULL_NATIVE=1 (GPU) to run the "
                           "full efficient-ALU recurrent build + probe (~15 min)")
def test_full_isa_byte_exact_conditional():
    """The ENTIRE ISA runs BYTE-EXACT through the conditional-block model vs
    ``isa.interpret`` — all ops NATIVE, compact ALU, conditional-sparsity run."""
    bundle = FNF.build_full_native_fast(device=_DEVICE, verbose=True)
    # the dense FFN is big and must NOT be materialised on the GPU.
    assert bundle.dense_gb > 10.0                      # deep/wide (~14.7 GB)
    assert bundle.active_gb * 1024 < 1000.0            # active block is a few hundred MB
    summary = FNF.verify_full_isa(bundle, verbose=True)
    assert summary["pass"] == summary["total"], summary["fails"]
