"""A tiny fixed-point Mandelbrot rendered THROUGH THE ONE UNIFIED MODEL —
neural verification via ``model.forward``.

The existing ``tests/test_programs.py::test_mandelbrot`` renders a 40x20 grid via
the DRAFT VM (``SpeculativeVM(validate_ratio=0.0)``) — it NEVER goes through the
transformer forward.  This module runs a TINY grid through the ACTUAL c4_min
``model.forward`` (the streaming sparse build + KV-cached driver with eviction),
capturing the ``printf`` (PRTF, op 33) stdout stream the model itself decodes,
and asserts it is byte-exact against the blessed reference interpreter
(``ref_interpret``, the unsigned-32-bit oracle the KV-cached driver is
byte-identical to).

Bounded scope (honest): a full render is a ~200-hour perf wall (~240k+ VM steps
at ~2.8s/step through the neural forward).  Even a single mandelbrot CELL is
~300 VM steps (the fixed-point iterate + escape test + PRTF), so the grid here is
deliberately tiny.  This is a REAL, byte-exact "mandelbrot on the neural VM"
demonstration, NOT the full render.  See
``docs/MANDELBROT_ON_THE_NEURAL_VM_2026_07_20.md``.

The program is compiled by the REAL c4_min C compiler (``src.compiler.compile_c``)
from ``_mandel_src.mandel_c``, which keeps every ``IMM`` immediate <= 255 (larger
constants are built via MUL/ADD of byte literals — the neural model and the
byte-masking ``ref_interpret`` only agree for byte-sized immediates) and uses only
the verified op set (IMM/PSH/ADD/SUB/MUL/DIV/LT/GT/EQ + JSR/ENT/LEV framing +
LEA/LI/SI locals + PRTF).

Run (heavy — bakes the streaming model + a few-hundred-step neural run):
    OMP_NUM_THREADS=4 C4_RUN_NEURAL_MANDELBROT=1 \
        python -m pytest c4_min/test_mandelbrot_neural.py -x -s
"""
from __future__ import annotations

import os

import pytest

from c4_min import isa
from c4_min._mandel_src import mandel_c


def _compile(width, height, maxiter):
    """Compile the tiny mandelbrot via the REAL c4_min C compiler and translate
    to the ISA the model executes.  Asserts no IMM > 255 leaked in (which would
    diverge the byte-masking reference from the model)."""
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    bytecode, data = compile_c(mandel_c(width, height, maxiter))
    code = bytecode_to_isa(bytecode)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked (would diverge from ref): {over}"
    return code, data


def test_mandelbrot_source_ops_are_verified_subset():
    """The compiled mandelbrot uses only the op set the neural model verifies
    (arithmetic + compare + framing + PRTF): no MALC/FREE/MSET/MCMP/OPEN/READ,
    and every IMM is byte-sized.  Cheap (compile-only) — always runs."""
    code, _ = _compile(2, 2, 3)
    used = {i.op for i in code}
    allowed = {
        isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
        isa.LT, isa.GT, isa.LE, isa.GE, isa.EQ, isa.NE,
        isa.LI, isa.LC, isa.SI, isa.SC,
        isa.JMP, isa.BZ, isa.BNZ, isa.JSR, isa.ENT, isa.ADJ, isa.LEV,
        isa.PRTF, isa.NOP, isa.HALT,
    }
    assert used <= allowed, f"unexpected ops: {sorted(used - allowed)}"
    assert isa.PRTF in used, "the render must emit via PRTF"
    assert isa.MUL in used and isa.DIV in used, "fixed-point needs MUL + DIV"


def test_mandelbrot_stays_nonnegative():
    """Every intermediate AX value the reference produces is NON-NEGATIVE (never
    wraps to the 2^31..2^32 range).  This is load-bearing for byte-exactness: a
    wrapped-negative in AX corrupts the model's next ``LEA`` (the AX 0xFF high-byte
    leak — verified in isolation).  The generator's ``if (b > t)`` escape guard
    keeps the ``zx2 - zy2`` subtraction non-negative; this test locks that in.
    Cheap (reference only) — always runs."""
    from c4_min.nibble_pure_forward_complete import ref_interpret

    code, _ = _compile(2, 1, 3)
    trace = ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF)
    wrapped = [v for v in trace if v >= 2 ** 31]
    assert not wrapped, f"{len(wrapped)} wrapped-negative AX values (would corrupt LEA)"


def test_mandelbrot_reference_shape_is_mixed():
    """The reference render of the tiny grid is a genuine mix of inside ('*') and
    escaped (' ') cells — a filled region, not a solid block.  Cheap (reference
    only) — always runs; documents the exact bytes the neural run must reproduce."""
    from c4_min.nibble_pure_forward_complete import ref_interpret

    code, _ = _compile(2, 1, 3)
    out = []
    ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF, out=out)
    text = "".join(chr(b) if b != 10 else "\n" for b in out)
    assert "*" in text and " " in text, f"expected a mix, got:\n{text!r}"
    assert out.count(10) == 1, "one newline per row (H=1)"


@pytest.mark.skipif(
    not os.environ.get("C4_RUN_NEURAL_MANDELBROT"),
    reason="heavy: bakes the streaming model + a few-hundred-step neural run "
           "(set C4_RUN_NEURAL_MANDELBROT=1)",
)
@pytest.mark.parametrize("width,height,maxiter", [(2, 1, 3)])
def test_mandelbrot_model_forward_byte_exact(width, height, maxiter):
    """The tiny mandelbrot printed by the ACTUAL streaming ``model.forward``
    (KV-cached driver, eviction ON) equals the reference interpreter's PRTF
    stdout, byte-for-byte — a real "mandelbrot on the neural VM" render.

    Memory: streaming build (~5 GB peak) + eviction (prune_interval=60) holds the
    whole run flat; ``OMP_NUM_THREADS=4``, single process.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from c4_min.nibble_pure_forward_complete import ref_interpret
    from c4_min.lib_neural import build_lib_model_streaming
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached

    code, data = _compile(width, height, maxiter)

    ref_out = []
    ref_tr = ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF, out=ref_out)
    n_ref_steps = len(ref_tr)

    sparse, L, _ = build_lib_model_streaming(
        code_size=max(len(code) + 2, 64), recurrent_divmod=True, addr32=True)

    out, stats = [], {}
    run_pure_forward_cached(
        sparse, L, code, max_steps=n_ref_steps + 6, mask=0xFFFFFFFF,
        evict=True, prune_interval=60, out=out, stats=stats, data_seg=data)

    def render(bs):
        return "".join(chr(b) if b != 10 else "\n" for b in bs)

    assert out == ref_out, (
        f"neural PRTF stdout != reference\n"
        f"--- neural ---\n{render(out)}\n--- ref ---\n{render(ref_out)}"
    )
    # sanity: a real mix (filled region), and the model actually stepped.
    assert "*" in render(out) and " " in render(out), "expected a mandelbrot mix"
    assert stats.get("steps", 0) >= n_ref_steps, "model under-stepped"


if __name__ == "__main__":
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ["C4_RUN_NEURAL_MANDELBROT"] = "1"
    test_mandelbrot_source_ops_are_verified_subset()
    test_mandelbrot_stays_nonnegative()
    test_mandelbrot_reference_shape_is_mixed()
    test_mandelbrot_model_forward_byte_exact(2, 1, 3)
    print("all mandelbrot-neural checks passed")
    sys.exit(0)
