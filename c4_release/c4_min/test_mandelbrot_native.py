"""CPU byte-exactness tests for the NATIVE-MUL mandelbrot (``mandelbrot_native``).

Every check runs on the CPU word reference
(``nibble_pure_forward_complete.ref_interpret`` at ``mask=0xFFFFFFFF`` — the
blessed oracle the full-native neural forward is byte-identical to); the neural
model is NEVER built here.  The gate is: the native-MUL escape counts match
``lean_mandelbrot._pixel_escape`` (the software-shift-add-mul render) byte for
byte, the pixel program is a LOOP (constant static size in ``max_iter``, not a
straight-line unroll), and the PPM stream is well-formed and round-trips to the
escape-count reference.

    python -m pytest c4_min/test_mandelbrot_native.py -q
"""
from __future__ import annotations

import pytest

from c4_min import mandelbrot_native as MN
from c4_min import lean_mandelbrot as LM
from c4_min import isa


def test_grid_mapping_matches_lean():
    """The fixed-point grid mapping is IDENTICAL to lean_mandelbrot, so escape
    counts are directly comparable."""
    for W, H in [(24, 12), (16, 8), (48, 20)]:
        assert MN.grid(W, H) == LM._grid(W, H)


def test_program_uses_native_mul_and_is_a_loop():
    """The pixel program uses the NATIVE MUL opcode, no software shift-add mul, and
    is a LOOP whose static size is CONSTANT in max_iter (not an unroll)."""
    prog = MN.pixel_program_native(-10, 5, 12)
    ops = {isa.NAMES[i.op] for i in prog}
    assert "MUL" in ops and "DIV" in ops
    assert "JSR" in ops and "BZ" in ops           # subroutine call + loop back-edge
    assert "SHL" not in ops                        # no software shift-add multiply
    counts = {mi: len(MN.pixel_program_native(-10, 5, mi)) for mi in (8, 12, 20)}
    assert len(set(counts.values())) == 1, f"not a loop (size grows): {counts}"


@pytest.mark.parametrize("W,H,MI", [(24, 12, 8), (24, 12, 12), (24, 12, 20)])
def test_escape_counts_byte_exact_vs_lean(W, H, MI):
    """Native-MUL escape counts == lean_mandelbrot._pixel_escape for every pixel.
    A single mismatch is the signed-truncate-toward-zero bug."""
    cxs, cys = MN.grid(W, H)
    diffs = []
    for cy in cys:
        for cx in cxs:
            vm = MN.escape_count_native(cx, cy, MI)
            ref = LM._pixel_escape(cx, cy, MI)
            if vm != ref:
                diffs.append((cx, cy, vm, ref))
    assert not diffs, f"{len(diffs)}/{W * H} pixels differ: {diffs[:8]}"


def test_signed_cross_term_truncates_toward_zero():
    """The load-bearing sign case: a NEGATIVE cross term must truncate toward zero
    (like _sfp_mul), not toward -inf.  Exercise operands that make zx*zy negative."""
    for zx, zy in [(-34, 17), (17, -34), (-57, 3), (3, -57), (-85, 5), (5, -85)]:
        # sfp result the native path yields (via a tiny standalone program)
        from c4_min.nibble_pure_forward_complete import ref_interpret
        a = MN.Asm()
        MN._stc(a, MN.A_ZX, (zx + MN.BIAS) & 0xFF)
        MN._stc(a, MN.A_ZY, (zy + MN.BIAS) & 0xFF)
        MN._call_sfp(a, MN.A_ZX, MN.A_ZY)
        a.exit_()
        MN._emit_sfp(a)
        got = MN._to_signed(ref_interpret(a.instrs(), max_steps=2000, mask=0xFFFFFFFF)[-1])
        assert got == MN._sfp_mul(zx, zy), (zx, zy, got, MN._sfp_mul(zx, zy))


def test_ppm_well_formed_and_matches_reference():
    """render_ppm_native emits a valid P6 stream of the right length that decodes to
    the escape-count reference grid."""
    W, H, MI = 24, 12, 12
    ppm = MN.render_ppm_native(W, H, MI)
    header = f"P6\n{W} {H}\n255\n".encode("ascii")
    assert ppm.startswith(header)
    assert len(ppm) == len(header) + W * H * 3
    w, h, pixels = MN.parse_ppm(ppm)
    assert (w, h) == (W, H)
    ref = MN.escape_grid_native(W, H, MI)
    for r in range(H):
        for c in range(W):
            assert pixels[r * W + c] == MN._palette(ref[r][c], MI)


def test_pixel_program_is_small():
    """The program fits a modest code_size (a LOOP, ~180 instrs), a ~300x reduction
    from the 53,963-instruction software-mul render — NOT a straight-line unroll."""
    n = len(MN.pixel_program_native(-10, 5, 8))
    assert n < 256, f"program too large for a modest code table: {n}"
    assert n < len(LM.pixel_program(-10, 5, 8)) // 100      # >100x smaller than sw-mul
