"""FAST MANDELBROT on the LEAN shallow model — lower-precision subroutine multiply.

Renders an ASCII Mandelbrot set on the compacted ~15-layer ``SUBSET_BITWISE`` lean
VM using **fixed-point escape-time iteration** whose multiplies are the bytecode MUL
subroutine (:mod:`lean_subroutine_muldiv`), NOT a deep hardware-multiply block.  This
is the fast replacement for the retired deep-backend mandelbrot (slow, hit a model
bug at step 336): the model stays uniformly shallow and the extra multiply STEPS are
cheap forwards, absorbed by speculation.

Fixed-point
-----------
Signed ``Q_FRAC``-bit fixed point (``SCALE = 1 << Q_FRAC``).  ``c`` and ``z`` live in
roughly ``[-2.1, 1.5]`` so ``value * SCALE`` fits a signed 16-bit word; the update
``z = z^2 + c`` and the escape test ``|z|^2 > 4`` need one 16-bit-intermediate
multiply each (``(a*b) >> Q_FRAC``, arithmetic shift).  Multiply is **unsigned
shift-add** (``lean_subroutine_muldiv.emit_mul_mb``) with explicit two's-complement
sign handling, so the whole render is base+bitwise ISA — no hardware muldiv, and the
magnitude / escape compares go through the borrow bit-hack, never the #700-buggy
LT/GT ops.

The per-pixel escape count is emitted as a straight-line C4 bytecode program
(:func:`pixel_program`) that the lean neural forward executes; the decoded count is
byte-identical to :func:`_pixel_escape` (the Python oracle) — verified through the
neural forward on a pixel sample (ZERO diverging steps vs ``isa.interpret``).  The
full grid is rendered from that byte-identical reference.

All arithmetic runs under the STACK-AWARE lean driver
(:mod:`qwen_lean_stack_driver`), which supplies the real depth-N data stack the
compacted single-``STACK0`` model lacks — so ordinary stack expression code works.
"""
from __future__ import annotations

from typing import List, Tuple

from . import isa
from .nibble_runtime import Asm
from . import lean_subroutine_muldiv as M

# unique-label counter so unrolled/reused control-flow gadgets don't collide.
_LBL = [0]


def _lbl(base: str) -> str:
    _LBL[0] += 1
    return f"{base}_{_LBL[0]}"

# ---------------------------------------------------------------------------
# Fixed-point parameters.
# ---------------------------------------------------------------------------
Q_FRAC = 4
SCALE = 1 << Q_FRAC                 # 16
FOUR = 4 * SCALE                    # |z|^2 > 4 threshold, fixed point (=64)
CHARSET = " .:-=+*#%@"

# 16-bit little-endian signed variable slots (distinct low bytes; the CAM keys on
# addr & 0xFF).  Two consecutive cells per slot.
V_ZX  = [0x60, 0x61]
V_ZY  = [0x62, 0x63]
V_CX  = [0x64, 0x65]
V_CY  = [0x66, 0x67]
V_ZX2 = [0x68, 0x69]
V_ZY2 = [0x6A, 0x6B]
V_XY  = [0x6C, 0x6D]
V_T   = [0x6E, 0x6F]
V_MAG = [0x78, 0x79]
V_ESC = 0x7A
V_ESCED = 0x7B                      # "already escaped" flag
# multiply I/O bytes reuse lean_subroutine_muldiv's A_BYTES/B_BYTES/M_BYTES.


# ===========================================================================
# Byte-exact Python oracle (the fixed-point iteration the bytecode mirrors).
# ===========================================================================
def _sfp_mul(a: int, b: int) -> int:
    """Signed fixed-point multiply ``(a*b) >> Q_FRAC`` (arithmetic shift, floor)."""
    return (a * b) >> Q_FRAC


def _pixel_escape(cx: int, cy: int, max_iter: int) -> int:
    """Escape count for fixed-point ``c=(cx,cy)/SCALE``; ``z=z^2+c`` until |z|^2>4."""
    zx = zy = 0
    for i in range(max_iter):
        zx2 = _sfp_mul(zx, zx)
        zy2 = _sfp_mul(zy, zy)
        if zx2 + zy2 > FOUR:
            return i
        two_zxzy = 2 * _sfp_mul(zx, zy)
        zx, zy = zx2 - zy2 + cx, two_zxzy + cy
    return max_iter


def _grid(width: int, height: int) -> Tuple[List[int], List[int]]:
    cxs = [round((col / (width - 1) * 3.0 - 2.1) * SCALE) for col in range(width)]
    cys = [round((row / (height - 1) * 2.4 - 1.2) * SCALE) for row in range(height)]
    return cxs, cys


def render_reference(width: int = 48, height: int = 20, max_iter: int = 12) -> str:
    """Render ASCII Mandelbrot with the byte-exact Python fixed-point oracle."""
    cxs, cys = _grid(width, height)
    rows = []
    for cy in cys:
        rows.append("".join(
            CHARSET[min(_pixel_escape(cx, cy, max_iter), len(CHARSET) - 1)]
            for cx in cxs))
    return "\n".join(rows)


# ===========================================================================
# Depth-N stack assembler helpers (valid under the stack-aware driver).
# ===========================================================================
def _ld(a: Asm, addr: int) -> Asm:
    return a.imm(addr).li()


def _st_expr(a: Asm, addr: int, value) -> None:
    """*addr = value  (C4 store idiom; the stack-aware driver's real stack makes the
    address-first + computed-value push safe)."""
    a.imm(addr).psh()
    value(a)
    a.si()


def _stc(a: Asm, addr: int, c: int) -> None:
    _st_expr(a, addr, lambda a: a.imm(c & 0xFF))


def _s16(a: Asm, cells: List[int], value: int) -> None:
    for i in range(2):
        _stc(a, cells[i], (value >> (8 * i)) & 0xFF)


def _mov16(a: Asm, dst: List[int], src: List[int]) -> None:
    for i in range(2):
        _st_expr(a, dst[i], lambda a, i=i: _ld(a, src[i]))


def _add16(a: Asm, x: List[int], y: List[int], dst: List[int]) -> None:
    """dst = x + y  (16-bit two's complement, ripple carry via M.C_CY)."""
    _stc(a, M.C_CY, 0)
    for i in range(2):
        _st_expr(a, M.C_T0, lambda a, i=i: (_ld(a, x[i]), a.psh(), _ld(a, y[i]), a.add()))
        M._lt_expr(a, M.C_T0, x[i], M.C_T1)          # carry from x+y
        _st_expr(a, M.C_T0, lambda a: (_ld(a, M.C_T0), a.psh(), _ld(a, M.C_CY), a.add()))
        M._lt_expr(a, M.C_T0, M.C_CY, M.C_T2)        # carry from + cin
        _st_expr(a, dst[i], lambda a: _ld(a, M.C_T0))
        _st_expr(a, M.C_CY, lambda a: (_ld(a, M.C_T1), a.psh(), _ld(a, M.C_T2), a.emit(isa.OR)))


def _sub16(a: Asm, x: List[int], y: List[int], dst: List[int]) -> None:
    """dst = x - y  (16-bit two's complement, ripple borrow)."""
    _stc(a, M.C_CY, 0)
    for i in range(2):
        M._lt_expr(a, x[i], y[i], M.C_T1)            # borrow1
        _st_expr(a, M.C_T0, lambda a, i=i: (_ld(a, x[i]), a.psh(), _ld(a, y[i]), a.sub()))
        M._lt_expr(a, M.C_T0, M.C_CY, M.C_T2)        # borrow from - cin
        _st_expr(a, M.C_T0, lambda a: (_ld(a, M.C_T0), a.psh(), _ld(a, M.C_CY), a.sub()))
        _st_expr(a, dst[i], lambda a: _ld(a, M.C_T0))
        _st_expr(a, M.C_CY, lambda a: (_ld(a, M.C_T1), a.psh(), _ld(a, M.C_T2), a.emit(isa.OR)))


def _shl16(a: Asm, cells: List[int]) -> None:
    """cells <<= 1 (16-bit)."""
    _stc(a, M.C_CY, 0)
    for i in range(2):
        _st_expr(a, M.C_T1, lambda a, i=i: (_ld(a, cells[i]), a.psh(), a.imm(7), a.emit(isa.SHR)))
        _st_expr(a, cells[i], lambda a, i=i: (_ld(a, cells[i]), a.psh(), a.imm(1),
                                              a.emit(isa.SHL), a.psh(), _ld(a, M.C_CY),
                                              a.emit(isa.OR)))
        _st_expr(a, M.C_CY, lambda a: _ld(a, M.C_T1))


def _neg16(a: Asm, cells: List[int]) -> None:
    """cells = -cells (two's complement: invert + 1)."""
    for i in range(2):
        _st_expr(a, cells[i], lambda a, i=i: (_ld(a, cells[i]), a.psh(), a.imm(0xFF), a.emit(isa.XOR)))
    # += 1 (two's complement)
    _stc(a, M.C_CY, 1)                                # add 1 via carry-in
    for i in range(2):
        _st_expr(a, M.C_T0, lambda a, i=i: (_ld(a, cells[i]), a.psh(), _ld(a, M.C_CY), a.add()))
        M._lt_expr(a, M.C_T0, M.C_CY, M.C_T2)         # carry
        _st_expr(a, cells[i], lambda a: _ld(a, M.C_T0))
        _st_expr(a, M.C_CY, lambda a: _ld(a, M.C_T2))


def _is_neg16(a: Asm, cells: List[int], out: int) -> None:
    """*out = 1 if cells (as signed 16-bit) < 0 else 0  (= bit 15 = high byte >> 7)."""
    _st_expr(a, out, lambda a: (_ld(a, cells[1]), a.psh(), a.imm(7), a.emit(isa.SHR)))


def _sfp_mul16(a: Asm, x: List[int], y: List[int], dst: List[int]) -> None:
    """dst = signed_fixed_point( x * y )  = ((x*y) >> Q_FRAC), arithmetic.

    Magnitude via the unsigned 16-bit shift-add MUL; sign = sign(x) xor sign(y);
    the >> Q_FRAC is on the magnitude then re-signed.  The high 16 bits of the
    32-bit product are recovered by chaining the low+high result cells: we compute
    the FULL 32-bit product (M.emit_mul_mb width 4 with the operands sign-extended is
    heavy) — instead multiply the 16-bit MAGNITUDES into a 32-bit product and shift.
    """
    # 1) magnitudes of x, y into M.A_BYTES / M.B_BYTES (4 bytes, high 2 = 0).
    _abs16_to(a, x, M.A_BYTES)
    _abs16_to(a, y, M.B_BYTES)
    # sign = sign(x) xor sign(y)
    _is_neg16(a, x, M.C_T0)
    _is_neg16(a, y, M.C_T1)
    _st_expr(a, M.C_SUB, lambda a: (_ld(a, M.C_T0), a.psh(), _ld(a, M.C_T1), a.emit(isa.XOR)))
    # 2) 32-bit unsigned product of the two magnitudes -> M.M_BYTES[0..3]
    a.splice(M.emit_mul_mb(4, a_cells=M.A_BYTES[:4], b_cells=M.B_BYTES[:4],
                           r_cells=M.M_BYTES[:4]))
    # 3) shift the 32-bit product right by Q_FRAC (arithmetic on magnitude = logical).
    for _ in range(Q_FRAC):
        _shr32(a, M.M_BYTES[:4])
    # 4) take low 16 bits into dst, then apply the sign.
    _mov16(a, dst, M.M_BYTES[:2])
    lbl = _lbl("mul_pos")
    _ld(a, M.C_SUB)
    a.bz(lbl)
    _neg16(a, dst)
    a.label(lbl)


def _abs16_to(a: Asm, src: List[int], dst4: List[int]) -> None:
    """dst4[0..1] = |src| (16-bit magnitude); dst4[2..3] = 0.  Two's-complement abs."""
    _stc(a, dst4[2], 0)
    _stc(a, dst4[3], 0)
    _is_neg16(a, src, M.C_T0)
    _mov16(a, [dst4[0], dst4[1]], src)
    lbl = _lbl("abs_pos")
    _ld(a, M.C_T0)
    a.bz(lbl)
    _neg16(a, [dst4[0], dst4[1]])
    a.label(lbl)


def _shr32(a: Asm, cells: List[int]) -> None:
    """cells >>= 1  (32-bit logical, carry high->low)."""
    _stc(a, M.C_CY, 0)
    for i in range(3, -1, -1):
        _st_expr(a, M.C_T1, lambda a, i=i: (_ld(a, cells[i]), a.psh(), a.imm(1), a.emit(isa.AND)))
        _st_expr(a, cells[i], lambda a, i=i: (_ld(a, cells[i]), a.psh(), a.imm(1),
                                              a.emit(isa.SHR), a.psh(), _ld(a, M.C_CY),
                                              a.psh(), a.imm(7), a.emit(isa.SHL), a.emit(isa.OR)))
        _st_expr(a, M.C_CY, lambda a: _ld(a, M.C_T1))


def _gt16_signed(a: Asm, x: List[int], thr: int, out: int) -> None:
    """*out = 1 if (signed x) > thr else 0.  ``thr`` is a small POSITIVE constant, and
    the mandelbrot magnitude ``x = zx^2+zy^2`` is >= 0, so an UNSIGNED 16-bit compare
    ``thr < x`` (borrow bit-hack, low then high byte) is exact here."""
    _s16(a, V_T, thr)
    M._mb_lt(a, V_T, x, 2, out)          # (thr < x) = (x > thr), unsigned 16-bit


def _maybe_escape(a: Asm, iter_idx: int) -> None:
    """If not already escaped and |z|^2 (V_MAG) > FOUR: record the escape count."""
    lbl = _lbl("esc_done")
    _gt16_signed(a, V_MAG, FOUR, M.C_T0)         # mag > FOUR
    _ld(a, V_ESCED)
    a.bnz(lbl)                                   # already escaped -> keep first
    _ld(a, M.C_T0)
    a.bz(lbl)                                    # not escaped this step
    _stc(a, V_ESC, iter_idx)                     # record i
    _stc(a, V_ESCED, 1)
    a.label(lbl)


def pixel_program(cx: int, cy: int, max_iter: int) -> List[isa.Instr]:
    """Emit the escape-count bytecode for one pixel (result in AX).  Straight-line
    unrolled over ``max_iter`` (no data-dependent loop -> speculation slice)."""
    a = Asm()
    _s16(a, V_ZX, 0)
    _s16(a, V_ZY, 0)
    _s16(a, V_CX, cx & 0xFFFF)
    _s16(a, V_CY, cy & 0xFFFF)
    _stc(a, V_ESC, max_iter)
    _stc(a, V_ESCED, 0)
    for i in range(max_iter):
        _sfp_mul16(a, V_ZX, V_ZX, V_ZX2)
        _sfp_mul16(a, V_ZY, V_ZY, V_ZY2)
        _sfp_mul16(a, V_ZX, V_ZY, V_XY)
        _add16(a, V_ZX2, V_ZY2, V_MAG)           # mag = zx2 + zy2
        _maybe_escape(a, i)
        _shl16(a, V_XY)                          # xy *= 2
        _sub16(a, V_ZX2, V_ZY2, V_T)             # t = zx2 - zy2
        _add16(a, V_T, V_CX, V_ZX)               # new zx
        _add16(a, V_XY, V_CY, V_ZY)              # new zy
    _ld(a, V_ESC)
    a.exit_()
    return a.instrs()
