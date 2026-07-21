"""MUL / DIV / MOD as C4 BYTECODE SUBROUTINES on the LEAN shallow model.

This is the *lean* alternative to baking MUL/DIV/MOD into model DEPTH.  The
lean compacted Qwen VM (:mod:`qwen_full_vm`, subset ``SUBSET_BITWISE`` — base +
memory + cmp + bitwise + shift, ~15 layers, NO deep muldiv blocks) has no
hardware multiply/divide.  Instead — exactly like the runtime library
(``malloc``/``free``/``memset``/``memcmp`` in :mod:`nibble_runtime`, which are
"compiled from C into VM bytecode and execute entirely neurally", BLOG_SPEC
§687/§747) — MUL/DIV/MOD are emitted as **straight-line + looping base-ISA
bytecode**:

  * **MUL** — shift-and-add:  ``r=0; while y: if y&1: r+=x; x<<=1; y>>=1``
    (ops ``AND``/``BNZ``/``ADD``/``SHL``/``SHR`` only).
  * **DIV** — restoring long division, one bit per iteration, MSB first.  The
    ``remainder >= divisor`` test is done with a **borrow bit-hack** (see
    :func:`_lt_expr`) because the lean model's ORDERING comparisons (LT/GT/LE/GE)
    are the known pre-existing signed-compare bug (#700); the borrow-hack uses
    only ``SUB``/``AND``/``OR``/``XOR``/``SHR`` and is byte-exact over all 256².
  * **MOD** — the same division loop, returning the remainder (``§Mod``).

The complexity is thus **VM STEPS** (each a cheap shallow forward on the SAME
~15-layer model) rather than model depth.  Perfect-draft speculation
(:func:`qwen_lean_forward.speculative_run_lean`) batches those extra steps, so
the wall-clock cost of the extra iterations is largely absorbed.

Precision variants
------------------
The lean VM's value path is **8-bit** (``draft_program_lean``/``run_program_lean``
fold AX to ``& 0xFF``), so multi-byte values live in memory as consecutive bytes
and the subroutine does byte-by-byte arithmetic with an explicit carry/borrow:

  * :func:`emit_mul8` / :func:`emit_divmod8` — 8-bit operands (fewest iterations).
  * :func:`emit_mul_mb` / :func:`emit_divmod_mb` — 16- / 32-bit fixed width
    (``nbytes`` consecutive memory bytes, LSB first).

Every subroutine is pure base+bitwise ISA — it runs on the ``SUBSET_BITWISE``
lean model with ZERO new neural op, and stays inside the speculation slice
(no JSR/ENT/LEV), so :func:`qwen_lean_forward.draft_program_lean` drafts it for
free and the big-K speculative driver verifies the whole trace in a handful of
batched forwards.

Address discipline
------------------
The lean memory CAM keys stores/loads on the LOW byte of the address
(``run_program_lean`` compacts ``store_log`` by ``addr & 0xFF``), so all scratch
cells and multi-byte operand/result byte cells are chosen with DISTINCT low
bytes (see :data:`SCRATCH`).
"""
from __future__ import annotations

from typing import List

from . import isa
from .nibble_runtime import Asm

SHL, SHR, AND, OR, XOR = isa.SHL, isa.SHR, isa.AND, isa.OR, isa.XOR
EQ, NE = isa.EQ, isa.NE

# ---------------------------------------------------------------------------
# Scratch memory cells (distinct LOW bytes — the lean CAM keys on addr & 0xFF).
# ---------------------------------------------------------------------------
C_X   = 0x10        # 8-bit multiplier / dividend
C_Y   = 0x11        # 8-bit multiplicand / divisor
C_R   = 0x12        # 8-bit running result (product / remainder)
C_Q   = 0x13        # 8-bit quotient
C_I   = 0x14        # bit / byte loop counter
C_BIT = 0x15        # extracted / mask bit
C_T0  = 0x16        # general caller temporaries (multi-byte helpers)
C_T1  = 0x17
C_T2  = 0x18
C_LT  = 0x19        # the "<" predicate result
C_CY  = 0x1A        # carry / shift-out / borrow chain
C_SUB = 0x1B        # subtract-condition
# PRIVATE scratch for _lt_expr / _eq0_expr ONLY (the borrow-hack clobbers these
# every call, so they MUST NOT alias the general C_T* the multi-byte helpers use).
LT_D  = 0x1C        # d = x - y
LT_NX = 0x1D        # nx = ~x
LT_OR = 0x1E        # nx | y
LT_EZ = 0x1F        # eq0 zero-constant scratch
# PRIVATE scratch for the bit select/set helpers (must not alias the caller's
# ``out`` or the C_T* the loop body already uses).
B_IDX = 0x50        # byte index (i >> 3)
B_OFF = 0x51        # bit offset (i & 7)
B_MSK = 0x52        # full-byte match mask (0x00 / 0xFF)
B_VAL = 0x53        # per-iteration extracted bit / mask
B_MCH = 0x54        # per-iteration match flag (0 / 1)

A_BYTES = [0x20, 0x21, 0x22, 0x23]     # operand A bytes (LSB first)
B_BYTES = [0x28, 0x29, 0x2A, 0x2B]     # operand B bytes
Q_BYTES = [0x40, 0x41, 0x42, 0x43]     # quotient bytes
M_BYTES = [0x48, 0x49, 0x4A, 0x4B]     # remainder / product bytes

SCRATCH = {k: v for k, v in globals().items() if k.startswith("C_")}


# ---------------------------------------------------------------------------
# Small code-gen helpers on top of Asm (all leave AX = the value they compute).
# ---------------------------------------------------------------------------
def _ld(a: Asm, addr: int) -> None:
    """AX = *addr."""
    a.imm(addr).li()


def _st(a: Asm, addr: int, value) -> None:
    """*addr = value.  ``value`` is a callable(a) that leaves the value in AX in
    a stack-neutral way (C4 store idiom: push addr, compute value, SI)."""
    a.imm(addr).psh()
    value(a)
    a.si()


def _stc(a: Asm, addr: int, const: int) -> None:
    """*addr = const."""
    _st(a, addr, lambda a: a.imm(const & 0xFF))


def _binop(a: Asm, addr_x: int, op: int, addr_y: int) -> None:
    """AX = *addr_x  OP  *addr_y."""
    _ld(a, addr_x)
    a.psh()
    _ld(a, addr_y)
    a.emit(op)


def _binop_imm(a: Asm, addr_x: int, op: int, imm: int) -> None:
    """AX = *addr_x  OP  imm."""
    _ld(a, addr_x)
    a.psh()
    a.imm(imm & 0xFF)
    a.emit(op)


# ---------------------------------------------------------------------------
# The "<" predicate via borrow bit-hack (LT/GT/LE/GE are the #700 bug).
#
#   borrow_out(x - y) == 1  iff  x < y   (unsigned 8-bit)
#   d  = (x - y) & 0xFF ;  nx = x XOR 0xFF
#   (x < y) = ( ((nx & y) | (d & (nx | y))) >> 7 ) & 1
# Byte-exact over all 256×256 pairs (verified).  Result -> *out (0 or 1).
# ---------------------------------------------------------------------------
def _lt_expr(a: Asm, addr_x: int, addr_y: int, out: int) -> None:
    """*out = 1 if *addr_x < *addr_y else 0  (unsigned, borrow bit-hack).

    Uses ONLY the private LT_* scratch cells (never the general C_T*), so callers
    can freely hold working values in C_T0/C_T1/C_T2 across a _lt_expr call.
    ``out`` may be any cell EXCEPT the LT_* set (and not addr_x/addr_y)."""
    _st(a, LT_D, lambda a: _binop(a, addr_x, isa.SUB, addr_y))       # d = x - y
    _st(a, LT_NX, lambda a: _binop_imm(a, addr_x, XOR, 0xFF))        # nx = ~x
    _st(a, LT_OR, lambda a: _binop(a, LT_NX, OR, addr_y))            # nx | y

    def _val(a: Asm) -> None:
        _binop(a, LT_NX, AND, addr_y)       # nx & y
        a.psh()
        _binop(a, LT_D, AND, LT_OR)         # d & (nx|y)
        a.emit(OR)
        a.psh()
        a.imm(7)
        a.emit(SHR)
    _st(a, out, _val)


def _eq0_expr(a: Asm, addr_x: int, out: int) -> None:
    """*out = 1 if *addr_x == 0 else 0.  (0 == x) via NOT (0 < x)."""
    _stc(a, LT_EZ, 0)
    _lt_expr(a, LT_EZ, addr_x, out)         # (0 < x) = (x != 0)
    _st(a, out, lambda a: _binop_imm(a, out, XOR, 1))   # == 0


# ===========================================================================
# 8-bit MUL / DIV / MOD
# ===========================================================================
def emit_mul8(x_cell: int = C_X, y_cell: int = C_Y, out_cell: int = C_R) -> Asm:
    """Shift-and-add MUL8.  Reads *x_cell,*y_cell; leaves the 8-bit product in
    *out_cell and AX.  ``r=0; while y: if y&1: r+=x; x<<=1; y>>=1``."""
    a = Asm()
    _stc(a, out_cell, 0)
    a.label("mtop")
    _ld(a, y_cell)
    a.bz("mdone")
    _binop_imm(a, y_cell, AND, 1)           # y & 1
    a.bz("mskip")
    _st(a, out_cell, lambda a: _binop(a, out_cell, isa.ADD, x_cell))   # r += x
    a.label("mskip")
    _st(a, x_cell, lambda a: _binop_imm(a, x_cell, SHL, 1))            # x <<= 1
    _st(a, y_cell, lambda a: _binop_imm(a, y_cell, SHR, 1))            # y >>= 1
    a.jmp("mtop")
    a.label("mdone")
    _ld(a, out_cell)
    return a


def emit_mul8x8_to16(x_cell: int = C_X, y_cell: int = C_Y,
                     out_lo: int = C_R, out_hi: int = C_Q) -> Asm:
    """8-bit x 8-bit -> 16-bit product shift-and-add.  Reads *x_cell,*y_cell; leaves
    the low byte in *out_lo, the high byte in *out_hi (product = lo | hi<<8).

    Like :func:`emit_mul8` but the running accumulator is 16-bit (two cells) so the
    full 0..65025 product survives — the cheap multiply the fixed-point mandelbrot
    needs (~one mul8's worth of steps, no 4-byte machinery).  ``r=0; while y: if y&1:
    r += (x<<... )`` done as a 16-bit ``r += shifted_x`` with an 8-bit x kept in a
    two-cell shifted operand."""
    a = Asm()
    _stc(a, out_lo, 0)
    _stc(a, out_hi, 0)
    # shifted x lives in two cells: X_LO (starts = x), X_HI (starts = 0).
    _st(a, LT_D, lambda a: _ld(a, x_cell))       # reuse LT_D/LT_NX as x_lo/x_hi (mul does not call _lt_expr)
    _stc(a, LT_NX, 0)
    a.label("m16top")
    _ld(a, y_cell)
    a.bz("m16done")
    _binop_imm(a, y_cell, AND, 1)
    a.bz("m16skip")
    # r += (X_HI:X_LO)  (16-bit add)
    _st(a, LT_OR, lambda a: _binop(a, out_lo, isa.ADD, LT_D))          # lo sum
    _lt_expr_free(a, LT_OR, out_lo, LT_EZ)                              # carry = (sum < out_lo)
    _st(a, out_lo, lambda a: _ld(a, LT_OR))
    _st(a, out_hi, lambda a: (_binop(a, out_hi, isa.ADD, LT_NX)))      # hi += x_hi
    _st(a, out_hi, lambda a: (_binop(a, out_hi, isa.ADD, LT_EZ)))      # + carry
    a.label("m16skip")
    # shifted_x <<= 1  (16-bit)
    _st(a, C_CY, lambda a: _binop_imm(a, LT_D, SHR, 7))                # carry out of lo
    _st(a, LT_D, lambda a: _binop_imm(a, LT_D, SHL, 1))
    _st(a, LT_NX, lambda a: (_binop_imm(a, LT_NX, SHL, 1), a.psh(), _ld(a, C_CY), a.emit(OR)))
    _st(a, y_cell, lambda a: _binop_imm(a, y_cell, SHR, 1))            # y >>= 1
    a.jmp("m16top")
    a.label("m16done")
    _ld(a, out_lo)
    return a


def _lt_expr_free(a: Asm, addr_x: int, addr_y: int, out: int) -> None:
    """``_lt_expr`` variant using DEDICATED cells that do NOT overlap the mul8x8's
    working set (LT_D/LT_NX are x_lo/x_hi there).  Uses B_MSK/B_VAL/B_MCH as private
    scratch."""
    _st(a, B_MSK, lambda a: _binop(a, addr_x, isa.SUB, addr_y))       # d = x - y
    _st(a, B_VAL, lambda a: _binop_imm(a, addr_x, XOR, 0xFF))         # nx = ~x
    _st(a, B_MCH, lambda a: _binop(a, B_VAL, OR, addr_y))             # nx | y

    def _val(a: Asm) -> None:
        _binop(a, B_VAL, AND, addr_y)
        a.psh()
        _binop(a, B_MSK, AND, B_MCH)
        a.emit(OR)
        a.psh()
        a.imm(7)
        a.emit(SHR)
    _st(a, out, _val)


def emit_divmod8(x_cell: int = C_X, y_cell: int = C_Y,
                 q_cell: int = C_Q, r_cell: int = C_R,
                 want_rem: bool = False) -> Asm:
    """Restoring 8-bit long division of *x_cell by *y_cell (8 bit-iterations,
    MSB first).  Leaves quotient in *q_cell, remainder in *r_cell.  AX = the
    requested one.  divisor 0 -> 0 (ISA_SPEC 4.2).

    The subtract condition is ``carry_out | (remainder >= divisor)`` where
    carry_out is remainder's dropped bit-8 — so the 8-bit remainder never
    overflows even though ``2*rem+bit`` can reach 9 bits."""
    a = Asm()
    _ld(a, y_cell)
    a.bnz("go")
    _stc(a, q_cell, 0)
    _stc(a, r_cell, 0)
    a.jmp("dend")
    a.label("go")
    _stc(a, q_cell, 0)
    _stc(a, r_cell, 0)                      # remainder = 0
    _stc(a, C_I, 8)
    a.label("dtop")
    _ld(a, C_I)
    a.bz("ddone")
    _st(a, C_I, lambda a: _binop_imm(a, C_I, isa.SUB, 1))     # i = i - 1 (bit index)
    _st(a, C_CY, lambda a: _binop_imm(a, r_cell, SHR, 7))     # carry_out = rem bit7
    _st(a, C_BIT, lambda a: (_ld(a, x_cell), a.psh(), _ld(a, C_I), a.emit(SHR),
                             a.psh(), a.imm(1), a.emit(AND)))  # bit = (x>>i)&1
    _st(a, r_cell, lambda a: (_binop_imm(a, r_cell, SHL, 1), a.psh(),
                              _ld(a, C_BIT), a.emit(OR)))       # rem = (rem<<1)|bit
    _lt_expr(a, r_cell, y_cell, C_LT)                          # rem < div
    _st(a, C_SUB, lambda a: _binop_imm(a, C_LT, XOR, 1))       # ge = !lt
    _st(a, C_SUB, lambda a: _binop(a, C_SUB, OR, C_CY))        # sub = ge | carry
    _ld(a, C_SUB)
    a.bz("nosub")
    _st(a, r_cell, lambda a: _binop(a, r_cell, isa.SUB, y_cell))    # rem -= div
    _st(a, C_BIT, lambda a: (a.imm(1), a.psh(), _ld(a, C_I), a.emit(SHL)))  # 1<<i
    _st(a, q_cell, lambda a: _binop(a, q_cell, OR, C_BIT))          # q |= 1<<i
    a.label("nosub")
    a.jmp("dtop")
    a.label("ddone")
    a.label("dend")
    _ld(a, r_cell if want_rem else q_cell)
    return a


def emit_mod8(x_cell: int = C_X, y_cell: int = C_Y,
              q_cell: int = C_Q, r_cell: int = C_R) -> Asm:
    """8-bit MOD = restoring division, return remainder."""
    return emit_divmod8(x_cell, y_cell, q_cell, r_cell, want_rem=True)


# ===========================================================================
# MULTI-BYTE (16 / 32-bit) — bytes in memory (LSB first), byte-by-byte carry.
# ===========================================================================
def _mb_zero(a: Asm, cells: List[int], nbytes: int) -> None:
    for i in range(nbytes):
        _stc(a, cells[i], 0)


def _mb_set(a: Asm, cells: List[int], value: int, nbytes: int) -> None:
    for i in range(nbytes):
        _stc(a, cells[i], (value >> (8 * i)) & 0xFF)


def _mb_add(a: Asm, dst: List[int], src: List[int], nbytes: int) -> None:
    """dst += src  (ripple carry through C_CY).  Per byte: t = dst+src+cin; the
    carry_out = 1 iff the true sum >= 256, detected as ``(t < src) | (t==src &
    cin)`` — exact for cin in {0,1}."""
    _stc(a, C_CY, 0)
    for i in range(nbytes):
        _st(a, C_T0, lambda a, i=i: _binop(a, dst[i], isa.ADD, src[i]))   # dst+src
        _lt_expr(a, C_T0, src[i], C_T1)                                   # carry from dst+src
        _st(a, C_T0, lambda a: _binop(a, C_T0, isa.ADD, C_CY))            # + cin
        # extra carry if adding cin wrapped: (t2 < cin)  (cin in {0,1})
        _lt_expr(a, C_T0, C_CY, C_T2)
        _st(a, dst[i], lambda a: _ld(a, C_T0))
        _st(a, C_CY, lambda a: _binop(a, C_T1, OR, C_T2))                 # carry_out


def _mb_sub(a: Asm, dst: List[int], src: List[int], nbytes: int) -> None:
    """dst -= src  (ripple borrow through C_CY)."""
    _stc(a, C_CY, 0)
    for i in range(nbytes):
        _lt_expr(a, dst[i], src[i], C_T1)                                 # borrow1
        _st(a, C_T0, lambda a, i=i: _binop(a, dst[i], isa.SUB, src[i]))   # dst-src
        _lt_expr(a, C_T0, C_CY, C_T2)                                     # borrow from - cin
        _st(a, C_T0, lambda a: _binop(a, C_T0, isa.SUB, C_CY))
        _st(a, dst[i], lambda a: _ld(a, C_T0))
        _st(a, C_CY, lambda a: _binop(a, C_T1, OR, C_T2))                 # borrow_out


def _mb_shl1(a: Asm, cells: List[int], nbytes: int) -> None:
    """cells <<= 1."""
    _stc(a, C_CY, 0)
    for i in range(nbytes):
        _st(a, C_T1, lambda a, i=i: _binop_imm(a, cells[i], SHR, 7))      # carry_out
        _st(a, cells[i], lambda a, i=i: (_binop_imm(a, cells[i], SHL, 1),
                                         a.psh(), _ld(a, C_CY), a.emit(OR)))
        _st(a, C_CY, lambda a: _ld(a, C_T1))


def _mb_or_any(a: Asm, cells: List[int], nbytes: int, out: int) -> None:
    """*out = OR of all bytes (nonzero iff the value is nonzero)."""
    _st(a, out, lambda a: _ld(a, cells[0]))
    for i in range(1, nbytes):
        _st(a, out, lambda a, i=i: _binop(a, out, OR, cells[i]))


def _mb_lt(a: Asm, x_cells: List[int], y_cells: List[int], nbytes: int, out: int) -> None:
    """*out = 1 if x < y else 0  (multi-byte unsigned, LSB->MSB borrow scan).
    The final borrow-out of ``x - y`` equals ``x < y``."""
    _stc(a, C_CY, 0)                        # borrow-in
    for i in range(nbytes):
        _lt_expr(a, x_cells[i], y_cells[i], C_T0)      # x_i < y_i
        _st(a, C_T1, lambda a, i=i: _binop(a, x_cells[i], XOR, y_cells[i]))   # d
        _eq0_expr(a, C_T1, C_T2)             # eq_i = (x_i == y_i)
        _st(a, C_T2, lambda a: _binop(a, C_T2, AND, C_CY))     # eq_i & borrow_in
        _st(a, C_CY, lambda a: _binop(a, C_T0, OR, C_T2))      # bl | (eq & bin)
    _st(a, out, lambda a: _ld(a, C_CY))


def _mb_extract_bit(a: Asm, cells: List[int], nbytes: int, i_cell: int, out: int) -> None:
    """*out = (multibyte >> *i_cell) & 1  (byte = i>>3, off = i&7; branch-free
    equality-mask select over the ``nbytes`` bytes).  ``out`` may alias anything
    except the B_* private cells."""
    _st(a, B_IDX, lambda a: _binop_imm(a, i_cell, SHR, 3))   # byte index
    _st(a, B_OFF, lambda a: _binop_imm(a, i_cell, AND, 7))   # bit offset
    _stc(a, out, 0)
    for bi in range(nbytes):
        _st(a, B_MCH, lambda a, bi=bi: _binop_imm(a, B_IDX, XOR, bi))     # 0 iff match
        _eq0_expr(a, B_MCH, B_MCH)           # match flag (byte_index == bi) -> 0/1
        _st(a, B_VAL, lambda a, bi=bi: (_ld(a, cells[bi]), a.psh(), _ld(a, B_OFF),
                                        a.emit(SHR), a.psh(), a.imm(1), a.emit(AND)))
        _st(a, B_VAL, lambda a: _binop(a, B_VAL, AND, B_MCH))    # bit & match (both 0/1)
        _st(a, out, lambda a: _binop(a, out, OR, B_VAL))


def _mb_set_bit(a: Asm, cells: List[int], nbytes: int, i_cell: int) -> None:
    """Set bit *i_cell of the multi-byte value in ``cells``."""
    _st(a, B_IDX, lambda a: _binop_imm(a, i_cell, SHR, 3))   # byte index
    _st(a, B_OFF, lambda a: _binop_imm(a, i_cell, AND, 7))   # bit offset
    _st(a, B_VAL, lambda a: (a.imm(1), a.psh(), _ld(a, B_OFF), a.emit(SHL)))   # 1<<off
    for bi in range(nbytes):
        _st(a, B_MCH, lambda a, bi=bi: _binop_imm(a, B_IDX, XOR, bi))
        _eq0_expr(a, B_MCH, B_MCH)           # match -> 0/1
        # full-byte mask = 0 - match  (0x00 if no match, 0xFF if match).
        _st(a, B_MSK, lambda a: (a.imm(0), a.psh(), _ld(a, B_MCH), a.emit(isa.SUB)))
        _st(a, B_MSK, lambda a: _binop(a, B_VAL, AND, B_MSK))     # (1<<off) if match else 0
        _st(a, cells[bi], lambda a, bi=bi: _binop(a, cells[bi], OR, B_MSK))


def emit_mul_mb(nbytes: int, a_cells: List[int] = None, b_cells: List[int] = None,
                r_cells: List[int] = None) -> Asm:
    """Multi-byte shift-and-add multiply, fixed ``nbytes`` width (result masked
    to ``nbytes``).  ``r=0; while b: if b&1: r+=a; a<<=1; b>>=1``."""
    a_cells = a_cells or A_BYTES[:nbytes]
    b_cells = b_cells or B_BYTES[:nbytes]
    r_cells = r_cells or M_BYTES[:nbytes]
    a = Asm()
    _mb_zero(a, r_cells, nbytes)
    a.label("mtop")
    _mb_or_any(a, b_cells, nbytes, C_LT)
    _ld(a, C_LT)
    a.bz("mdone")
    _st(a, C_BIT, lambda a: _binop_imm(a, b_cells[0], AND, 1))    # b & 1
    _ld(a, C_BIT)
    a.bz("mskip")
    _mb_add(a, r_cells, a_cells, nbytes)      # r += a
    a.label("mskip")
    _mb_shl1(a, a_cells, nbytes)              # a <<= 1
    _mb_shr1(a, b_cells, nbytes)              # b >>= 1
    a.jmp("mtop")
    a.label("mdone")
    _ld(a, r_cells[0])
    return a


def _mb_shr1(a: Asm, cells: List[int], nbytes: int) -> None:
    """cells >>= 1  (carry from high byte down)."""
    _stc(a, C_CY, 0)
    for i in range(nbytes - 1, -1, -1):
        _st(a, C_T1, lambda a, i=i: _binop_imm(a, cells[i], AND, 1))     # bit passed down
        _st(a, cells[i], lambda a, i=i: (_binop_imm(a, cells[i], SHR, 1), a.psh(),
                                         _ld(a, C_CY), a.psh(), a.imm(7), a.emit(SHL),
                                         a.emit(OR)))
        _st(a, C_CY, lambda a: _ld(a, C_T1))


def emit_divmod_mb(nbytes: int, want_rem: bool = False, a_cells: List[int] = None,
                   b_cells: List[int] = None, q_cells: List[int] = None,
                   m_cells: List[int] = None) -> Asm:
    """Multi-byte restoring long division of A by B (``nbytes`` each).  Leaves
    quotient in ``q_cells``, remainder in ``m_cells``.  ``8*nbytes``
    bit-iterations, MSB first.  B == 0 -> 0."""
    a_cells = a_cells or A_BYTES[:nbytes]
    b_cells = b_cells or B_BYTES[:nbytes]
    q_cells = q_cells or Q_BYTES[:nbytes]
    m_cells = m_cells or M_BYTES[:nbytes]
    nbits = 8 * nbytes
    a = Asm()
    _mb_or_any(a, b_cells, nbytes, C_LT)
    _ld(a, C_LT)
    a.bnz("dgo")
    _mb_zero(a, q_cells, nbytes)
    _mb_zero(a, m_cells, nbytes)
    a.jmp("dend")
    a.label("dgo")
    _mb_zero(a, q_cells, nbytes)
    _mb_zero(a, m_cells, nbytes)              # remainder = 0
    _stc(a, C_I, nbits)
    a.label("dtop")
    _ld(a, C_I)
    a.bz("ddone")
    _st(a, C_I, lambda a: _binop_imm(a, C_I, isa.SUB, 1))     # i -= 1 (bit index)
    _mb_extract_bit(a, a_cells, nbytes, C_I, C_BIT)           # bit = (A>>i)&1
    _mb_shl1(a, m_cells, nbytes)                              # rem <<= 1
    _st(a, m_cells[0], lambda a: _binop(a, m_cells[0], OR, C_BIT))   # rem |= bit
    _mb_lt(a, m_cells, b_cells, nbytes, C_LT)                # rem < div
    _ld(a, C_LT)
    a.bnz("nosub")
    _mb_sub(a, m_cells, b_cells, nbytes)                     # rem -= div
    _mb_set_bit(a, q_cells, nbytes, C_I)                     # q bit i = 1
    a.label("nosub")
    a.jmp("dtop")
    a.label("ddone")
    a.label("dend")
    _ld(a, (m_cells if want_rem else q_cells)[0])
    return a


# ===========================================================================
# Driver programs (operands baked as leading IMM;SI stores).
# ===========================================================================
def program_mul8(a_val: int, b_val: int) -> List[isa.Instr]:
    a = Asm()
    _stc(a, C_X, a_val)
    _stc(a, C_Y, b_val)
    a.splice(emit_mul8())
    a.exit_()
    return a.instrs()


def program_div8(a_val: int, b_val: int) -> List[isa.Instr]:
    a = Asm()
    _stc(a, C_X, a_val)
    _stc(a, C_Y, b_val)
    a.splice(emit_divmod8())
    a.exit_()
    return a.instrs()


def program_mod8(a_val: int, b_val: int) -> List[isa.Instr]:
    a = Asm()
    _stc(a, C_X, a_val)
    _stc(a, C_Y, b_val)
    a.splice(emit_mod8())
    a.exit_()
    return a.instrs()


def program_mul_mb(a_val: int, b_val: int, nbytes: int) -> List[isa.Instr]:
    a = Asm()
    _mb_set(a, A_BYTES[:nbytes], a_val, nbytes)
    _mb_set(a, B_BYTES[:nbytes], b_val, nbytes)
    a.splice(emit_mul_mb(nbytes))
    a.exit_()
    return a.instrs()


def program_div_mb(a_val: int, b_val: int, nbytes: int) -> List[isa.Instr]:
    a = Asm()
    _mb_set(a, A_BYTES[:nbytes], a_val, nbytes)
    _mb_set(a, B_BYTES[:nbytes], b_val, nbytes)
    a.splice(emit_divmod_mb(nbytes, want_rem=False))
    a.exit_()
    return a.instrs()


def program_mod_mb(a_val: int, b_val: int, nbytes: int) -> List[isa.Instr]:
    a = Asm()
    _mb_set(a, A_BYTES[:nbytes], a_val, nbytes)
    _mb_set(a, B_BYTES[:nbytes], b_val, nbytes)
    a.splice(emit_divmod_mb(nbytes, want_rem=True))
    a.exit_()
    return a.instrs()
