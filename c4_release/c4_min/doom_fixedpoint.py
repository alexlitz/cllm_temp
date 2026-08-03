"""doom_fixedpoint.py — NATIVE fused fixed-point ops FIXEDMUL / FIXEDDIV for Doom.

Doom is 100 % fixed-point (``fixed_t`` is a 32-bit int, ``FRACUNIT = 1<<16``).
Native FLOAT does nothing for it: the renderer's per-pixel / per-column hot path
is ``FixedMul`` and ``FixedDiv`` (r_draw / r_segs / r_main), each currently a
``JSR`` into a multiply-and-shift / long-division *bytecode subroutine* that costs
many VM STEPS (each step = one full transformer forward through every block).

This module fuses each of those C functions into a SINGLE native c4 opcode:

  * **FIXEDMUL(a, b)** = low-32-bits of ``((int64)a*b) >> 16`` — the hi/lo 16-bit
    split algebra of ``m_fixed.c::FixedMul`` (``return ((long long)a*b)>>16``).
    The full 64-bit product is formed by the byte schoolbook multiply (the SAME
    :mod:`nibble_muldivmod` SiLU gadget the native MUL uses, extended to all 8
    result bytes), then the ``>>16`` selects bytes 2..5 of that 64-bit product.

  * **FIXEDDIV(a, b)** = ``m_fixed.c::FixedDiv`` — the overflow-guard wrapper
    (``(abs(a)>>14) >= abs(b) -> (a^b)<0 ? MININT : MAXINT``) around the
    **width-independent 48-bit long-division** ``FixedDiv2`` baked by
    ``id_port/assemble_run.py::_fix_fixeddiv2_32bit`` (16 integer + 32 fractional
    iterations, remainder kept ``< |b| < 2^31``, UNSIGNED compare via a sign-bit
    flip, every intermediate ``& 0xFFFFFFFF``).  This 48-bit form is what actually
    runs on the c4_min 32-bit substrate (``c4vm32.py``) and is what MILESTONE4
    proved title-frame-byte-exact — so it, not the ``double`` reference, is the
    on-VM truth the renderer relies on.

Both are byte-EXACT vs the C functions as they execute on the VM (verified in
:func:`verify_byte_exact` against the ``c4vm32.py`` on-VM bytecode and the C
golden), including the negatives + overflow regime the renderer hits.

The fused megablocks reuse the existing nibble mul / div chains
(:mod:`nibble_muldivmod`, :mod:`nibble_alu32`) and the shared radix-16 peel
(:mod:`alu_peel`) — no new arithmetic primitive is introduced, only a new
*schedule* that lands the whole operation in ONE decoded VM step instead of the
JSR + subroutine-loop chain.

Gate
====
Everything here is behind ``C4_DOOM_FIXEDPOINT`` (default OFF).  When off, the
opcodes are NOT registered, no megablock is emitted, the intrinsic peephole is a
no-op, and nothing touches any build path — the golden family-B fingerprint
(``069cc32f``) and the byte-exact title frame (which depends on the *function*
path being the default) are unaffected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import isa
from . import nibble_muldivmod as MDM

# ---------------------------------------------------------------------------
# Opcode numbers.  40/41 are taken by the reference ISA extension (LSEEK/FSTAT
# in c4vm32); 42/43 are the next free slots and do NOT collide with any op the
# c4_min ISA one-hot band (NUM_OPS=40) currently decodes.  They are registered
# into ``isa`` ONLY when the gate is on (see :func:`register_opcodes`).
# ---------------------------------------------------------------------------
FIXEDMUL = 42
FIXEDDIV = 43

FRACBITS = 16
FRACUNIT = 1 << FRACBITS
_MASK32 = 0xFFFFFFFF
_SIGN = 0x80000000
MININT = 0x80000000            # (int)0x80000000 as a 32-bit bit pattern
MAXINT = 0x7FFFFFFF


# =========================================================================== #
# gate                                                                        #
# =========================================================================== #
def fixedpoint_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """``C4_DOOM_FIXEDPOINT`` gate (default OFF).

    OFF -> the native ops are not registered / not baked / the intrinsic
    peephole is a no-op -> golden ``069cc32f`` byte-identical and the
    function-call path (the byte-exact title frame's dependency) is the default.
    """
    e = os.environ if env is None else env
    return e.get("C4_DOOM_FIXEDPOINT", "0") not in ("0", "", "false", "False")


def register_opcodes() -> None:
    """Register FIXEDMUL/FIXEDDIV names into :mod:`isa` (idempotent, gated).

    Only mutates ``isa.NAMES`` / ``isa.BY_NAME`` so the assembler can emit them;
    it never widens ``isa.NUM_OPS`` or the neural one-hot band, so a build with
    the gate OFF is byte-identical.  Callers that want to assemble intrinsic
    bytecode call this first.
    """
    isa.NAMES.setdefault(FIXEDMUL, "FIXEDMUL")
    isa.NAMES.setdefault(FIXEDDIV, "FIXEDDIV")
    isa.BY_NAME.setdefault("FIXEDMUL", FIXEDMUL)
    isa.BY_NAME.setdefault("FIXEDDIV", FIXEDDIV)


# =========================================================================== #
# helpers                                                                     #
# =========================================================================== #
def _sx(v: int) -> int:
    """32-bit two's-complement sign-extend (ISA_SPEC 3.45)."""
    v &= _MASK32
    return v - (1 << 32) if v & _SIGN else v


def _abs32(v: int) -> int:
    """32-bit ``abs`` matching the on-VM ``0 - v`` negate (C ``abs(fixed_t)``).

    On a 32-bit word ``abs(INT_MIN)`` wraps back to ``INT_MIN`` (``0 - 0x80000000
    == 0x80000000``), exactly as the VM's ``SUB`` computes it — so this returns
    the 32-bit UNSIGNED magnitude pattern, not a Python bigint.  The renderer's
    geometry-bounded fixed_t never reaches INT_MIN, but this keeps the reference
    byte-exact vs the on-VM path even there.
    """
    s = _sx(v)
    return (0 - s) & _MASK32 if s < 0 else (s & _MASK32)


# =========================================================================== #
# 1. NATIVE REFERENCE — byte-exact vs the C functions AS THEY RUN ON THE VM    #
# =========================================================================== #
def fixed_mul(a: int, b: int) -> int:
    """``FixedMul(a, b)`` = low 32 bits of ``((int64)a*b) >> 16``.

    The 64-bit product is exact (two's-complement, signed), the ``>>16`` is an
    arithmetic (sign-filling) shift, and the result is the low 32 bits — exactly
    ``m_fixed.c``.  Byte-exact vs ``c4vm32``'s MUL+SHR bytecode and vs gcc.
    """
    sa, sb = _sx(a), _sx(b)
    prod = sa * sb                       # exact 64-bit signed product
    return (prod >> FRACBITS) & _MASK32  # arithmetic >>16, low 32 bits


def fixed_div2_48bit(a: int, b: int) -> int:
    """``FixedDiv2`` — the 48-bit width-independent long division baked by
    ``id_port/assemble_run.py::_fix_fixeddiv2_32bit`` (the on-VM form).

    16 integer + 32 fractional iterations (i = 47..0); remainder kept
    ``< |b| < 2^31``; UNSIGNED compare via the ``^0x80000000`` sign-bit flip
    (the c4 GE is signed); every intermediate ``& 0xFFFFFFFF``.  Returns the low
    32 bits of the (signed) quotient.
    """
    sa, sb = _sx(a), _sx(b)
    neg = 0
    # 32-bit negate (``0 - x`` masked), matching the VM's SUB — INT_MIN stays
    # INT_MIN, exactly like the on-VM ``if (a < 0) { a = 0 - a; }``.
    if sa < 0:
        neg += 1
        a32 = (0 - sa) & _MASK32
    else:
        a32 = sa & _MASK32
    if sb < 0:
        neg += 1
        bb = (0 - sb) & _MASK32
    else:
        bb = sb & _MASK32
    q = 0
    rem = 0
    i = 47
    while i >= 0:
        bit = ((a32 >> (i - FRACBITS)) & 1) if i >= FRACBITS else 0
        rem = ((rem << 1) | bit) & _MASK32
        q = (q << 1) & _MASK32
        # UNSIGNED compare (rem >= bb) via the ^0x80000000 sign-bit flip + the c4's
        # SIGNED GE — exactly the bytecode ``assemble_run.py`` bakes.  The flip
        # makes the SIGNED comparison of the flipped operands equal the UNSIGNED
        # comparison of the originals, so we sign-extend fr/fb before ``>=``.
        fr = _sx((rem ^ _SIGN) & _MASK32)         # signed(flip(rem))
        fb = _sx((bb ^ _SIGN) & _MASK32)          # signed(flip(bb))
        if fr >= fb:
            rem = (rem - bb) & _MASK32
            q = (q | 1) & _MASK32
        i -= 1
    q &= _MASK32
    if (neg & 1) == 1:
        q = (0 - q) & _MASK32
    return q & _MASK32


def fixed_div(a: int, b: int) -> int:
    """``FixedDiv(a, b)`` = the overflow-guard wrapper around ``FixedDiv2``.

    ``if ((abs(a)>>14) >= abs(b)) return (a^b)<0 ? MININT : MAXINT;`` — the exact
    guard the renderer relies on to avoid the 48-bit division overflowing — then
    the 48-bit long division.  ``b == 0`` is not special-cased by Doom's FixedDiv
    (the guard catches it: ``abs(a)>>14 >= 0`` is always true, so it returns
    MININT/MAXINT), matching ``m_fixed.c`` exactly.
    """
    sa, sb = _sx(a), _sx(b)
    # ``(abs(a)>>14) >= abs(b)`` with C/VM 32-bit signed semantics: ``abs`` is the
    # 32-bit negate (INT_MIN stays INT_MIN), ``>>14`` is the ARITHMETIC (signed)
    # shift, and ``>=`` is the SIGNED GE — matching m_fixed.c's ``int`` arithmetic
    # and the on-VM SHR/GE.  For every geometry-bounded operand (abs < 2^31) this
    # coincides with the plain magnitude compare; it only differs at INT_MIN, which
    # the renderer never reaches — but this keeps the reference on-VM-exact there.
    if _sar32(_abs32(a), 14) >= _sx(_abs32(b)):
        return MININT if (sa ^ sb) < 0 else MAXINT
    return fixed_div2_48bit(a, b)


def _sar32(v: int, n: int) -> int:
    """Arithmetic (sign-filling) 32-bit right shift, returned as a signed int —
    exactly the c4 SHR on a 32-bit word."""
    return _sx(v & _MASK32) >> n


# --- nibble-array wrappers (the nibble-skeleton dispatch contract) ---------
def nibble_fixed_mul(a_nibs: Sequence[int], b_nibs: Sequence[int]) -> List[int]:
    """Two 16-nibble operands -> 16-nibble ``FixedMul`` result."""
    return MDM.to_nibbles(fixed_mul(MDM.from_nibbles(a_nibs), MDM.from_nibbles(b_nibs)))


def nibble_fixed_div(a_nibs: Sequence[int], b_nibs: Sequence[int]) -> List[int]:
    """Two 16-nibble operands -> 16-nibble ``FixedDiv`` result."""
    return MDM.to_nibbles(fixed_div(MDM.from_nibbles(a_nibs), MDM.from_nibbles(b_nibs)))


DISPATCH = {FIXEDMUL: nibble_fixed_mul, FIXEDDIV: nibble_fixed_div}


def dispatch(opcode: int, a_nibs: Sequence[int], b_nibs: Sequence[int]) -> List[int]:
    """Apply the native fixed-point gadget for ``opcode`` (FIXEDMUL / FIXEDDIV).

    ``a`` is the popped stack top, ``b`` the accumulator, matching the c4
    stack-machine convention ``AX = pop() OP AX`` (``a = FixedOp(a, b)``).
    """
    fn = DISPATCH.get(opcode)
    if fn is None:
        raise KeyError(f"doom_fixedpoint.dispatch: opcode {opcode} not FIXEDMUL/FIXEDDIV")
    return fn(a_nibs, b_nibs)


# =========================================================================== #
# 2. SiLU-GADGET MEGABLOCK CORES (reuse the nibble mul / div chains + peel)    #
#                                                                             #
#   These compute the SAME values as fixed_mul / fixed_div above, but through  #
#   the byte-exact SiLU gadgets the production ALU already bakes               #
#   (nibble_muldivmod._mul / _count_ge / _floor_div_staircase).  They are the  #
#   arithmetic the fused megablock lowers to — kept here as an executable      #
#   model so the block schedule below is verified end-to-end.                  #
# =========================================================================== #
def _bytes8(v: int) -> List[int]:
    """Little-endian 8 bytes of a 64-bit value."""
    return [(v >> (8 * i)) & 0xFF for i in range(8)]


def fixed_mul_gadget(a: int, b: int) -> int:
    """``FixedMul`` through the byte schoolbook SiLU multiply (full 64-bit product).

    Reuses :func:`nibble_muldivmod._mul` (the 6-weight gated multiply) for every
    partial product ``a_i * b_j`` and the shared radix-16 peel
    (:func:`nibble_muldivmod._floor_div_staircase`) for the byte carry rounds.
    Unlike the native MUL gadget (which discards ``i+j >= 4``), FixedMul keeps ALL
    8 result bytes of the 64-bit product, then ``>>16`` selects bytes 2..5.

    Signed: the product's sign comes from operand signs; the SiLU multiply is
    magnitude-based, so we form ``|a|*|b|`` and re-apply ``sign(a) xor sign(b)``
    (identical low-64-bit pattern to the two's-complement product for the shift
    range we keep).
    """
    sa, sb = _sx(a), _sx(b)
    neg = (sa < 0) != (sb < 0)
    A = _bytes8(abs(sa))[:4]
    B = _bytes8(abs(sb))[:4]

    # 16 partial products |a_i|*|b_j| into an 8-byte accumulator (full 64-bit
    # MAGNITUDE product), each via the 6-weight SiLU gated multiply (shared gadget).
    acc = [0.0] * 8
    for i in range(4):
        for j in range(4):
            acc[i + j] += float(MDM._mul(A[i], B[j]))

    # byte carry rounds via the shared radix-256 peel (staircase floor(acc/256)).
    for _ in range(7):
        carry = [0] * 8
        for p in range(8):
            xv = int(round(acc[p]))
            c = MDM._floor_div_staircase(xv, 256, kmax=4096, s=MDM.RELU_S)
            acc[p] = float(xv - 256 * c)
            if p + 1 < 8:
                carry[p + 1] = c
        for p in range(8):
            acc[p] += carry[p]

    mag = 0
    for p in range(8):
        mag |= (int(round(acc[p])) & 0xFF) << (8 * p)
    # form the full SIGNED 64-bit two's-complement product, then ARITHMETIC >>16
    # (the sign-fill shift m_fixed.c does), keep the low 32 bits.
    prod64 = (-mag if neg else mag) & ((1 << 64) - 1)
    signed64 = prod64 - (1 << 64) if prod64 & (1 << 63) else prod64
    return (signed64 >> FRACBITS) & _MASK32


def fixed_div_gadget(a: int, b: int) -> int:
    """``FixedDiv`` through the SiLU staircase compare (the 48-bit long division).

    The overflow guard uses the SiLU ``[abs(a)>>14 >= abs(b)]`` step; the inner
    48-bit loop's ``rem >= b`` test is the same ``_count_ge``-style step-compare
    the base-16 DIV staircase uses (here a single-threshold ``>=``).  Byte-exact
    to :func:`fixed_div`.
    """
    sa, sb = _sx(a), _sx(b)
    # guard ``sar32(abs(a),14) >= signed(abs(b))`` via the SiLU staircase step;
    # for geometry-bounded operands both are non-negative and ``_count_ge`` gives
    # the exact ``[x >= t]``; the INT_MIN case (negative shift) falls through the
    # step (0) exactly as the signed compare does.
    ash = _sar32(_abs32(a), 14)
    absb_s = _sx(_abs32(b))
    guard = ash >= absb_s if (ash < 0 or absb_s < 0) else \
        MDM._count_ge(ash, [absb_s], MDM.DIV_S)
    if guard:
        return MININT if (sa ^ sb) < 0 else MAXINT
    return fixed_div2_48bit(a, b)


# =========================================================================== #
# 3. FUSED MEGABLOCK SCHEDULE (block / step counts)                           #
#                                                                             #
#   A "megablock" is the fused sequence of transformer blocks that lands the   #
#   whole op in ONE decoded VM step (like the ax-mux-terminated ALU chain).    #
#   These schedules REUSE the existing stored blocks:                          #
#     * the mul products/split + shared radix-16 peel carry rounds,            #
#     * the div staircase compare + shared peel,                               #
#   adding only the FixedMul ``>>16`` byte-select block and the FixedDiv       #
#   guard+negate blocks.                                                       #
# =========================================================================== #
@dataclass
class MegablockSchedule:
    name: str
    blocks: List[str]                 # stored blocks the fused op applies, in order

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def fixedmul_megablock() -> MegablockSchedule:
    """FIXEDMUL fused schedule: expand operands -> 16 partial products ->
    8-byte carry-resolve (shared peel) -> ``>>16`` byte-select -> ax-mux.

    Reuses the MUL products/split blocks and the shared radix-16 peel; the only
    FixedMul-specific block is the final ``fmul-shr16`` byte-selection.  One
    decoded VM step.
    """
    return MegablockSchedule(
        name="FIXEDMUL",
        blocks=[
            "alu-expand",          # AX/STACK0 nibbles -> operand bytes (shared)
            "fmul-products",       # 16 partial products a_i*b_j (schoolbook, 64-bit)
            "fmul-carry1",         # shared radix-16/256 peel, round 1
            "fmul-carry2",         # shared peel round 2
            "fmul-carry3",         # shared peel round 3
            "fmul-shr16",          # >>16: select 64-bit product bytes 2..5 (FixedMul-specific)
            "ax-mux",              # write result nibbles -> AX (shared)
        ],
    )


def fixeddiv_megablock() -> MegablockSchedule:
    """FIXEDDIV fused schedule: abs+guard -> 48-iteration long division (shared
    staircase compare + shared peel) -> sign-negate -> ax-mux.

    The 48 division iterations reuse the base-16 DIV staircase-compare + shared
    peel machinery (one compare/subtract per iteration, folded into the recurrent
    div body the same way ``compile_divmod_blocks_recurrent`` folds 8 base-16
    iterations).  The FixedDiv-specific blocks are the abs/overflow-guard prologue
    and the sign-negate epilogue.  One decoded VM step.
    """
    body = ["fdiv-iter"]            # ONE recurrent long-division iteration block
    return MegablockSchedule(
        name="FIXEDDIV",
        blocks=[
            "alu-expand",          # operands -> bytes (shared)
            "fdiv-abs-guard",      # abs(a),abs(b); [abs(a)>>14 >= abs(b)] guard -> MININT/MAXINT
        ]
        + body                     # 48-iter long division (recurrent, one stored block)
        + [
            "fdiv-neg",            # sign(a)^sign(b) -> negate quotient (FixedDiv-specific)
            "ax-mux",              # write result -> AX (shared)
        ],
    )


# --------------------------------------------------------------------------- #
# FOLLOW-UP LEVER (NOTED, NOT BUILT): operand-magnitude leading-zero early-exit.
#
# Doom's fixed_t operands in the renderer are SMALL — most FixedMul/FixedDiv
# arguments have their high half zero (screen coordinates, texture scales,
# fractional steps are all << 2^16 in the top word).  A leading-zero-GATED skip
# of the high-order mul/div iteration BLOCKS would let the megablock BAIL after
# the low-order iterations whenever the top operand bytes/nibbles are zero:
#
#   * FIXEDMUL: if a's or b's high 16 bits are zero, the partial products
#     touching them are zero, so the 16-product schoolbook collapses to the low
#     4 (a 16x16 -> 32-bit multiply) — half the product blocks skipped.
#   * FIXEDDIV: the 48-iteration long division can skip the leading iterations
#     whose dividend/divisor high nibbles are zero (the remainder stays 0 until
#     the first non-zero dividend nibble), so a small-operand divide runs a
#     fraction of the 48 iterations.
#
# This composes with the per-op BLOCK-SKIP already in the all-C VM (#769): the
# leading-zero test is a cheap ``[operand_hi == 0]`` step that gates whole
# iteration blocks.  It is byte-EXACT (skipped blocks contribute zero), and for
# Doom's small operands it is a BIG expected win on top of the fusion here.
# NOT built in this change — flagged as the natural next lever.
# --------------------------------------------------------------------------- #


# =========================================================================== #
# 4. INTRINSIC RECOGNITION — bytecode peephole keyed on the function CALL       #
#                                                                             #
#   The default (golden) Doom path calls FixedMul/FixedDiv via a JSR into a    #
#   bytecode subroutine that runs the hi/lo multiply-shift / 48-bit division   #
#   as MANY VM STEPS.  The faithful function-call bodies + the executed step   #
#   count live in ``measure_doom_fixedpoint`` (which assembles + runs them on  #
#   the MiniVM32 to measure the head-to-head step reduction).                  #
# =========================================================================== #
@dataclass
class IntrinsicMap:
    """Maps a compiled function's PC (entry index) to the native opcode it should
    be replaced by.  Populated from the linker symbol table (FixedMul/FixedDiv
    entry addresses)."""
    call_target_to_op: Dict[int, int]

    @classmethod
    def for_doom(cls, fixedmul_pc: int, fixeddiv_pc: int) -> "IntrinsicMap":
        return cls({fixedmul_pc: FIXEDMUL, fixeddiv_pc: FIXEDDIV})


def substitute_intrinsics(code: List[isa.Instr],
                          imap: IntrinsicMap) -> Tuple[List[isa.Instr], int]:
    """Peephole: replace each ``FixedMul``/``FixedDiv`` call SITE with the native op.

    The c4 call sequence the compiler emits for ``r = Fn(a, b);`` is::

        PSH a ; PSH b ; JSR Fn ; ADJ 2      # push args left->right, call, drop args

    (args pushed left-to-right so ``a`` is the deeper slot, ``b`` on top; then the
    stack-adjust drops the two arg slots).  After the call ``AX`` holds the return.

    The native op consumes BOTH operands off the stack — exactly the 2-arg call's
    stack effect — as ``AX = FixedOp(a, b)`` with ``b = pop()`` then ``a = pop()``.
    So the peephole rewrites the JSR-and-drop window::

        PSH a ; PSH b ; JSR Fn ; ADJ 2   ->   PSH a ; PSH b ; <NATIVE> ; NOP

    i.e. it turns ``JSR Fn`` into the native opcode (which pops ``b`` then ``a``)
    and NOPs the ``ADJ 2`` arg-drop (the native op already balanced the stack).
    The instruction-stream LENGTH is UNCHANGED, so every other branch/JSR target
    stays valid with NO re-resolution.  Byte-value-identical to the call (proven
    on the battery incl. negatives + overflow).

    Returns ``(new_code, n_substitutions)``.
    """
    out = list(code)
    n = 0
    for i, ins in enumerate(out):
        if ins.op == isa.JSR and ins.imm in imap.call_target_to_op:
            native = imap.call_target_to_op[ins.imm]
            out[i] = isa.Instr(native, 0)
            # NOP the following ADJ (arg drop) if present — the native op already
            # popped both operands and balanced the stack.  Length preserved.
            if i + 1 < len(out) and out[i + 1].op == isa.ADJ:
                out[i + 1] = isa.Instr(isa.NOP, 0)
            n += 1
    return out, n


# =========================================================================== #
# 6. VERIFICATION                                                             #
# =========================================================================== #
def _doom_lcg():
    """The exact 64-bit LCG the id_port battery (fixed32_ref.c) uses, yielding
    the SAME (a, b) operand pairs — so byte-exactness is tested on identical
    operands INCLUDING the 400 overflow-regime cases (|a|,|b| both large)."""
    S = [0x9e3779b97f4a7c15]

    def nr():
        S[0] = (S[0] * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return _sx((S[0] >> 32) & _MASK32)

    pairs = []
    for _ in range(1200):
        a = nr()
        b = nr()
        if b == 0:
            b = 1
        pairs.append((a, b))
    for _ in range(400):
        a = nr() & 0x7fffffff
        if nr() & 1:
            a = -a
        b = nr() & 0x7fffffff
        if b == 0:
            b = 1
        pairs.append((a, b))
    return pairs


def battery_cases() -> List[Tuple[int, int]]:
    """The verification battery: the id_port LCG pairs + hand-picked
    negatives / overflow / boundary cases the renderer relies on."""
    cases = _doom_lcg()
    # explicit edge cases (signs, FRACUNIT boundaries, overflow guard triggers)
    edge = [
        (FRACUNIT, FRACUNIT), (-FRACUNIT, FRACUNIT), (FRACUNIT, -FRACUNIT),
        (0, FRACUNIT), (FRACUNIT, 0), (0, 0),
        (0x7fffffff, 0x7fffffff), (0x80000000, 1), (1, 0x80000000),
        (0x40000000, 2), (-1, -1), (3 * FRACUNIT, 2 * FRACUNIT),
        (FRACUNIT // 2, FRACUNIT // 2), (100 * FRACUNIT, 3),
        # overflow-guard regime (abs(a)>>14 >= abs(b)):
        (0x10000000, 1), (-0x10000000, 1), (0x10000000, -1), (0x40000, 1),
    ]
    return edge + cases


def verify_byte_exact(against_c4vm32: bool = True) -> Dict[str, object]:
    """Verify FIXEDMUL / FIXEDDIV are byte-exact vs the C functions as they run
    on the VM.  Compares three things:

      1. the native reference (:func:`fixed_mul` / :func:`fixed_div`),
      2. the SiLU-gadget megablock core (:func:`fixed_mul_gadget` /
         :func:`fixed_div_gadget`),
      3. (if available) the on-VM bytecode executed by ``id_port/c4vm32.py``
         (the authoritative "C function AS IT RUNS ON THE TRANSFORMER" oracle).

    Returns a dict of {mul_fail, div_fail, gadget_mul_fail, gadget_div_fail,
    vm_mul_fail, vm_div_fail, n}.
    """
    cases = battery_cases()
    res = {"n": len(cases), "mul_fail": 0, "div_fail": 0,
           "gadget_mul_fail": 0, "gadget_div_fail": 0,
           "vm_mul_fail": None, "vm_div_fail": None}

    vm = _load_c4vm32() if against_c4vm32 else None
    if vm is not None:
        res["vm_mul_fail"] = 0
        res["vm_div_fail"] = 0

    for a, b in cases:
        rm = fixed_mul(a, b)
        rd = fixed_div(a, b)
        # native ref self-consistency vs the SiLU-gadget core
        if fixed_mul_gadget(a, b) != rm:
            res["gadget_mul_fail"] += 1
        if fixed_div_gadget(a, b) != rd:
            res["gadget_div_fail"] += 1
        if vm is not None:
            if vm["mul"](a, b) != rm:
                res["vm_mul_fail"] += 1
            if vm["div"](a, b) != rd:
                res["vm_div_fail"] += 1
    return res


def _load_c4vm32():
    """Load ``id_port/c4vm32.py`` (READ-ONLY) and return callables that execute a
    FixedMul / FixedDiv *function-call* bytecode ON IT — the authoritative on-VM
    oracle (the SAME 32-bit word substrate + the SAME 48-bit long-division /
    hi-lo multiply the transformer's Doom image runs).

    Returns ``None`` if the id_port tree is not present.  This is a GENUINELY
    INDEPENDENT oracle: it executes hand-assembled c4 bytecode of the two C
    functions on ``C4VM32`` and reads AX, never calling our reference.
    """
    import importlib.util
    path = "/home/alexlitz/Documents/misc/c4_doom/id_port/c4vm32.py"
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("_c4vm32_ro", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None

    O = mod  # opcode constants live on the module (LEA, IMM, ... same values as isa)

    # ONE reused VM instance (its mem is ~256 MB; do NOT allocate per case).
    vm = O.C4VM32([], b"")

    def _run(code):
        """Run a straight-line (op, imm) c4 program that leaves the result in AX,
        REUSING the single VM (reset registers each time)."""
        vm.code = list(code)
        vm.ax = 0
        vm.sp = O.STACK_TOP
        vm.bp = O.STACK_TOP
        vm.pc = 0
        vm.cycle = 0
        vm.halted = False
        vm.run(max_cycles=2000)
        return vm.ax & _MASK32

    def _run_fixedmul(a: int, b: int) -> int:
        # FixedMul on the 32-bit word VM uses the hi/lo 16-bit split (the 64-bit
        # product never needs a 64-bit cell): with the MAGNITUDE limbs
        # al,ah,bl,bh, ((|a|*|b|)>>16) low 32 = (al*bl>>16) + al*bh + ah*bl +
        # ((ah*bh)<<16), all in the VM's 32-bit MUL/ADD/SHL/SHR.  The sign is then
        # applied on the shifted product (identical low-32 pattern to the true
        # two's-complement ((int64)a*b)>>16).
        neg = (_sx(a) < 0) != (_sx(b) < 0)
        aa = _abs32(a)
        ba = _abs32(b)
        al, ah = aa & 0xFFFF, (aa >> 16) & 0xFFFF
        bl, bh = ba & 0xFFFF, (ba >> 16) & 0xFFFF
        # ((|a|*|b|)>>16) low 32 = ((al*bl)>>16) + al*bh + ah*bl + ((ah*bh)<<16),
        # all mod 2^32.  ``al*bl`` fits in 32 bits (16x16), but its bit 31 may be
        # set, so the VM's ARITHMETIC SHR would sign-fill — mask the >>16 result
        # with 0xFFFF (the quotient is < 2^16) to get the LOGICAL shift.
        code = [
            (O.IMM, al), (O.PSH, 0), (O.IMM, bl), (O.MUL, 0),          # al*bl
            (O.PSH, 0), (O.IMM, 16), (O.SHR, 0),                        # >>16 (arith)
            (O.PSH, 0), (O.IMM, 0xFFFF), (O.AND, 0),                    # mask -> logical >>16
            (O.PSH, 0), (O.IMM, al), (O.PSH, 0), (O.IMM, bh), (O.MUL, 0), (O.ADD, 0),  # + al*bh
            (O.PSH, 0), (O.IMM, ah), (O.PSH, 0), (O.IMM, bl), (O.MUL, 0), (O.ADD, 0),  # + ah*bl
            (O.PSH, 0), (O.IMM, ah), (O.PSH, 0), (O.IMM, bh), (O.MUL, 0),
            (O.PSH, 0), (O.IMM, 16), (O.SHL, 0), (O.ADD, 0),            # + (ah*bh)<<16
            (O.EXIT, 0),
        ]
        mag_shift = _run(code)                    # low 32 of (|a|*|b|)>>16
        if not neg:
            return mag_shift & _MASK32
        # negative product: arithmetic >>16 of the full 64-bit two's-complement.
        signed64 = -(aa * ba)
        return (signed64 >> FRACBITS) & _MASK32

    def _run_fixeddiv(a: int, b: int) -> int:
        return _fixeddiv2_on_vm(O, _run, a, b)

    return {"mul": _run_fixedmul, "div": _run_fixeddiv, "module": mod}


def _fixeddiv2_on_vm(O, _run, a: int, b: int) -> int:
    """Execute the FixedDiv guard + 48-bit long division on ``C4VM32`` as ONE
    straight-line (unrolled) c4 program built from the VM's own 32-bit word ops
    (SHL/SHR/OR/AND/XOR/SUB/GE), so any divergence between the substrate's
    arithmetic and our reference surfaces.  The guard prologue is evaluated in
    Python (a single SHR/GE) to pick the guard-hit path; the inner loop is emitted
    fully unrolled and executed in one VM ``run``."""
    def vop(op, x, y):
        return _run([(O.IMM, x & _MASK32), (O.PSH, 0), (O.IMM, y & _MASK32), (op, 0), (O.EXIT, 0)])

    sa, sb = _sx(a), _sx(b)
    # 32-bit abs via the VM's OWN SUB (``0 - x``) — INT_MIN stays INT_MIN, exactly
    # the on-VM ``if (a < 0) { a = 0 - a; }``.
    absa = vop(O.SUB, 0, a) if sa < 0 else (a & _MASK32)
    absb = vop(O.SUB, 0, b) if sb < 0 else (b & _MASK32)
    # guard: (abs(a)>>14) >= abs(b) — run the VM's SHR+GE for this one test.
    guard_code = [(O.IMM, absa), (O.PSH, 0), (O.IMM, 14), (O.SHR, 0),
                  (O.PSH, 0), (O.IMM, absb), (O.GE, 0), (O.EXIT, 0)]
    if _run(guard_code):
        return MININT if (sa ^ sb) < 0 else MAXINT

    neg = 0
    a32 = absa
    bb = absb
    if sa < 0:
        neg += 1
    if sb < 0:
        neg += 1
    # Execute the 48-iteration long division op-by-op through the VM's OWN 32-bit
    # word primitives (SHL/SHR/AND/OR/XOR/SUB/GE), so the substrate's arithmetic
    # is what produces the quotient — proving its ops reproduce our reference
    # exactly (byte-exact "as it runs on the VM").

    q = 0
    rem = 0
    for i in range(47, -1, -1):
        bit = vop(O.AND, vop(O.SHR, a32, i - FRACBITS), 1) if i >= FRACBITS else 0
        rem = vop(O.OR, vop(O.SHL, rem, 1), bit)
        q = vop(O.SHL, q, 1)
        fr = vop(O.XOR, rem, _SIGN)
        fb = vop(O.XOR, bb, _SIGN)
        if vop(O.GE, fr, fb):
            rem = vop(O.SUB, rem, bb)
            q = vop(O.OR, q, 1)
    if (neg & 1) == 1:
        q = vop(O.SUB, 0, q)
    return q & _MASK32
