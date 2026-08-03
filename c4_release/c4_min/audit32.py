#!/usr/bin/env python3
"""c4_min 32-BIT EXACTNESS AUDIT — the per-opcode, per-case, per-PATH matrix.

WHY
---
The existing ``edge_corpus.py`` pins the doom-critical UNTESTED opcodes but its
operands are almost all < 256 (the ISA ``IMM`` folds a constant to its low byte,
so a hand-assembled ``IMM 0x80000000`` is impossible).  A doom port drives the
FULL opcode set at the FULL 32-bit range — mid-range values, ``INT_MIN`` /
``INT_MAX``, byte / 16-bit / 32-bit carries, signed shifts of negatives.  This
module is the SYSTEMATIC 32-bit exactness audit: every ISA opcode across sign
combos + boundaries + carries, run through EVERY execution path, with every gap
localized.

INJECTING A GENUINE 32-BIT OPERAND
----------------------------------
Two mechanisms bypass the 8-bit ``IMM`` fold:
  * ``seed_mem={addr: word}``  — a 32-bit word placed in memory; ``LI addr``
    reads it back at full width (both the golden here and the neural model do a
    32-bit ``LI`` when the word is seeded as a §Memory KV frame).  This is the
    doom operand path (values spill to a ``char*`` / ``int*`` buffer and are
    read back).
  * ALU construction — ``1 << 31`` (``SHL``) builds ``INT_MIN``; ``0 - 1``
    (``SUB``) builds ``0xFFFFFFFF``; combine for any 32-bit literal.  Pure-IMM,
    no seed.

THE PATHS
---------
  GOLDEN   ``golden32`` (below) — the 32-bit c4_min reference the NEURAL model
           reproduces byte-exactly.  Semantics (matching ``ref_interpret`` +
           the neural gadgets): ADD/SUB/MUL wrap mod 2^32; DIV/MOD UNSIGNED
           floor (b==0 -> 0, ISA_SPEC 4.2); LT/GT/LE/GE SIGNED 32-bit; EQ/NE
           bit-equal; SHL logical, SHR ARITHMETIC (sign-fill on the signed
           operand); OR/XOR/AND stay 8-bit (per-nibble table over the loaded
           byte); LC SIGN-extends a byte, LI zero-extends; SC/SI store the low
           byte / full word; IMM/LEA fold to 8 bits.  seed_mem LI is 32-bit.
  NEURAL   the pure-forward model (``run_pure_forward_complete``, ``C4_PF_CFM=1``
           — see the note in ``audit_matrix.py``).  The primary byte-exact
           target.
  DRAFT    ``ref_interpret_word32`` (``selfhost/word32_draft_vm.py``) — the
           speculative doom fast path.  DIVERGES from GOLDEN on: SHR (LOGICAL,
           not arithmetic), LC (UNSIGNED 32-bit load, no sign-extend), DIV/MOD
           (SIGNED C-trunc, not unsigned floor), OR/XOR/AND (32-bit, not 8-bit).
           Flagged per-case.
  NATIVE   ``./c4`` (the reference c4 binary) — spot-checked from C source where
           a C program can express the case (``audit_matrix.py --native``).

This module authors NO weights (golden 069cc32f untouched); it only defines the
case corpus + the golden VM.  ``audit_matrix.py`` runs the matrix; the permanent
regression gate is ``tests/test_32bit_exact.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from c4_min import isa

M32 = 0xFFFFFFFF
SIGN = 1 << 31
INT_MIN = 0x80000000
INT_MAX = 0x7FFFFFFF


def s32(v: int) -> int:
    """Signed 32-bit interpretation of a masked word."""
    v &= M32
    return v - (1 << 32) if v & SIGN else v


# ===========================================================================
# THE GOLDEN 32-bit VM.  Mirrors ``ref_interpret(mask=0xFFFFFFFF)`` EXACTLY but
# adds a ``seed_mem`` data segment with a 32-bit ``LI`` (the neural model's
# seed-frame contract), so a genuine 32-bit operand can be injected and read
# back.  Every other op is byte-for-byte the production ``ref_interpret``.
# ===========================================================================
def golden32(code: List[isa.Instr], max_steps: int = 512,
             seed_mem: Optional[Dict[int, int]] = None,
             out: Optional[List[int]] = None) -> List[int]:
    """32-bit golden AX trace.  ``seed_mem`` {addr: word} pre-seeds memory (LI
    reads it 32-bit; LC reads its low byte sign-extended)."""
    mem: Dict[int, int] = dict(seed_mem or {})
    # SP base: use the SAME SP_INIT the neural driver + word32 draft use (the
    # runners pin it to 0xFC), so LEA / ENT / frame / stack-address cases are
    # apples-to-apples across the three paths.
    import c4_min.nibble_pure_forward_complete as _PFC
    sp = bp = _PFC.SP_INIT
    ax = pc = steps = 0
    trace: List[int] = []
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFF
        elif op == isa.PSH:
            sp -= 4; mem[sp] = ax & M32
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0) & M32; sp += 4
            if op == isa.ADD:
                ax = (v + ax) & M32
            elif op == isa.SUB:
                ax = (v - ax) & M32
            elif op == isa.MUL:
                ax = (v * ax) & M32
            elif op == isa.DIV:
                ax = ((v // ax) if ax else 0) & M32          # UNSIGNED floor
            else:
                ax = ((v % ax) if ax else 0) & M32           # UNSIGNED floor
        elif op in (isa.OR, isa.XOR, isa.AND):
            v = mem.get(sp, 0) & 0xFF; sp += 4               # 8-bit bitwise
            if op == isa.OR:
                ax = (v | (ax & 0xFF)) & 0xFF
            elif op == isa.XOR:
                ax = (v ^ (ax & 0xFF)) & 0xFF
            else:
                ax = (v & (ax & 0xFF)) & 0xFF
        elif op in (isa.SHL, isa.SHR):
            v = mem.get(sp, 0) & M32; sp += 4
            if op == isa.SHL:
                ax = (v << (ax & M32)) & M32                 # logical
            else:
                sv = s32(v)                                  # ARITHMETIC (sign-fill)
                ax = (sv >> (ax & M32)) & M32
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0) & M32; sp += 4; av = ax & M32
            sv, sax = s32(v), s32(av)
            r = {isa.EQ: v == av, isa.NE: v != av, isa.LT: sv < sax,
                 isa.GT: sv > sax, isa.LE: sv <= sax, isa.GE: sv >= sax}[op]
            ax = 1 if r else 0
        elif op == isa.LI:
            ax = mem.get(ax, 0) & M32                        # 32-bit load
        elif op == isa.LC:
            b = mem.get(ax, 0) & 0xFF                        # signed char load
            ax = (b - 0x100 if b & 0x80 else b) & M32
        elif op == isa.SI:
            addr = mem.get(sp, 0); sp += 4; mem[addr] = ax & M32   # 32-bit store
        elif op == isa.SC:
            addr = mem.get(sp, 0); sp += 4; mem[addr] = ax & 0xFF  # byte store
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            sp -= 4; mem[sp] = (i + 1) & M32; pc = imm
        elif op == isa.ENT:
            mem[sp - 4] = bp & M32; sp -= 4; bp = sp; sp -= 4 * imm
        elif op == isa.ADJ:
            sp += 4 * imm
        elif op == isa.LEV:
            sp = bp; bp = mem.get(sp, 0); pc = mem.get(sp + 4, 0); sp += 8
        elif op == isa.PRTF:
            if out is not None:
                out.append(ax & 0xFF)
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            trace.append(ax & M32); break
        else:
            raise NotImplementedError(f"golden32: op {isa.NAMES.get(op, op)}")
        trace.append(ax & M32)
    return trace


# ===========================================================================
# Assembly helpers — inject a genuine 32-bit operand.
# ===========================================================================
def li(addr: int) -> List[Tuple[str, int]]:
    """LI mem[addr] (32-bit load of a seeded word)."""
    return [("IMM", addr & 0xFF), ("LI", 0)]


def push_word(addr: int) -> List[Tuple[str, int]]:
    """Load a seeded 32-bit word into AX then PSH it (stack operand)."""
    return li(addr) + [("PSH", 0)]


def binop_words(a_addr: int, b_addr: int, op: str) -> List[Tuple[str, int]]:
    """``mem[a] OP mem[b]`` : LI a ; PSH ; LI b ; OP ; HALT.  Both operands are
    genuine 32-bit words read from the seeded data segment."""
    return push_word(a_addr) + li(b_addr) + [(op, 0), ("HALT", 0)]


def shift_word(a_addr: int, amt: int, op: str) -> List[Tuple[str, int]]:
    """``mem[a] SHL/SHR amt`` : LI a ; PSH ; IMM amt ; OP ; HALT.  The operand is
    a genuine 32-bit word; the shift amount is a small (< 256) IMM."""
    return push_word(a_addr) + [("IMM", amt & 0xFF), (op, 0), ("HALT", 0)]


# ===========================================================================
# Case model.
# ===========================================================================
@dataclass
class Case:
    name: str
    op: str                              # the ISA opcode under test
    body: List[Tuple[str, int]]          # (opname, imm) program
    seed_mem: Dict[int, int] = field(default_factory=dict)
    note: str = ""
    # per-path EXPECTATION (None => same as golden; a value => that path DIVERGES
    # here and this is the documented divergent value).
    draft_diverges: Optional[int] = None   # ref_interpret_word32 value if != golden
    neural_xfail: bool = False             # documented neural gap (#702 etc.)
    max_steps: int = 64
    shift_amt: Optional[int] = None        # for SHL/SHR: the (explicit) shift amount

    def code(self) -> List[isa.Instr]:
        return isa.assemble(self.body)

    def golden(self) -> int:
        tr = golden32(self.code(), max_steps=self.max_steps, seed_mem=self.seed_mem)
        return tr[-1] & M32 if tr else 0


# The two data-segment slots (a, b) genuine 32-bit operands live at.
A = 0x40
B = 0x44


# ---------------------------------------------------------------------------
# The 32-bit operand boundary set the audit sweeps.
# ---------------------------------------------------------------------------
BOUNDARIES = [
    0x00000000, 0x00000001, 0xFFFFFFFF,          # 0, 1, -1
    INT_MIN, INT_MAX, 0x80000001,                # INT_MIN, INT_MAX, INT_MIN+1
    0x00000100, 0x0000FFFF, 0x00010000,          # byte / 16-bit / carry boundaries
    0x7FFFFFFE, 0x00008000, 0xFFFF0000,          # near-max, 15-bit, hi-half
    0x12345678, 0xDEADBEEF, 0x0000002A,          # mid-range, mid-range, 42
]


def _sign_pairs():
    """Representative (+,+) (+,-) (-,+) (-,-) 32-bit operand pairs."""
    P, N = 0x0000002A, 0xFFFFFFD6      # +42, -42
    P2, N2 = 0x00000007, 0xFFFFFFFB    # +7, -5
    return [
        ("pp", P, P2), ("pn", P, N2), ("np", N, P2), ("nn", N, N2),
    ]


# ===========================================================================
# CLUSTER: ADD/SUB — carries / borrows across byte, 16-bit, 32-bit boundaries.
# ===========================================================================
def _cluster_add_sub() -> List[Case]:
    cs: List[Case] = []
    add_cases = [
        ("add_byte_carry", 0x000000FF, 0x00000001, "255+1 -> byte carry"),
        ("add_16_carry", 0x0000FFFF, 0x00000001, "0xFFFF+1 -> 16-bit carry"),
        ("add_32_wrap", 0xFFFFFFFF, 0x00000001, "0xFFFFFFFF+1 -> 0 (32-bit wrap)"),
        ("add_intmax_1", INT_MAX, 0x00000001, "INT_MAX+1 -> INT_MIN (signed ovf)"),
        ("add_mid", 0x12345678, 0x11111111, "mid-range add, no wrap"),
        ("add_intmin_intmin", INT_MIN, INT_MIN, "INT_MIN+INT_MIN -> 0 (wrap)"),
        ("add_neg_neg", 0xFFFFFFD6, 0xFFFFFFFB, "-42 + -5 -> -47"),
    ]
    for nm, a, b, note in add_cases:
        cs.append(Case(nm, "ADD", binop_words(A, B, "ADD"),
                       seed_mem={A: a, B: b}, note=note))
    sub_cases = [
        ("sub_borrow", 0x00000000, 0x00000001, "0-1 -> -1 (borrow cascade)"),
        ("sub_16_borrow", 0x00010000, 0x00000001, "0x10000-1 -> 0xFFFF"),
        ("sub_intmin_1", INT_MIN, 0x00000001, "INT_MIN-1 -> INT_MAX (signed ovf)"),
        ("sub_mid", 0xDEADBEEF, 0x0000BEEF, "mid-range sub"),
        ("sub_neg_pos", 0xFFFFFFD6, 0x0000002A, "-42 - 42 -> -84"),
        ("sub_eq", 0x12345678, 0x12345678, "x - x -> 0"),
    ]
    for nm, a, b, note in sub_cases:
        cs.append(Case(nm, "SUB", binop_words(A, B, "SUB"),
                       seed_mem={A: a, B: b}, note=note))
    return cs


# ===========================================================================
# CLUSTER: MUL — products exceeding 32 bits (truncation).
# ===========================================================================
def _cluster_mul() -> List[Case]:
    cs: List[Case] = []
    mul_cases = [
        ("mul_small", 0x00000040, 0x00000004, "64*4 -> 256"),
        ("mul_16_16", 0x0000FFFF, 0x0000FFFF, "0xFFFF*0xFFFF -> 0xFFFE0001"),
        ("mul_overflow", 0x00010000, 0x00010000, "0x10000*0x10000 -> 0 (>32b trunc)"),
        ("mul_by_zero", 0x12345678, 0x00000000, "x*0 -> 0"),
        ("mul_by_one", 0xDEADBEEF, 0x00000001, "x*1 -> x"),
        ("mul_max_2", INT_MAX, 0x00000002, "INT_MAX*2 -> 0xFFFFFFFE (trunc)"),
        ("mul_neg_pos", 0xFFFFFFFF, 0x00000002, "0xFFFFFFFF*2 -> 0xFFFFFFFE (unsigned)"),
        ("mul_pow2", 0x00000003, 0x40000000, "3*2^30 -> 0xC0000000 (trunc)"),
    ]
    for nm, a, b, note in mul_cases:
        cs.append(Case(nm, "MUL", binop_words(A, B, "MUL"),
                       seed_mem={A: a, B: b}, note=note))
    return cs


# ===========================================================================
# CLUSTER: DIV/MOD — unsigned floor (golden/neural) vs signed C-trunc (draft).
# ===========================================================================
def _cluster_div_mod() -> List[Case]:
    cs: List[Case] = []
    # Positive/positive: golden==draft==native (all agree).
    for nm, a, b, q, note in [
        ("div_pp", 100, 7, 14, "100/7 -> 14"),
        ("div_exact", 0x00010000, 0x00000100, 0x100, "0x10000/0x100 -> 0x100"),
        ("div_big", 0x7FFFFFFF, 0x00010000, 0x7FFF, "INT_MAX/0x10000"),
        ("div_by_1", 0xDEADBEEF, 1, 0xDEADBEEF, "x/1 -> x"),
    ]:
        cs.append(Case(nm, "DIV", binop_words(A, B, "DIV"),
                       seed_mem={A: a, B: b}, note=note))
    for nm, a, b, r, note in [
        ("mod_pp", 100, 7, 2, "100%7 -> 2"),
        ("mod_16", 0x0000FFFF, 0x00000010, 0x0F, "0xFFFF%0x10 -> 15"),
        ("mod_big", 0x7FFFFFFF, 0x00010000, 0xFFFF, "INT_MAX % 0x10000"),
    ]:
        cs.append(Case(nm, "MOD", binop_words(A, B, "MOD"),
                       seed_mem={A: a, B: b}, note=note))
    # div/mod by zero -> 0 (ISA guard) — golden & draft both 0 here.
    cs.append(Case("div_by_zero", "DIV", binop_words(A, B, "DIV"),
                   seed_mem={A: 7, B: 0}, note="7/0 -> 0 (ISA guard)"))
    cs.append(Case("mod_by_zero", "MOD", binop_words(A, B, "MOD"),
                   seed_mem={A: 7, B: 0}, note="7%0 -> 0 (ISA guard)"))
    # NEGATIVE dividend: golden = UNSIGNED floor; draft = SIGNED C-trunc.  These
    # DIVERGE by path (documented).  (0-1)=0xFFFFFFFF as an UNSIGNED /1 == itself;
    # signed it's -1/1 == -1 == 0xFFFFFFFF too — pick cases where they differ.
    #   0xFFFFFFFF / 2 : unsigned 0x7FFFFFFF ; signed (-1)/2 == 0 (C trunc).
    dm = golden32  # local
    for nm, a, b, note in [
        ("div_neg_by2", 0xFFFFFFFF, 2, "(-1)/2: unsigned 0x7FFFFFFF vs signed 0"),
        ("div_intmin_neg1", INT_MIN, 0xFFFFFFFF,
         "INT_MIN/-1: unsigned 0 vs signed overflow"),
        ("div_neg_pos", 0xFFFFFFD6, 5, "(-42)/5: unsigned huge vs signed -8"),
    ]:
        c = Case(nm, "DIV", binop_words(A, B, "DIV"),
                 seed_mem={A: a, B: b}, note=note)
        cs.append(c)
    for nm, a, b, note in [
        ("mod_neg_pos", 0xFFFFFFD6, 5, "(-42)%5: unsigned vs signed -2 (C trunc)"),
        ("mod_neg_by2", 0xFFFFFFFF, 2, "(-1)%2: unsigned 1 vs signed -1"),
    ]:
        c = Case(nm, "MOD", binop_words(A, B, "MOD"),
                 seed_mem={A: a, B: b}, note=note)
        cs.append(c)
    return cs


# ===========================================================================
# CLUSTER: SHL/SHR — by 0/1/15/31; arithmetic-vs-logical; MULTI-BYTE.
# ===========================================================================
# NEURAL SHL/SHR GAP (audited 2026-08, BROADER than the documented #702):
# the neural SHL/SHR gadget reads its OPERAND from the 1-slot STACK0 relay, which
# carries only the LOW BYTE of the pushed word, and writes only a byte-wide result.
# So EVERY SHL/SHR whose OPERAND *or* RESULT exceeds one byte diverges on the
# neural path (it returns the low byte, e.g. ``1<<15 -> 0x00``, ``0x7FFFFFFE>>1 ->
# 0xFF``) — NOT just the "signed-SHR-of-a-negative" arithmetic sign-fill case #702
# names.  #702's negatives are a SUBSET.  These are all ``neural_xfail`` (they are
# byte-exact vs the GOLDEN, so the divergence is NEURAL-only).  Doom uses
# fixed-point ``>>`` heavily -> this is the #1 doom-fast-path fix.
def _cluster_shifts() -> List[Case]:
    cs: List[Case] = []
    # Byte-sized SHL/SHR (operand AND result fit a byte) — the neural model DOES
    # get these right (positive controls that the shift gadget itself works).
    cs.append(Case("shl_1_by0", "SHL", shift_word(A, 0, "SHL"),
                   seed_mem={A: 1}, note="1 << 0 == 1 (byte)", shift_amt=0))
    cs.append(Case("shl_1_by1", "SHL", shift_word(A, 1, "SHL"),
                   seed_mem={A: 1}, note="1 << 1 == 2 (byte)", shift_amt=1))
    cs.append(Case("shl_3_by4", "SHL", shift_word(A, 4, "SHL"),
                   seed_mem={A: 3}, note="3 << 4 == 0x30 (byte result)", shift_amt=4))
    cs.append(Case("shr_7f_by1", "SHR", shift_word(A, 1, "SHR"),
                   seed_mem={A: 0x7F}, note="0x7F >> 1 == 0x3F (byte)", shift_amt=1))
    cs.append(Case("shr_f0_by4", "SHR", shift_word(A, 4, "SHR"),
                   seed_mem={A: 0xF0}, note="0xF0 >> 4 == 0x0F (byte)", shift_amt=4))
    # MULTI-BYTE SHL (result exceeds a byte) — NEURAL truncates to the low byte.
    for amt, seed, note in [(15, 1, "1<<15 -> 0x8000"), (31, 1, "1<<31 -> INT_MIN")]:
        cs.append(Case(f"shl_1_by{amt}", "SHL", shift_word(A, amt, "SHL"),
                       seed_mem={A: seed}, note=note, shift_amt=amt,
                       neural_xfail=True))
    cs.append(Case("shl_mid_by4", "SHL", shift_word(A, 4, "SHL"),
                   seed_mem={A: 0x0FFFFFFF}, note="0x0FFFFFFF<<4 (top bits lost)",
                   shift_amt=4, neural_xfail=True))
    cs.append(Case("shl_sign_by1", "SHL", shift_word(A, 1, "SHL"),
                   seed_mem={A: 0x40000000}, note="0x40000000<<1 -> INT_MIN",
                   shift_amt=1, neural_xfail=True))
    # MULTI-BYTE SHR of a POSITIVE word — golden == draft (both logical on
    # positives) but NEURAL truncates to the low byte (the broader gap).
    cs.append(Case("shr_pos_by1", "SHR", shift_word(A, 1, "SHR"),
                   seed_mem={A: 0x7FFFFFFE}, note="0x7FFFFFFE>>1 -> 0x3FFFFFFF",
                   shift_amt=1, neural_xfail=True))
    cs.append(Case("shr_pos_by16", "SHR", shift_word(A, 16, "SHR"),
                   seed_mem={A: 0x12345678}, note="0x12345678>>16 -> 0x1234",
                   shift_amt=16, neural_xfail=True))
    # SHR of a NEGATIVE 32-bit word — golden ARITHMETIC (sign-fill) vs draft
    # LOGICAL (the DRAFT also diverges).  The #702 signed-SHR cases.
    for amt, note in [(1, "-1>>1"), (4, "-1>>4"), (31, "-1>>31")]:
        c = Case(f"shr_neg1_by{amt}", "SHR", shift_word(A, amt, "SHR"),
                 seed_mem={A: 0xFFFFFFFF}, note=f"(-1)>>{amt} == -1 (arith)",
                 neural_xfail=True, shift_amt=amt)
        cs.append(c)
    cs.append(Case("shr_intmin_by1", "SHR", shift_word(A, 1, "SHR"),
                   seed_mem={A: INT_MIN}, note="INT_MIN>>1 -> 0xC0000000 (arith)",
                   neural_xfail=True, shift_amt=1))
    cs.append(Case("shr_neg_by8", "SHR", shift_word(A, 8, "SHR"),
                   seed_mem={A: 0xFF000000}, note="0xFF000000>>8 -> 0xFFFF0000 (arith)",
                   neural_xfail=True, shift_amt=8))
    return cs


# ===========================================================================
# CLUSTER: bitwise — AND/OR/XOR on 32-bit operands (golden is 8-BIT; draft 32b).
# ===========================================================================
def _cluster_bitwise() -> List[Case]:
    cs: List[Case] = []
    # NB: the golden (and neural model) do bitwise on the LOW BYTE only (per-nibble
    # table over the loaded byte); the DRAFT does full 32-bit bitwise.  A 32-bit
    # operand therefore diverges golden-vs-draft above the low byte.
    for nm, a, b, note in [
        ("and_ff_low", 0x12345678, 0x000000FF, "x & 0xFF (low byte survives)"),
        ("or_low", 0x00000080, 0x00000001, "0x80 | 1"),
        ("xor_self", 0x0000005A, 0x0000005A, "x ^ x -> 0"),
        ("and_sign", 0x000000FF, 0x00000080, "0xFF & 0x80 -> 0x80"),
    ]:
        cs.append(Case(nm, {"and": "AND", "or": "OR", "xor": "XOR"}[nm.split("_")[0]],
                       binop_words(A, B, {"and": "AND", "or": "OR", "xor": "XOR"}[nm.split("_")[0]]),
                       seed_mem={A: a, B: b}, note=note))
    return cs


# ===========================================================================
# CLUSTER: comparisons — SIGNED 32-bit ordering across the sign boundary.
# ===========================================================================
# NEURAL COMPARISON GAP (audited 2026-08): the SIGNED ORDER gadget (LT/GT/LE/GE)
# is 32-bit-exact when the sign bit OR a low byte is DECISIVE — it correctly ranks
# every sign-crossing pair (neg vs pos, INT_MIN vs INT_MAX, pos vs neg).  But the
# EQUALITY resolution across HIGH bytes is byte-limited: when two operands are equal
# in the full 32-bit word (the ``eq_big`` pair, 0x12345678==0x12345678) the neural
# EQ / GT / LE mis-decide the tie (EQ->0, GT->1, LE->0).  NE is broadly broken on
# multi-byte operands (it returns garbage, e.g. 0x21, on neg_vs_pos / INT_MIN-vs-MAX
# / eq_big — only pos_vs_neg's low bytes save it).  LT/GE happen to land the right
# answer on the tested pairs.  These are ``neural_xfail`` (byte-exact vs GOLDEN).
def _cluster_cmp() -> List[Case]:
    cs: List[Case] = []
    cmp_ops = ["EQ", "NE", "LT", "GT", "LE", "GE"]
    # sign-crossing pairs where signed vs unsigned order DIFFERS.
    pairs = [
        ("neg_vs_pos", 0xFFFFFFFF, 0x00000001, "-1 vs 1"),
        ("intmin_vs_intmax", INT_MIN, INT_MAX, "INT_MIN vs INT_MAX"),
        ("eq_big", 0x12345678, 0x12345678, "x vs x"),
        ("pos_vs_neg", 0x00000005, 0xFFFFFFFB, "5 vs -5"),
        # WIDE-MAGNITUDE ORDER (#825, the doom step-30,850 GT wall): the decisive
        # nibble sits in a HIGH byte, so the golden ``step(STK_VAL-AX_VAL)`` on the
        # ~10^9-scale wide-scalar recompose collapses under fp32's 2^24 exact range.
        # C4_CMP32's per-nibble lexicographic order decides these exactly.
        ("doom_heap_ptr", 0x40001000, 0x00001000, "1073744896 vs 4096 (GT=1)"),
        ("hi_byte_gt", 0x40000000, 0x3FFFFFFF, "hi-byte decides GT (2^30 vs 2^30-1)"),
        ("hi_byte_lt", 0x01000000, 0x01000001, "hi-byte tie, low nibble LT"),
        ("wide_pos_gt", 0x7FFFFF00, 0x00000100, "large pos vs small pos"),
        ("both_neg_ord", 0xFFFFFF00, 0xFFFFFFF0, "-256 vs -16 (both neg, LT)"),
        ("big_neg_vs_big_pos", 0x80001000, 0x40001000, "2^31-block neg vs pos"),
    ]
    # (op, pair) combinations the NEURAL model gets wrong WITHOUT the C4_CMP32 order
    # extension (measured 2026-08).  The wide-magnitude pairs are ALL xfail off-flag
    # (the golden wide-scalar order collapses) and become PASSES under C4_CMP32.
    _WIDE = {"doom_heap_ptr", "hi_byte_gt", "hi_byte_lt", "wide_pos_gt",
             "both_neg_ord", "big_neg_vs_big_pos"}
    _NEURAL_CMP_XFAIL = {
        ("EQ", "eq_big"),
        ("NE", "neg_vs_pos"), ("NE", "intmin_vs_intmax"), ("NE", "eq_big"),
        ("GT", "eq_big"), ("LE", "eq_big"),
    }
    # every order op on a wide pair is off-flag-broken (the high-byte order collapses).
    for op in ("LT", "GT", "LE", "GE", "NE", "EQ"):
        for pname in _WIDE:
            _NEURAL_CMP_XFAIL.add((op, pname))
    for op in cmp_ops:
        for pname, a, b, note in pairs:
            cs.append(Case(f"{op.lower()}_{pname}", op,
                           binop_words(A, B, op), seed_mem={A: a, B: b},
                           note=f"{op}: {note}",
                           neural_xfail=((op, pname) in _NEURAL_CMP_XFAIL)))
    return cs


# ===========================================================================
# CLUSTER: LC/SC/LI/SI — sign-extend, truncate, 32-bit load/store.
# ===========================================================================
def _cluster_mem() -> List[Case]:
    cs: List[Case] = []
    # LC sign-extends the byte at a seeded address (only the low byte matters).
    for b, want, note in [(0x00, 0, "LC 0x00"), (0x7F, 0x7F, "LC 0x7F"),
                          (0x80, 0xFFFFFF80, "LC 0x80 sign-extend"),
                          (0xFF, 0xFFFFFFFF, "LC 0xFF -> -1")]:
        cs.append(Case(f"lc_{b:02x}", "LC", li(A)[:1] + [("LC", 0), ("HALT", 0)],
                       seed_mem={A: b}, note=note))
    # LI zero-extends a seeded 32-bit word (full width).
    for v, note in [(0x80, "LI 0x80 stays 128"), (0x12345678, "LI 32-bit word"),
                    (0xFFFFFFFF, "LI 0xFFFFFFFF")]:
        cs.append(Case(f"li_{v:08x}", "LI", [("IMM", A), ("LI", 0), ("HALT", 0)],
                       seed_mem={A: v}, note=note))
    # SC truncates a 32-bit AX to a byte then LC reads it back sign-extended.
    #   LI a (0x1FF via seed low byte 0xFF) ; ... but seed a full word then SC to
    #   B, LC B back.
    cs.append(Case("sc_trunc", "SC",
                   li(A) + [("PSH", 0), ("IMM", B), ("PSH", 0)]  # this is wrong order
                   , seed_mem={A: 0x1FF}, note="placeholder"))
    # Fix sc_trunc properly: IMM B (addr) ; PSH ; LI a (val) ; SC (store low byte) ;
    #   IMM B ; LC (read back signed).
    cs[-1] = Case("sc_trunc_lc", "SC",
                  [("IMM", B), ("PSH", 0)] + li(A) + [("SC", 0), ("IMM", B), ("LC", 0), ("HALT", 0)],
                  seed_mem={A: 0x000001FF},
                  note="SC truncates 0x1FF->0xFF, LC reads -1")
    # SI stores a full 32-bit word then LI reads it back.
    cs.append(Case("si_li_roundtrip", "SI",
                   [("IMM", B), ("PSH", 0)] + li(A) + [("SI", 0), ("IMM", B), ("LI", 0), ("HALT", 0)],
                   seed_mem={A: 0xDEADBEEF},
                   note="SI stores 0xDEADBEEF, LI reads it back 32-bit"))
    return cs


# ===========================================================================
# CLUSTER: control / stack / frame — IMM, LEA, PSH, JMP, BZ, BNZ, JSR/ENT/ADJ/LEV.
# ===========================================================================
def _cluster_control() -> List[Case]:
    cs: List[Case] = []
    # IMM folds to 8 bits (documented): IMM 0x2A -> 0x2A.
    cs.append(Case("imm_byte", "IMM", [("IMM", 0x2A), ("HALT", 0)], note="IMM 0x2A"))
    # BZ taken / not, BNZ taken / not (branch on 32-bit AX).
    cs.append(Case("bz_taken", "BZ",
                   [("IMM", 0), ("BZ", 4), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
                   note="BZ taken -> 7"))
    cs.append(Case("bnz_taken", "BNZ",
                   [("IMM", 1), ("BNZ", 4), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
                   note="BNZ taken -> 7"))
    cs.append(Case("bnz_not_taken", "BNZ",
                   [("IMM", 0), ("BNZ", 4), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
                   note="BNZ not taken -> 9"))
    # BNZ on a seeded 32-bit nonzero-in-high-bytes value (low byte 0): a value like
    # 0x10000 is NONZERO so BNZ must branch even though its low byte is 0 — a 32-bit
    # branch-condition edge.
    cs.append(Case("bnz_high_only", "BNZ",
                   li(A) + [("BNZ", 5), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
                   seed_mem={A: 0x00010000},
                   note="BNZ on 0x10000 (low byte 0, high nonzero) must branch"))
    cs.append(Case("bz_high_only", "BZ",
                   li(A) + [("BZ", 5), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
                   seed_mem={A: 0x00010000},
                   note="BZ on 0x10000 must NOT branch -> 9"))
    # JMP.
    cs.append(Case("jmp", "JMP",
                   [("JMP", 3), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
                   note="JMP over -> 7"))
    # JSR/ENT/ADJ/LEV round trip: call a leaf that returns a constant.
    #   main: JSR 3 ; HALT ; leaf: ENT 0 ; IMM 42 ; LEV
    cs.append(Case("jsr_leaf", "JSR",
                   [("JSR", 2), ("HALT", 0), ("ENT", 0), ("IMM", 42), ("LEV", 0)],
                   note="JSR to leaf returning 42"))
    # LEA computes a frame-relative address (BP + 4*imm) low byte.
    cs.append(Case("lea_frame", "LEA",
                   [("ENT", 0), ("LEA", 1), ("HALT", 0)],
                   note="LEA BP+4 (frame-relative addr)"))
    return cs


# ===========================================================================
# Whole corpus.
# ===========================================================================
def generate_cases() -> List[Case]:
    cases: List[Case] = []
    for builder in (_cluster_add_sub, _cluster_mul, _cluster_div_mod,
                    _cluster_shifts, _cluster_bitwise, _cluster_cmp,
                    _cluster_mem, _cluster_control):
        cases.extend(builder())
    seen = set()
    for c in cases:
        assert c.name not in seen, f"duplicate case {c.name}"
        seen.add(c.name)
    return cases


if __name__ == "__main__":
    from collections import Counter
    cs = generate_cases()
    by_op = Counter(c.op for c in cs)
    print(f"32-bit audit corpus: {len(cs)} cases across {len(by_op)} opcodes")
    for op, n in sorted(by_op.items()):
        print(f"  {op:6s} {n}")
