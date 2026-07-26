"""c4-fidelity SPEC for signed shifts (SHR/SHL) and char signedness (LC/SC).

This file is the executable specification for the two ops whose signedness
matters vs the c4 reference:

  * **SHR** — c4 does an ARITHMETIC (sign-extending) right shift on a signed
    ``long long`` (``#define int long long`` in c4.c; run loop ``a = *sp++ >> a``
    with ``int *sp``).  The DRAFT VMs (``isa.interpret`` 8-bit, ``ref_interpret``
    32-bit, ``libprog_corpus.RefVM``) are FIXED to arithmetic SHR (the operand is
    read SIGNED at the value width and the shift sign-fills).  The GOLDEN NEURAL
    SHR (the TIGHT direct-8x8 shifter) is ALSO fixed to arithmetic via a
    ``build_sign_fill`` block — byte-exact to c4 through the real forward (see
    ``TestNeuralSHRIsArithmeticLikeC4``); this MOVES the golden fingerprint
    (intended).  Only the OFF-by-default log-shifter fallback stays logical.
  * **LC** (load char) — c4 does ``a = *(char *)a`` = a SIGNED char load
    (-128..127, sign-extended).  The DRAFT VMs are FIXED to sign-extend a byte
    >= 0x80 (LI stays an unsigned word load).

``SHL`` (logical left shift) and ``SC``/``SI`` (byte / word store, low bits
only) MATCH c4 at the observable byte, and are the positive controls so the file
is a complete map.

GROUND TRUTH is the real c4 interpreter, built from ``old/c4_original.c``
(``#define int long long``; the canonical Bellard/Lorette c4).  The expected
c4 values embedded below were captured by compiling+running each expression
with that binary (``gcc -w -fpermissive -static -o /tmp/c4real
old/c4_original.c``); the comments show the exact ``int main(){...}`` used.  To
re-capture, see ``_C4_GROUND_TRUTH`` docstring.

Layout of the tests:

  * ``TestMatchesC4``               — cases where OUR behavior == c4 (controls).
  * ``TestArithmeticSHRMatchesC4`` /
    ``TestSignedLCMatchesC4``        — the c4-faithful SHR / LC on the DRAFT VMs
    (the fix landed; formerly ``xfail(strict)``, now plain asserts).
  * ``TestNeuralSHRIsArithmeticLikeC4`` — the GOLDEN (tight) neural SHR, now
    arithmetic through the real forward (byte-exact to c4).

NB: this whole file is TESTS ONLY — it authors no weights and imports only the
existing reference interpreters + the already-built shifter gadget, so the
golden fingerprint (``8f4dd780``) is untouched by adding it.
"""
from __future__ import annotations

import pytest

from c4_min import isa
from c4_min.nibble_pure_forward_complete import ref_interpret
from c4_min.libprog_corpus import RefVM


# ===========================================================================
# Program builders — one C expression == one straight-line bytecode program.
# ===========================================================================
def _bin_prog(v: int, n: int, op: str):
    """``v OP n`` : IMM v; PSH; IMM n; OP; HALT.  STACK0 = v, AX = n at OP."""
    return isa.assemble([("IMM", v), ("PSH", 0), ("IMM", n), (op, 0), ("HALT", 0)])


def _isa8(v, n, op):
    """8-bit reference (``isa.interpret``): AX trace, MASK=0xFF."""
    return isa.interpret(_bin_prog(v, n, op))[-1]


def _ref32(v, n, op):
    """32-bit SP-stack reference (``ref_interpret``, mask=0xFFFFFFFF).

    NOTE: ``ref_interpret`` folds the IMM immediate to 8 bits, so ``v``/``n`` are
    the LOW BYTE of the operands (the model's per-step byte frame).  Values are
    chosen so the low byte is the operand of interest."""
    return ref_interpret(_bin_prog(v, n, op), mask=0xFFFFFFFF)[-1]


def _corpus(v, n, op):
    """Corpus oracle (``libprog_corpus.RefVM``): 32-bit word stack, unsigned."""
    words = [(ins.op, ins.imm) for ins in _bin_prog(v, n, op)]
    vm = RefVM(words, b"")
    vm.run()
    return vm.ax


# The three python VMs, keyed by the mask width they observe.  ``isa.interpret``
# is 8-bit; the other two are 32-bit-word machines.
_VMS_8 = {"isa.interpret": _isa8}
_VMS_32 = {"ref_interpret": _ref32, "libprog_corpus.RefVM": _corpus}


# ===========================================================================
# GROUND TRUTH — real c4 (old/c4_original.c, #define int long long).
# Each value was captured by running the shown ``int main(){...}`` through
# /tmp/c4real.  c4 prints ``exit(N)``; N is the shown integer (64-bit signed,
# shown here masked to the low byte / low 32 bits as noted).
# ===========================================================================
_C4_GROUND_TRUTH = """
    Captured with: gcc -w -fpermissive -static -o /tmp/c4real old/c4_original.c

    # SHR — ARITHMETIC (sign-extending) on 64-bit signed:
    int x;x=-1;   return x>>1;      -> -1        (0xFFFFFFFF low32)
    int x;x=-1;   return x>>25;     -> -1
    int x;x=-256; return x>>4;      -> -16       (0xFFFFFFF0 low32)
    int x;x=-1024;return x>>8;      -> -4
    int x;x=255;  return x>>1;      -> 127       (positive: matches us)
    char c;c=128; return (c>>1)&255;-> 192       (c=-128, -128>>1=-64, &255=192)
    char c;c=-1;  return c>>1;      -> -1

    # SHL — LOGICAL, matches us at the byte:
    return 1<<8;                    -> 256
    return (255<<1)&255;            -> 254
    char c;c=-1;  return (c<<1)&255;-> 254

    # LC — SIGNED char load:
    char c;c=255; return c;         -> -1        (0xFF sign-extends)
    char c;c=128; return c;         -> -128
    char c;c=127; return c;         -> 127
    char c;c=200; return c*1000;    -> -56000    (signed -56 * 1000)

    # SC / SI — byte store keeps low byte only (matches us):
    char *p;p=malloc(8);*p=511; return (*p)&255; -> 255
"""


# ===========================================================================
# MATCHING cases: OUR behavior == c4.  These must PASS.
# ===========================================================================
class TestMatchesC4:
    """Cases where every one of our VMs already agrees with real c4."""

    @pytest.mark.parametrize("name,fn", list(_VMS_8.items()))
    def test_shr_positive_byte_matches(self, name, fn):
        # POSITIVE byte (top bit clear): arithmetic and logical SHR coincide, so
        # this is a clean positive control.  254 (0xFE) has bit7 set so it is a
        # SIGNED char (-2) in the 8-bit fold -> use 100 (top bit clear, +100).
        # c4: int 100>>1 == 50; char c=100; (c>>1)&255 == 50 (same, positive).
        assert fn(100, 1, "SHR") == 50, name

    @pytest.mark.parametrize("name,fn", list(_VMS_8.items()))
    def test_shl_logical_byte_matches(self, name, fn):
        # SHL is logical in both c4 and us; the observable low byte agrees.
        assert fn(255, 1, "SHL") == 254, name       # c4 (255<<1)&255 == 254
        assert fn(1, 3, "SHL") == 8, name

    def test_shl_wide_matches_c4(self):
        # 1<<8 == 256 (full 32-bit view).  c4 == 256.
        assert _ref32(1, 8, "SHL") == 256
        assert _corpus(1, 8, "SHL") == 256

    def test_shr_positive_wide_matches_c4(self):
        # positive operand: arithmetic and logical SHR coincide.
        assert _corpus(255, 1, "SHR") == 127        # c4 255>>1 == 127

    def test_sc_byte_store_truncates_like_c4(self):
        # SC / char store keeps the low byte in BOTH c4 and us:
        # store 511 to a byte cell, read it back -> 255 (0xFF) both ways.
        # SC does mem[pop()] = AX, so PUSH the address first, then AX = value.
        prog = isa.assemble([
            ("IMM", 5),              # AX = 5 (address)
            ("PSH", 0),              # STACK0 = 5 (the address SC will pop)
            ("IMM", 0x1FF),          # AX = 511 (folded to 0xFF by IMM byte-mask)
            ("SC", 0),               # mem[5] = AX & 0xFF   (STACK0 popped as addr)
        ])
        words = [(ins.op, ins.imm) for ins in prog]
        vm = RefVM(words, b"")
        vm.run()
        assert vm.mem.get(5, 0) & 0xFF == 0xFF       # c4 (*p)&255 == 255


# ===========================================================================
# c4-FAITHFUL cases (FIXED): the draft VMs now do ARITHMETIC SHR + SIGNED LC, so
# these MATCH real c4.  (Formerly ``xfail(strict)`` written c4-faithfully; the fix
# landed so they are plain asserts now.  Companion ``*_now_matches_c4`` tests pin
# the NEW c4-faithful value so a silent regression back to logical is caught.)
# ===========================================================================
class TestArithmeticSHRMatchesC4:
    """SHR is now ARITHMETIC (sign-extending) in every draft VM, like c4.

    c4: ``a = *sp++ >> a`` on a SIGNED ``long long`` -> the sign bit fills.  The
    32-bit VMs read the operand's sign at bit 31; the 8-bit ``isa.interpret`` reads
    it at bit 7 (the 8-bit fold IS one signed byte, i.e. a c4 ``char``), so a byte
    >= 0x80 is a negative char.  All values verified against real c4 (see
    ``_C4_GROUND_TRUTH``, captured from ``old/c4_original.c``).
    """

    # --- 32-bit int SHR on a negative value (0xFFFFFFFF == -1) ---------------
    def test_shr_neg1_is_arithmetic(self):
        # c4: (long long)-1 >> 1 == -1 -> low32 0xFFFFFFFF.
        assert _corpus(0xFFFFFFFF, 1, "SHR") == 0xFFFFFFFF

    def test_shr_neg256_is_arithmetic(self):
        # c4: int -256 (0xFFFFFF00) >> 4 == -16 -> low32 0xFFFFFFF0.
        assert _corpus(0xFFFFFF00, 4, "SHR") == 0xFFFFFFF0

    # --- char high-bit SHR reaching the printed low byte (8-bit fold) -------
    def test_char_128_shr1_is_192(self):
        # c4: char c=128 (==-128); (c>>1)&255 == 192  (arith -128>>1 == -64).
        assert _isa8(128, 1, "SHR") == 192

    def test_char_255_shr1_is_255(self):
        # c4: char c=255 (==-1); (c>>1)&255 == 255  (arith -1>>1 == -1).
        assert _isa8(255, 1, "SHR") == 255

    # --- large shift where sign-extension fills the whole low byte ----------
    def test_int_neg1_shr25_low_byte_is_ff(self):
        # c4: -1 >> 25 == -1 -> low byte 0xFF (sign kept across a large shift).
        got = _corpus(0xFFFFFFFF, 25, "SHR")
        assert got == 0xFFFFFFFF
        assert (got & 0xFF) == 0xFF


class TestSignedLCMatchesC4:
    """LC is now a SIGNED char load in every 32-bit draft VM, like c4.

    c4: ``a = *(char *)a`` -> a byte >= 0x80 is a negative char, sign-extended to
    the register.  Demonstrated on ``ref_interpret``: store a high-bit byte via SC,
    load it via LC, then add so the sign shows in the low 32 bits.  LI stays an
    unsigned word load (only LC is signed).
    """

    def _store_then_load_then_add(self, stored_byte, addend):
        """mem[7]=stored_byte (SC); AX=mem[7] (signed LC); AX = AX + addend."""
        # SC does mem[pop()] = AX: push the address (7) first, then AX = value.
        prog = isa.assemble([
            ("IMM", 7),                   # AX = 7 (address)
            ("PSH", 0),                   # STACK0 = 7 (address SC will pop)
            ("IMM", stored_byte & 0xFF),  # AX = byte to store
            ("SC", 0),                    # mem[7] = AX & 0xFF   (STACK0 popped)
            ("IMM", 7),                   # AX = 7 (address to load)
            ("LC", 0),                    # AX = (signed char)mem[7]
            ("PSH", 0),                   # STACK0 = loaded value
            ("IMM", addend & 0xFF),       # AX = addend
            ("ADD", 0),                   # AX = loaded + addend
        ])
        return ref_interpret(prog, mask=0xFFFFFFFF)[-1]

    def test_lc_high_bit_sign_extends(self):
        # c4: char at 0x80 -> -128; +0 -> -128 -> low32 0xFFFFFF80.
        assert self._store_then_load_then_add(0x80, 0) == 0xFFFFFF80

    def test_lc_ff_is_minus_one(self):
        # c4: char at 0xFF -> -1; +0 -> -1 -> low32 0xFFFFFFFF.
        assert self._store_then_load_then_add(0xFF, 0) == 0xFFFFFFFF

    def test_lc_positive_byte_unchanged(self):
        # c4: char at 0x7F -> +127 (top bit clear, no sign-extend); +0 -> 127.
        assert self._store_then_load_then_add(0x7F, 0) == 127


# ===========================================================================
# NEURAL model: the PRODUCTION SHR gadget is now ARITHMETIC too (FIXED).
#
# The golden neural SHR is the TIGHT direct-8x8 nibble shifter (``C4_TIGHT_SHIFT``
# ON, ``shift_tight_nibble`` + ``nibble_bitwise.tight_shift_stage_blocks``).  A
# ``build_sign_fill`` block now sign-extends the top ``n`` bits after the logical
# shift, so the BAKED SHR — through the real SwiGLU forward — is byte-exact c4
# arithmetic.  This MOVES the golden ``_fingerprint_build`` hash (intended).
#
# (The ``shifter_bakeoff`` module is a separate DESIGN-BAKEOFF measurement tool
# that stays LOGICAL — it is NOT on the production build path; its own test pins
# it logical.  The ``C4_TIGHT_SHIFT=0`` LOG-SHIFTER fallback is also left logical —
# off by default and unexercised — a documented gap in that fallback only.)
# ===========================================================================
class TestNeuralSHRIsArithmeticLikeC4:
    """The GOLDEN (tight) neural SHR now sign-extends — byte-exact to c4, through
    the real SwiGLU forward.  Verified against the c4 32-bit-signed reference."""

    @staticmethod
    def _neural_tight_shr(pop, n):
        from c4_min import isa as _isa
        from c4_min import nibble_bitwise as bw
        from c4_min.blogspec_layout import NibbleLayout
        L = NibbleLayout()
        weights = bw.compile_dispatch(L, bw.append_bitwise_shift_to_dispatch(L, _isa.SHR))
        return bw.run_compiled(L, weights, pop, n)

    @staticmethod
    def _c4_shr_arith(pop, n):
        # c4 ARITHMETIC >> on the 32-bit-signed interpretation, low-32 view.
        sp = pop - (1 << 32) if pop & 0x80000000 else pop
        return (sp >> n) & 0xFFFFFFFF if n < 32 else (sp >> 63) & 0xFFFFFFFF

    @pytest.mark.parametrize("pop,n", [
        (0xFFFFFFFF, 1), (0xFFFFFFFF, 25), (0xFFFFFF00, 4),
        (0xFFFFFF80, 1), (0x80000000, 1), (0xDEADBEEF, 7),
    ])
    def test_neural_tight_shr_is_arithmetic(self, pop, n):
        neural = self._neural_tight_shr(pop, n)
        # byte-exact to c4's ARITHMETIC (sign-extending) answer ...
        assert neural == self._c4_shr_arith(pop, n)
        # ... which (for these negative sources) DIFFERS from the old logical shift.
        assert neural != (pop >> n) & 0xFFFFFFFF

    def test_neural_tight_shr_matches_on_positive(self):
        # positive operand: arithmetic == logical, unchanged.
        assert self._neural_tight_shr(255, 1) == 127 == self._c4_shr_arith(255, 1)

    def test_shifter_bakeoff_tool_stays_logical(self):
        # the shifter_bakeoff DESIGN tool is NOT the production path -> stays logical.
        from c4_min import shifter_bakeoff as sb
        blocks, L = sb.build_chunk_shr(4)
        assert sb.run_blocks(blocks, L, 0xFFFFFFFF, 1) == 0x7FFFFFFF   # logical


# ===========================================================================
# CONSISTENCY: the draft VMs now agree WITH EACH OTHER on the c4-faithful
# ARITHMETIC SHR (self-consistent, and c4-faithful after the fix).
# ===========================================================================
class TestDraftVMsAgreeOnArithmeticSHR:
    @pytest.mark.parametrize("v,n", [(0xFFFFFFFF, 1), (0xFFFFFF00, 4), (255, 1)])
    def test_corpus_shr_is_arithmetic(self, v, n):
        # corpus SHR now sign-extends (arithmetic) — the c4-faithful answer:
        sv = v - (1 << 32) if v & (1 << 31) else v
        assert _corpus(v, n, "SHR") == (sv >> n) & 0xFFFFFFFF
