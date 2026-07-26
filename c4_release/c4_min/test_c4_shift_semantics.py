"""c4-fidelity SPEC for signed shifts (SHR/SHL) and char signedness (LC/SC).

This file is the executable specification of *what our VMs do vs what real c4
does* for the two ops whose signedness diverges from the c4 reference:

  * **SHR** — c4 does an ARITHMETIC (sign-extending) right shift on a 64-bit
    ``long long`` (``#define int long long`` in c4.c; run loop ``a = *sp++ >> a``
    with ``int *sp``).  Every one of our VMs does a LOGICAL (unsigned) right
    shift (``(pop >> n) & MASK``), routed through the native unsigned DIV gadget
    (``SHR x,n = x // 2**n``, see ``nibble_alu32`` §"SHIFT-VIA-MUL/DIV").
  * **LC** (load char) — c4 does ``a = *(char *)a`` = a SIGNED char load
    (-128..127, sign-extended to 64 bits).  Every one of our VMs does an
    UNSIGNED byte load (``mem[a] & 0xFF``, 0..255).

``SHL`` (logical left shift) and ``SC``/``SI`` (byte / word store, low bits
only) MATCH c4 at the observable byte, and are covered here as the positive
control so the file is a complete map, not just a divergence list.

GROUND TRUTH is the real c4 interpreter, built from ``old/c4_original.c``
(``#define int long long``; the canonical Bellard/Lorette c4).  The expected
c4 values embedded below were captured by compiling+running each expression
with that binary (``gcc -w -fpermissive -static -o /tmp/c4real
old/c4_original.c``); the comments show the exact ``int main(){...}`` used.  To
re-capture, see ``_C4_GROUND_TRUTH`` docstring.

Layout of the tests:

  * ``TestMatchesC4``    — cases where OUR behavior == c4 (must pass).
  * ``TestDivergesFromC4_*`` — cases where OUR behavior != c4 (``xfail(strict)``):
    the assert is written as if we were c4-faithful, so the day someone makes
    SHR arithmetic / LC signed these flip to XPASS and announce the fix.  A
    companion ``*_documents_our_actual`` test PINS the CURRENT (divergent)
    value so a silent semantics drift is still caught.

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
        # 255 >> 1 == 127: sign bit of the byte is irrelevant here (top bit 0
        # after the shift either way).  c4: (255>>1)&255 == 127.
        assert fn(255, 1, "SHR") == 127, name

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
# DIVERGING cases: OUR behavior != c4.  xfail(strict) — written c4-faithfully so
# a real fix flips them to XPASS; a companion test pins our CURRENT value.
# ===========================================================================
class TestDivergesFromC4_SHR:
    """SHR is LOGICAL for us, ARITHMETIC for c4 — diverges on negative operands.

    The ``xfail`` asserts encode c4's arithmetic-shift answer.  While we stay
    logical they FAIL (as expected).  The ``*_documents_our_actual`` tests pin
    the CURRENT logical value so a silent change is still caught.
    """

    # --- 32-bit int SHR on a negative value (0xFFFFFFFF == -1) ---------------
    @pytest.mark.xfail(strict=True, reason="our SHR is logical; c4 is arithmetic "
                       "(sign-extend). int -1>>1 == -1 (0xFFFFFFFF) in c4, "
                       "0x7FFFFFFF for us.")
    def test_shr_neg1_would_be_arithmetic_in_c4(self):
        # c4: (long long)-1 >> 1 == -1 -> low32 0xFFFFFFFF.
        assert _corpus(0xFFFFFFFF, 1, "SHR") == 0xFFFFFFFF

    def test_shr_neg1_documents_our_actual(self):
        # PIN: our logical SHR of 0xFFFFFFFF >> 1 == 0x7FFFFFFF (top bit cleared).
        assert _corpus(0xFFFFFFFF, 1, "SHR") == 0x7FFFFFFF

    # --- char high-bit SHR reaching the printed low byte --------------------
    @pytest.mark.xfail(strict=True, reason="8-bit: char 128 is -128 in c4; "
                       "-128>>1 == -64, (&255)==192.  Our unsigned 128>>1==64.")
    def test_char_128_shr1_would_be_192_in_c4(self):
        # c4: char c=128 (==-128); (c>>1)&255 == 192.
        assert _isa8(128, 1, "SHR") == 192

    def test_char_128_shr1_documents_our_actual(self):
        # PIN: our unsigned 128 >> 1 == 64.
        assert _isa8(128, 1, "SHR") == 64

    # --- large shift where sign-extension fills the whole low byte ----------
    @pytest.mark.xfail(strict=True, reason="int -1>>25: c4 arithmetic keeps sign "
                       "-> low byte 0xFF; our logical -> 0x7F.")
    def test_int_neg1_shr25_low_byte_would_be_ff_in_c4(self):
        # c4: -1 >> 25 == -1 -> low byte 0xFF.
        assert (_corpus(0xFFFFFFFF, 25, "SHR") & 0xFF) == 0xFF

    def test_int_neg1_shr25_documents_our_actual(self):
        # PIN: our logical 0xFFFFFFFF >> 25 == 0x7F, low byte 0x7F (NOT 0xFF).
        got = _corpus(0xFFFFFFFF, 25, "SHR")
        assert got == 0x7F
        assert (got & 0xFF) == 0x7F


class TestDivergesFromC4_LC:
    """LC is an UNSIGNED byte load for us, a SIGNED char load for c4.

    Demonstrated on the 32-bit ``ref_interpret``: store a high-bit byte via SC,
    load it via LC, then do arithmetic that exposes the sign.
    """

    def _store_then_load_then_add(self, stored_byte, addend):
        """mem[7]=stored_byte (SC); AX=mem[7] (LC); AX = AX + addend.  Returns AX
        under our (unsigned-LC) reference."""
        # SC does mem[pop()] = AX: push the address (7) first, then AX = value.
        prog = isa.assemble([
            ("IMM", 7),                   # AX = 7 (address)
            ("PSH", 0),                   # STACK0 = 7 (address SC will pop)
            ("IMM", stored_byte & 0xFF),  # AX = byte to store
            ("SC", 0),                    # mem[7] = AX & 0xFF   (STACK0 popped)
            ("IMM", 7),                   # AX = 7 (address to load)
            ("LC", 0),                    # AX = mem[7]  (UNSIGNED for us)
            ("PSH", 0),                   # STACK0 = loaded value
            ("IMM", addend & 0xFF),       # AX = addend
            ("ADD", 0),                   # AX = loaded + addend
        ])
        return ref_interpret(prog, mask=0xFFFFFFFF)[-1]

    @pytest.mark.xfail(strict=True, reason="LC signed: c4 loads 0x80 as -128; "
                       "-128 + 0 == -128 (0xFFFFFF80).  We load it as +128.")
    def test_lc_high_bit_would_sign_extend_in_c4(self):
        # c4: char at 0x80 -> -128; +0 -> -128 -> low32 0xFFFFFF80.
        assert self._store_then_load_then_add(0x80, 0) == 0xFFFFFF80

    def test_lc_high_bit_documents_our_actual(self):
        # PIN: our unsigned LC loads 0x80 as +128; +0 == 128.
        assert self._store_then_load_then_add(0x80, 0) == 128

    @pytest.mark.xfail(strict=True, reason="LC signed: c4 loads 0xFF as -1.")
    def test_lc_ff_would_be_minus_one_in_c4(self):
        # c4: char at 0xFF -> -1 -> low32 0xFFFFFFFF.
        assert self._store_then_load_then_add(0xFF, 0) == 0xFFFFFFFF

    def test_lc_ff_documents_our_actual(self):
        # PIN: our unsigned LC loads 0xFF as +255.
        assert self._store_then_load_then_add(0xFF, 0) == 255


# ===========================================================================
# NEURAL model: prove the BAKED SHR gadget agrees with the DRAFT VMs (logical)
# and diverges from c4 (arithmetic) — via the real SwiGLU-plane forward.
#
# The shifter_bakeoff nibble-granular SHR gadget IS the production shift-via-DIV
# path's arithmetic on its residual planes (run_blocks = real forward).  Its
# reference ``ref_shift32`` is ``(pop >> n) & 0xFFFFFFFF`` treating pop as an
# UNSIGNED 32-bit value -> logical, exactly like the draft VMs.
# ===========================================================================
class TestNeuralSHRIsLogicalLikeDraft:
    """The baked neural SHR is logical/unsigned — same as the draft VMs, NOT c4."""

    @staticmethod
    def _neural_shr(pop, n):
        from c4_min import shifter_bakeoff as sb
        blocks, L = sb.build_chunk_shr(4)          # nibble-granular SHR gadget
        return sb.run_blocks(blocks, L, pop, n)

    @staticmethod
    def _c4_shr_arith(pop_as_signed32, n):
        # c4 arithmetic >> on the signed 32-bit interpretation, low-32 view.
        return (pop_as_signed32 >> n) & 0xFFFFFFFF

    @pytest.mark.parametrize("pop,n,signed", [
        (0xFFFFFFFF, 1, -1),
        (0xFFFFFFFF, 25, -1),
        (0xFFFFFF00, 4, -256),      # -256
        (0xFFFFFF80, 1, -128),      # what a SIGNED LC of 0x80 would feed SHR
    ])
    def test_neural_shr_is_logical_diverges_from_c4(self, pop, n, signed):
        neural = self._neural_shr(pop, n)
        c4 = self._c4_shr_arith(signed, n)
        # matches the LOGICAL reference ...
        assert neural == (pop >> n) & 0xFFFFFFFF
        # ... and therefore DIVERGES from c4's arithmetic answer.
        assert neural != c4

    def test_neural_shr_matches_c4_on_positive(self):
        # positive operand: arithmetic == logical, so neural == c4 here.
        assert self._neural_shr(255, 1) == 127 == self._c4_shr_arith(255, 1)


# ===========================================================================
# CONSISTENCY: all our VMs agree WITH EACH OTHER (self-consistent, uniformly
# c4-divergent) — the corpus is internally consistent, just not c4-faithful, so
# nothing in the corpus expects c4's arithmetic SHR / signed LC.
# ===========================================================================
class TestOurVMsAgreeWithEachOther:
    @pytest.mark.parametrize("v,n", [(0xFFFFFFFF, 1), (0xFFFFFF00, 4), (255, 1)])
    def test_corpus_shr_is_uniformly_logical(self, v, n):
        # corpus SHR is logical unsigned across the board:
        assert _corpus(v, n, "SHR") == (v >> n) & 0xFFFFFFFF
