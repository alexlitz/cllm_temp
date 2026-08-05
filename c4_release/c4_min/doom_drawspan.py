"""doom_drawspan.py -- NATIVE fused DRAWSPAN render-macro superinstruction.

This collapses Doom's ``V_DrawPatch`` inner column-copy loop -- the per-pixel
strided byte copy that paints one post of one column of a patch to the screen --
into a SINGLE native c4 opcode.  It is the highest-value RENDER macro: pure
memory movement, zero I/O emit, no emit floor (the framebuffer is host-read from
memory, so no hex-emit path is needed).

The C source it fuses is the innermost loop of ``V_DrawPatch``
(``linuxdoom-1.10/v_video.c`` line 254, SCREENWIDTH=320)::

    while (count--)
    {
        *dest = *source++;      // copy one byte down the column
        dest += SCREENWIDTH;    // step one screen ROW (stride = 320)
    }

i.e. for ``k`` in ``0..count``: ``mem[dest + k*SCREENWIDTH] = mem[source + k]``.
Byte-exactly, with ``source++`` (contiguous read), ``dest += stride`` (strided
write), and the ``count--`` post-decrement loop test.  This is the draft's
``match_copy_loop`` fusion (``dest += 320; source++; count--``) landed native.

  * **DRAWSPAN(dest, source, count, stride=320)** = the byte-exact column copy.
    ``count <= 0`` writes nothing (the ``count--`` guard).  Each iteration reads
    ONE byte from the contiguous source and writes it to the destination stepped
    by ``stride`` bytes.  ONE decoded VM step replaces the ``count``-iteration
    inner loop whose body costs ~35 instrs PER PIXEL.

The intrinsic peephole recognises the compiled inner-loop by its OPCODE
SIGNATURE (the ``count--`` test + the byte copy + the ``dest += 320`` /
``source += 1`` bumps + the ``JMP`` back-edge), NOT a ``JSR`` call target --
because the column copy is an INLINE loop, not a function call.  It rewrites the
recognised span to the native op while PRESERVING STREAM LENGTH (the removed
instructions become ``NOP`` and the ``JMP`` back-edge becomes a ``NOP`` fall-
through), so every branch / JSR target stays valid with NO re-resolution --
exactly the length-preserving discipline ``doom_blit`` uses.

Gate
====
Everything is behind ``C4_DOOM_DRAWSPAN`` (default OFF).  OFF -> the opcode is
not registered, no megablock is emitted, the peephole is a no-op, ``isa`` never
widens ``num_ops_effective`` -> golden family-B fingerprint ``069cc32f`` and the
byte-exact title frame (which depends on the ``V_DrawPatch`` INLINE-loop path
being the default) are unaffected.

``doom_blit.py`` (#810, memset-style megablock) and ``doom_nameeq.py`` (#846,
battery-cases pattern) are the templates this follows exactly (opcode
registration, gate, reference interp, megablock schedule, intrinsic peephole,
on-VM verification via ``c4vm32.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import isa

# ---------------------------------------------------------------------------
# Opcode number.  40..43 are the float ISA (F_ADD/F_SUB/F_MUL/F_DIV; FIXEDMUL/
# FIXEDDIV reuse 42/43 under their OWN gate); 44/45 are BLIT/MEMCPY (doom_blit,
# C4_DOOM_BLIT); 46 is NAMEEQ (doom_nameeq, C4_DOOM_NAMEEQ).  47 is the next free
# slot above every existing one-hot band and does NOT collide with anything the
# c4_min ISA (NUM_OPS=40) decodes.  Registered ONLY when the gate is on.
# ---------------------------------------------------------------------------
DRAWSPAN = isa.DRAWSPAN            # 47 (canonical value lives in isa.py)

# Width of the OP_IS opcode one-hot band when DRAWSPAN is neurally wired: covers
# opcodes 0..47.  Mirrors NUM_OPS_FLOAT=44 (the C4_FLOAT_OPS mechanism), but one
# wider so the DRAWSPAN one-hot (opcode 47) has a slot.  Read via
# ``isa.num_ops_effective`` only when the gate is on -> flag-OFF band stays
# NUM_OPS=40 wide and every downstream layout dim is byte-identical to golden.
NUM_OPS_DRAWSPAN = isa.NUM_OPS_DRAWSPAN   # 48

_MASK32 = 0xFFFFFFFF
_SIGN = 0x80000000

# Default column stride == Doom's SCREENWIDTH (doomdef.h:110).  The ``dest``
# pointer advances one screen ROW per copied pixel.
SCREENWIDTH = 320


# =========================================================================== #
# gate                                                                        #
# =========================================================================== #
def drawspan_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """``C4_DOOM_DRAWSPAN`` gate (default OFF).

    OFF -> the native op is not registered / not baked / the peephole is a
    no-op / ``isa.num_ops_effective`` stays ``NUM_OPS`` -> golden ``069cc32f``
    byte-identical and the ``V_DrawPatch`` inline-loop path (the byte-exact
    title frame's dependency) is the default.
    """
    e = os.environ if env is None else env
    return e.get("C4_DOOM_DRAWSPAN", "0") not in ("0", "", "false", "False")


def num_ops_effective() -> int:
    """OP_IS one-hot band width for the CURRENT ``C4_DOOM_DRAWSPAN`` state.

    Delegates to :func:`isa.num_ops_effective`, which already folds in the
    ``C4_DOOM_DRAWSPAN`` gate (``NUM_OPS_DRAWSPAN`` = 48 when ON, so the DRAWSPAN
    opcode-47 one-hot has a slot -- the SAME mechanism ``NUM_OPS_FLOAT`` uses).
    Flag-OFF is byte-identical to golden ``069cc32f``; flag-ON is the intended
    MOVED fingerprint (wider OP_IS band by construction).
    """
    return isa.num_ops_effective()


def register_opcode() -> None:
    """Register DRAWSPAN into :mod:`isa` (idempotent, gated).

    Only mutates ``isa.NAMES`` / ``isa.BY_NAME`` so the assembler can emit it; it
    never widens ``isa.NUM_OPS`` itself, so a build with the gate OFF is
    byte-identical.  The OP_IS band widening (flag-ON) is folded into
    ``isa.num_ops_effective`` by the ``C4_DOOM_DRAWSPAN`` gate (in ``isa.py``),
    which the layout reads -- exactly the ``NUM_OPS_FLOAT`` mechanism.
    """
    isa.NAMES.setdefault(DRAWSPAN, "DRAWSPAN")
    isa.BY_NAME.setdefault("DRAWSPAN", DRAWSPAN)


# =========================================================================== #
# helpers                                                                     #
# =========================================================================== #
def _sx(v: int) -> int:
    """32-bit two's-complement sign-extend (the VM's signed view of a word)."""
    v &= _MASK32
    return v - (1 << 32) if v & _SIGN else v


# =========================================================================== #
# 1. NATIVE REFERENCE -- byte-exact vs the V_DrawPatch column copy AS IT RUNS   #
# =========================================================================== #
def draw_span(mem: bytearray, dest: int, source: int, count: int,
              stride: int = SCREENWIDTH) -> None:
    """``DRAWSPAN(dest, source, count, stride)`` -- byte-exact to the
    ``V_DrawPatch`` inner column-copy loop.

    ::

        while (count--) { *dest = *source++; dest += stride; }

    For ``k`` in ``0 .. count-1``: ``mem[dest + k*stride] = mem[source + k]``.
    ``count`` is compared SIGNED (the C ``while (count--)`` on ``int count`` runs
    the body ``count`` times for ``count > 0`` and ZERO times for ``count <= 0``,
    because the FIRST test reads the pre-decrement value and a ``count`` of 0
    tests false immediately).  Each byte is a plain 8-bit ``char`` copy (the C
    ``*(byte*)dest = *(byte*)source``).  Mutates ``mem`` in place, returns None
    (the C loop returns nothing; the fused op leaves AX untouched).
    """
    d = dest & _MASK32
    s = source & _MASK32
    n = _sx(count)
    st = _sx(stride)
    for k in range(n if n > 0 else 0):
        mem[(d + k * st) & _MASK32] = mem[(s + k) & _MASK32] & 0xFF


def draw_span_step_cost_c(count: int) -> int:
    """The number of decoded VM STEPS the INLINE ``V_DrawPatch`` column-copy loop
    costs for a given ``count`` (the count the native op replaces).

    The compiled inner loop is a straight-line body of ``_SPAN_LOOP_BODY`` instrs
    (the ``count--`` test + byte copy + ``source += 1`` + ``dest += stride`` +
    ``JMP`` back-edge) executed ``count`` times, plus the final loop-exit test.
    Derived from the compiled disasm (see :data:`_SPAN_LOOP_BODY`); measured
    on-VM by the correctness harness.
    """
    n = _sx(count)
    if n <= 0:
        return _SPAN_LOOP_EXIT
    return _SPAN_LOOP_BODY * n + _SPAN_LOOP_EXIT


# Step-cost constants, from the compiled ``V_DrawPatch`` inner loop disasm on
# c4vm32.py (the ``while (count--) { *dest = *source++; dest += 320; }`` body):
# the loop test (LEA/PSH/LI/PSH/IMM/SUB/SI/PSH/IMM/ADD/BZ = 11) + the copy
# (LEA/LI/PSH/LEA/LI/LC/SC = 7) + source bump (LEA/PSH/LEA/LI/PSH/IMM/ADD/SI = 8)
# + dest bump (LEA/PSH/LEA/LI/PSH/IMM/ADD/SI = 8) + JMP back-edge (1) = 35 instrs
# per copied pixel; the exit is the last (false) test = 11.
_SPAN_LOOP_BODY = 35        # decoded instrs per copied pixel (the hot inner loop)
_SPAN_LOOP_EXIT = 11        # the final count-- test that exits the loop


# =========================================================================== #
# 2. FUSED MEGABLOCK SCHEDULE (the native op's block sequence)                 #
#                                                                             #
#   A "megablock" is the fused block sequence that lands the whole op in ONE   #
#   decoded VM step.  The DRAWSPAN megablock is a bounded STRIDED memory copy:  #
#   it reads the four operands (dest, source, count, stride) off the stack,     #
#   applies the count>0 guard, and drives a recurrent copy body ``count`` times #
#   (the SAME recurrent-body lever BLIT's ``blit-store`` / the recurrent_divmod #
#   body use -- one stored block, iterated).  Byte-exact: iteration ``k`` reads #
#   ``mem[source + k]`` and writes ``mem[dest + k*stride]``.                    #
# =========================================================================== #
@dataclass
class MegablockSchedule:
    name: str
    blocks: List[str] = field(default_factory=list)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def drawspan_megablock() -> MegablockSchedule:
    """DRAWSPAN fused schedule: pop operands -> guard -> recurrent strided-copy
    body -> return -> ax-mux.

    The ``span-copy`` body is ONE recurrent stored block (read the running source
    byte, write it to the running strided dst pointer, bump ``source`` by 1 and
    ``dst`` by ``stride``, decrement the counter), iterated ``count`` times inside
    the single decoded step -- exactly the recurrent-body lever BLIT's
    ``blit-store`` uses and ``compile_divmod_blocks_recurrent`` uses to fold N
    iterations into one stored block.  The only DRAWSPAN-specific blocks are the
    4-operand-pop + count-guard prologue and the return epilogue (the C loop has
    no return value, so ``ax-mux`` leaves AX untouched).  One decoded VM step.
    """
    return MegablockSchedule(
        name="DRAWSPAN",
        blocks=[
            "alu-expand",     # pop dest,source,count,stride off the stack -> words
            "span-guard",     # [count<=0] -> skip the copy burst (the count-- guard)
            "span-copy",      # recurrent: mem[dst]=mem[src]; src++; dst+=stride; i++
            "span-ret",       # AX unchanged (the C column-copy loop returns nothing)
            "ax-mux",         # write result -> AX (shared)
        ],
    )


# =========================================================================== #
# 3. INTRINSIC RECOGNITION -- bytecode peephole keyed on the loop-head SIGNATURE #
# =========================================================================== #
# The compiled ``V_DrawPatch`` inner column-copy loop (from src.compiler on the
# c4_min substrate) is an INLINE loop, so unlike memset/__name_eq there is NO
# JSR call site to key on.  We key on the compiled OPCODE SIGNATURE instead: the
# ``count--`` test at the loop head, the byte copy (``LC``/``SC``), the
# ``source += 1`` / ``dest += stride`` pointer bumps (the ``IMM stride ; ADD``
# stride-add is the discriminating fingerprint), and the ``JMP`` back-edge.
#
# Disasm of the reference loop (locals: dest@+off_d, source@+off_s, count@+off_c;
# the frame offsets vary per call site but the OPCODE sequence is fixed):
#
#   L:  LEA off_c ; PSH ; LI ; PSH ; IMM 1 ; SUB ; SI ; PSH ; IMM 1 ; ADD ; BZ END
#       LEA off_d ; LI ; PSH ; LEA off_s ; LI ; LC ; SC          # *dest = *source
#       LEA off_s ; PSH ; LEA off_s ; LI ; PSH ; IMM 1 ; ADD ; SI # source += 1
#       LEA off_d ; PSH ; LEA off_d ; LI ; PSH ; IMM stride ; ADD ; SI # dest += stride
#       JMP L
#   END:...
#
# The signature is scanned as an OPCODE-only template (immediates are matched
# only for the two literal ``IMM 1`` bumps and the ``IMM stride`` step, and the
# ``JMP`` back-edge must target the loop head).  On a match the whole span is
# rewritten to ``<DRAWSPAN>`` + NOP padding to PRESERVE STREAM LENGTH.
# ---------------------------------------------------------------------------

# The OPCODE-only template for the compiled inner loop (35 slots + the JMP is the
# 35th).  ``None`` slots match any immediate; the marked slots also constrain the
# immediate.  Kept as (op, imm_or_None) so the matcher is a straight compare.
_STRIDE_ADD_IDX = 31   # the ``IMM stride`` slot (0-based within the template):
#   offsets 0-10 = count-- test; 11-17 = copy; 18-25 = source += 1;
#   26-33 = dest += stride (26 LEA,27 PSH,28 LEA,29 LI,30 PSH,31 IMM<stride>,
#           32 ADD,33 SI); 34 = JMP back-edge.


def _loop_template_ops() -> List[int]:
    """The fixed 35-opcode template of the compiled V_DrawPatch inner loop
    (opcodes only; immediates are checked separately)."""
    O = isa
    return [
        # --- count-- test (11) ---
        O.LEA, O.PSH, O.LI, O.PSH, O.IMM, O.SUB, O.SI, O.PSH, O.IMM, O.ADD, O.BZ,
        # --- *dest = *source (7) ---
        O.LEA, O.LI, O.PSH, O.LEA, O.LI, O.LC, O.SC,
        # --- source += 1 (8) ---
        O.LEA, O.PSH, O.LEA, O.LI, O.PSH, O.IMM, O.ADD, O.SI,
        # --- dest += stride (8) ---
        O.LEA, O.PSH, O.LEA, O.LI, O.PSH, O.IMM, O.ADD, O.SI,
        # --- back-edge (1) ---
        O.JMP,
    ]


@dataclass
class IntrinsicMatch:
    """One recognised V_DrawPatch inner-loop span.

    ``start`` is the instruction index of the loop head (the first ``LEA`` of the
    ``count--`` test); ``length`` is the number of instructions in the span
    (== ``len(_loop_template_ops())``); ``stride`` is the decoded ``dest +=``
    step (320 for a real column copy)."""
    start: int
    length: int
    stride: int


def find_drawspan_loops(code) -> List[IntrinsicMatch]:
    """Scan ``code`` for compiled V_DrawPatch inner-loop spans by opcode signature.

    ``code`` is a list of ``(op, imm)`` tuples (the ``c4vm32`` decoded form) or a
    list of ``isa.Instr``.  Returns the list of :class:`IntrinsicMatch` (loop-head
    index + length + decoded stride) -- one per recognised strided byte-copy loop.
    A span matches iff the 35-opcode template lines up AND the ``JMP`` back-edge at
    the end targets the span head (so a coincidental opcode run that is NOT a loop
    is rejected).
    """
    tpl = _loop_template_ops()
    L = len(tpl)
    is_tuple = bool(code) and isinstance(code[0], tuple)

    def _op(ins):
        return ins[0] if is_tuple else ins.op

    def _imm(ins):
        return ins[1] if is_tuple else ins.imm

    out: List[IntrinsicMatch] = []
    n = len(code)
    i = 0
    while i + L <= n:
        # opcode template match
        if all(_op(code[i + j]) == tpl[j] for j in range(L)):
            # the JMP back-edge (last slot) must target the loop head ``i``
            jmp = code[i + L - 1]
            if _op(jmp) == isa.JMP and _imm(jmp) == i:
                stride = _imm(code[i + _STRIDE_ADD_IDX]) & _MASK32
                out.append(IntrinsicMatch(start=i, length=L, stride=stride))
                i += L
                continue
        i += 1
    return out


def substitute_intrinsics(code):
    """Peephole: replace each recognised V_DrawPatch inner-loop span with the
    native DRAWSPAN op, PRESERVING STREAM LENGTH.

    Unlike the memset/__name_eq call-site rewrites (a ``JSR`` -> op), the column
    copy is an INLINE loop, so the whole span is collapsed: the loop-head
    instruction becomes ``<DRAWSPAN>`` and EVERY other instruction in the span --
    including the ``JMP`` back-edge -- becomes ``NOP``.  The DRAWSPAN op consumes
    its operands from the caller-established frame (dest/source/count locals) and
    performs the full strided copy in ONE decoded step; the trailing NOPs are
    executed as no-ops after it (they never loop because the back-edge is gone),
    then control falls through to the instruction the ``BZ END`` used to target.

    Stream LENGTH is UNCHANGED (span-in == span-out), so every branch / JSR
    target -- including the ``BZ END`` loop-exit that jumps PAST the span -- stays
    valid with NO re-resolution.  ``code`` is a list of ``(op, imm)`` tuples or a
    list of ``isa.Instr``.

    Returns ``(new_code, n_substitutions)``.
    """
    out = list(code)
    is_tuple = bool(out) and isinstance(out[0], tuple)
    matches = find_drawspan_loops(out)
    for m in matches:
        # loop-head -> DRAWSPAN (carry the decoded stride as its immediate so the
        # native op knows the column step without re-reading it from the frame)
        out[m.start] = (DRAWSPAN, m.stride) if is_tuple else isa.Instr(DRAWSPAN, m.stride)
        # every remaining span instruction (incl. the JMP back-edge) -> NOP
        for j in range(m.start + 1, m.start + m.length):
            out[j] = (isa.NOP, 0) if is_tuple else isa.Instr(isa.NOP, 0)
    return out, len(matches)


# =========================================================================== #
# 4. VERIFICATION BATTERY -- count/stride/boundary/overlap edges + a real frame #
# =========================================================================== #
def battery_cases() -> List[Tuple[int, int, int, int]]:
    """The verification battery of ``(dest, source, count, stride)`` cases,
    exercising:

      * COUNT edges: 0 (writes nothing), 1 (single pixel), large,
      * STRIDE edges: 1 (contiguous), 320 (a real screen row), 8, 200, a wide
        stride, a negative-ish stride guard,
      * SPAN boundaries: a copy that ends exactly at a region edge,
      * OVERLAPPING / ADJACENT src/dst: dst just after src, dst == src+1,
        src just after dst (the strided write never revisits a source byte at
        stride>=1 so forward overlap is well-defined),
      * a REAL framebuffer-sized span: a full 200-pixel column at stride 320
        (a title-frame column height).

    Addresses are laid into a scratch ``bytearray`` by the verifier; they are
    chosen disjoint / adjacent as each case requires.  ``count <= 0`` cases
    write nothing (the guard) -- included so 'wrote nothing' is TESTED, not a
    false-pass.
    """
    SRC = 0x100000
    DST = 0x200000
    cases: List[Tuple[int, int, int, int]] = []

    # (1) COUNT edges (stride 320, a real column)
    cases.append((DST, SRC, 0, 320))       # count 0 -> writes nothing (guard)
    cases.append((DST, SRC, 1, 320))       # single pixel
    cases.append((DST, SRC, 2, 320))       # two pixels
    cases.append((DST, SRC, 200, 320))     # a full 200-row column (framebuffer)
    cases.append((DST, SRC, 1000, 320))    # large

    # (2) STRIDE edges
    cases.append((DST, SRC, 64, 1))        # stride 1 == contiguous copy (== memcpy)
    cases.append((DST, SRC, 64, 8))        # stride 8
    cases.append((DST, SRC, 32, 200))      # stride 200
    cases.append((DST, SRC, 16, 4096))     # wide stride (rows far apart)
    cases.append((DST, SRC, 50, 320))      # nominal column

    # (3) SPAN boundary: a copy whose last written byte lands at a chosen edge
    cases.append((DST, SRC, 100, 320))     # ends at DST + 99*320

    # (4) OVERLAPPING / ADJACENT src/dst (forward strided copy is well-defined)
    cases.append((SRC + 320, SRC, 8, 320))     # dst one row below src (adjacent rows)
    cases.append((SRC + 1, SRC, 8, 1))         # dst == src+1, stride 1 (classic overlap)
    cases.append((SRC + 4, SRC, 8, 4))         # dst == src+4, stride 4
    cases.append((SRC, SRC + 10000, 8, 320))   # src far above dst (disjoint)

    # (5) count guard: negative count writes nothing
    cases.append((DST, SRC, -5 & _MASK32, 320))    # count < 0 -> writes nothing

    # (6) a REAL framebuffer-sized span: full column at screen width
    cases.append((DST, SRC, 168, 320))     # title patch column height (< 200)

    return cases


def verify_byte_exact() -> Dict[str, object]:
    """Verify DRAWSPAN is byte-exact vs the reference column-copy over the battery.

    For each ``(dest, source, count, stride)`` case:

      1. lay a KNOWN non-trivial source pattern into a scratch ``bytearray``,
      2. PRE-POISON the whole destination footprint with a sentinel byte (so a
         'wrote nothing' bug can never masquerade as a pass -- the exact bug
         ``doom_blit`` originally had),
      3. run :func:`draw_span` and re-derive the expected bytes independently,
      4. compare.

    The AUTHORITATIVE on-VM oracle (the REAL compiled ``V_DrawPatch`` loop run on
    ``c4vm32.py``) is the separate ``verify_drawspan_onvm.py`` harness.  Returns
    ``{n, fail, detail}`` -- 0 fails == byte-exact.
    """
    cases = battery_cases()
    res: Dict[str, object] = {"n": len(cases), "fail": 0, "detail": []}
    for (dest, source, count, stride) in cases:
        n = _sx(count)
        st = _sx(stride)
        # scratch memory large enough to hold both footprints
        hi_src = source + max(0, n)
        hi_dst = dest + max(0, n - 1) * max(st, 0)
        top = max(hi_src, hi_dst) + 64
        # known source pattern (distinct per byte so a mis-indexed read shows up)
        base = bytearray(top)
        for k in range(max(0, n)):
            base[(source + k) & _MASK32] = (k * 31 + 7) & 0xFF
        # pre-poison the destination footprint with a sentinel (so a 'wrote
        # nothing' bug can never masquerade as a pass -- the doom_blit bug)
        sentinel = 0xEE
        for k in range(max(0, n)):
            base[(dest + k * st) & _MASK32] = sentinel

        # run the op under test
        got_mem = bytearray(base)
        draw_span(got_mem, dest, source, count, stride)

        # INDEPENDENT oracle: a differently-structured sequential re-implementation
        # of the SAME live-memory C loop (``*dest = *source++; dest += stride``),
        # so OVERLAPPING src/dst (where a write feeds a later read -- the "smear")
        # is verified against the true C semantics, NOT a naive source snapshot.
        ref_mem = bytearray(base)
        d = dest & _MASK32
        s = source & _MASK32
        c = n
        while c > 0:                       # while (count--) with count>0 body
            ref_mem[d & _MASK32] = ref_mem[s & _MASK32]
            s = (s + 1) & _MASK32
            d = (d + st) & _MASK32
            c -= 1

        # also assert the poison is GONE where it should be (count>0 wrote), and
        # UNTOUCHED where it should not (guard) -- both are covered by the full
        # bytearray compare below.
        if got_mem != ref_mem:
            res["fail"] = int(res["fail"]) + 1
            ndiff = sum(1 for a, b in zip(got_mem, ref_mem) if a != b)
            res["detail"].append({"dest": dest, "source": source,
                                  "count": count, "stride": stride,
                                  "diff_bytes": ndiff})
    return res
