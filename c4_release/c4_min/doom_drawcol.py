"""doom_drawcol.py -- NATIVE fused GAMEPLAY render-macro superinstructions
DRAWCOL / DRAWSPANF (the 3D-view analogs of the title's DRAWSPAN).

The title's ``DRAWSPAN`` (``doom_drawspan.py``, #810) collapses ``V_DrawPatch``'s
per-pixel strided BYTE COPY -- the 2D page blit that paints the title screen.
This module collapses the two per-pixel TEXTURE-MAPPED FILL loops that dominate a
GAMEPLAY (3D-view) frame:

  * **DRAWCOL** (opcode 48) == ``R_DrawColumn`` -- a texture-mapped VERTICAL WALL
    column.  The C inner loop (``linuxdoom-1.10/r_draw.c`` line 138, and its
    c4-ported form ``doom_run.c`` line 28254)::

        do {
            *dest = dc_colormap[dc_source[(frac>>FRACBITS)&127]];  // FRACBITS=16
            dest += SCREENWIDTH;    // SCREENWIDTH=320  (strided down the column)
            frac += fracstep;
        } while (count--);

    i.e. a fixed-point DDA source-texel walk (``frac += fracstep``), a masked
    128-tall texture read (``dc_source[(frac>>16)&127]``), a colormap/light LUT
    remap (``dc_colormap[...]``), and a STRIDED byte write (``dest += 320``).
    The ``do..while(count--)`` runs the body ``count+1`` times for ``count >= 0``
    and ZERO times for ``count < 0`` (the ``R_DrawColumn`` early-return guard is
    ``if (count < 0) return;`` -- so the fused op mirrors the loop's post-test
    exactly: it runs ``max(count+1, 0)`` iterations).

  * **DRAWSPANF** (opcode 49) == ``R_DrawSpan`` -- a texture-mapped HORIZONTAL
    FLOOR/CEILING span.  The C inner loop (``r_draw.c`` line 549, ported form
    ``doom_run.c`` line 28503)::

        do {
            spot = ((yfrac>>(16-6))&(63*64)) + ((xfrac>>16)&63);
            *dest++ = ds_colormap[ds_source[spot]];  // CONTIGUOUS write
            xfrac += ds_xstep;
            yfrac += ds_ystep;
        } while (count--);

    i.e. a 2D ``(u,v)`` walk over a 64x64 flat tile (``spot`` folds ``yfrac`` and
    ``xfrac`` into a ``0..4095`` tile index), the same colormap LUT remap, and a
    CONTIGUOUS byte write (``*dest++``).  ``R_DrawSpan`` has NO ``count < 0``
    early return, so the body runs ``count+1`` times for ``count >= 0`` and the
    fused op mirrors that (``count`` is ``ds_x2 - ds_x1``, always ``>= 0`` for a
    real span; a negative count writes nothing under the post-test).

Both replace a ~28..40-instr-per-pixel inner loop with ONE decoded VM step, so a
gameplay frame's per-pixel fill (walls + floors) folds the same way the title's
``V_DrawPatch`` blit did (~69x for the title; ~3.44x for a complete gameplay
frame -> the ~0.29 fps gameplay run projects to ~1 fps).  Byte-exact: the native
op reproduces the loop's exact live-memory reads/writes, colormap indirection,
fixed-point wrap, and post-test iteration count.

Gate
====
Everything is behind ``C4_DOOM_DRAWCOL`` (default OFF).  OFF -> the opcodes are
not registered, no megablock is emitted, the peephole is a no-op, ``isa`` never
widens ``num_ops_effective`` past its non-DRAWCOL width -> golden family-B
fingerprint ``069cc32f`` is unaffected and the title ``V_DrawPatch``/DRAWSPAN
path is untouched (DRAWCOL/DRAWSPANF opcodes 48/49 sit ABOVE DRAWSPAN's 47).

``doom_drawspan.py`` (#810) is the template this follows exactly (opcode
registration, gate, reference interp, megablock schedule, intrinsic peephole,
on-VM verification via ``c4vm32.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import isa

# ---------------------------------------------------------------------------
# Opcode numbers.  47 is DRAWSPAN (title V_DrawPatch, doom_drawspan).  48/49 are
# the two GAMEPLAY texture-mapped fill macros; both sit ABOVE every existing
# one-hot band and above DRAWSPAN, so the baseline OP_IS layout (NUM_OPS=40) and
# the DRAWSPAN layout are UNCHANGED when this gate is off.  Registered ONLY when
# the gate is on.  Canonical values live in ``isa.py``.
# ---------------------------------------------------------------------------
DRAWCOL = isa.DRAWCOL              # 48 (R_DrawColumn -- vertical wall column)
DRAWSPANF = isa.DRAWSPANF          # 49 (R_DrawSpan -- horizontal floor span)

# Width of the OP_IS opcode one-hot band when DRAWCOL/DRAWSPANF are neurally
# wired: covers opcodes 0..49.  Read via ``isa.num_ops_effective`` only when the
# gate is on -> flag-OFF band stays NUM_OPS=40 (or DRAWSPAN's 48) wide.
NUM_OPS_DRAWCOL = isa.NUM_OPS_DRAWCOL   # 50

_MASK32 = 0xFFFFFFFF
_SIGN = 0x80000000

# Doom render constants (doomdef.h / m_fixed.h).
SCREENWIDTH = 320    # dest += SCREENWIDTH (the R_DrawColumn strided column step)
FRACBITS = 16        # fixed-point fractional bits (frac>>FRACBITS = whole texel)
TEXHEIGHT_MASK = 127  # dc_source[(frac>>16)&127] -- 128-tall wall texture column
# R_DrawSpan spot fold over a 64x64 flat tile:
#   spot = ((yfrac>>(16-6)) & (63*64)) + ((xfrac>>16) & 63)
_SPAN_V_SHIFT = FRACBITS - 6   # 10
_SPAN_V_MASK = 63 * 64         # 4032
_SPAN_U_MASK = 63              # 63


# =========================================================================== #
# gate                                                                        #
# =========================================================================== #
def drawcol_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """``C4_DOOM_DRAWCOL`` gate (default OFF).

    OFF -> the native ops are not registered / not baked / the peephole is a
    no-op / ``isa.num_ops_effective`` stays at its non-DRAWCOL width -> golden
    ``069cc32f`` byte-identical and the title ``V_DrawPatch``/DRAWSPAN path is
    the default.
    """
    e = os.environ if env is None else env
    return e.get("C4_DOOM_DRAWCOL", "0") not in ("0", "", "false", "False")


def num_ops_effective() -> int:
    """OP_IS one-hot band width for the CURRENT ``C4_DOOM_DRAWCOL`` state.

    Delegates to :func:`isa.num_ops_effective`, which folds in the
    ``C4_DOOM_DRAWCOL`` gate (``NUM_OPS_DRAWCOL`` = 50 when ON, so the
    DRAWSPANF opcode-49 one-hot has a slot).  Flag-OFF is byte-identical to
    golden ``069cc32f``; flag-ON is the intended MOVED fingerprint (wider OP_IS
    band by construction).
    """
    return isa.num_ops_effective()


def register_opcode() -> None:
    """Register DRAWCOL/DRAWSPANF into :mod:`isa` (idempotent, gated).

    Only mutates ``isa.NAMES`` / ``isa.BY_NAME`` so the assembler can emit them;
    it never widens ``isa.NUM_OPS`` itself, so a build with the gate OFF is
    byte-identical.  The OP_IS band widening (flag-ON) is folded into
    ``isa.num_ops_effective`` by the ``C4_DOOM_DRAWCOL`` gate -- exactly the
    ``NUM_OPS_FLOAT`` / DRAWSPAN mechanism.
    """
    isa.NAMES.setdefault(DRAWCOL, "DRAWCOL")
    isa.BY_NAME.setdefault("DRAWCOL", DRAWCOL)
    isa.NAMES.setdefault(DRAWSPANF, "DRAWSPANF")
    isa.BY_NAME.setdefault("DRAWSPANF", DRAWSPANF)


# =========================================================================== #
# helpers                                                                     #
# =========================================================================== #
def _sx(v: int) -> int:
    """32-bit two's-complement sign-extend (the VM's signed view of a word)."""
    v &= _MASK32
    return v - (1 << 32) if v & _SIGN else v


# =========================================================================== #
# 1. NATIVE REFERENCE -- byte-exact vs the R_DrawColumn / R_DrawSpan loops      #
# =========================================================================== #
def draw_column(mem: bytearray, dest: int, source: int, colormap: int,
                count: int, frac: int, fracstep: int,
                stride: int = SCREENWIDTH) -> None:
    """``DRAWCOL(dest, source, colormap, count, frac, fracstep, stride)`` --
    byte-exact to the ``R_DrawColumn`` texture-mapped wall-column fill loop::

        do {
            *dest = colormap[source[(frac>>16)&127]];
            dest += stride;   // SCREENWIDTH = 320
            frac += fracstep;
        } while (count--);

    Runs the body ``count+1`` times for ``count >= 0`` (the ``do..while(count--)``
    post-test) and ZERO times for ``count < 0`` (``R_DrawColumn``'s early
    ``if (count < 0) return;``).  Each iteration:

      * ``t   = source[(frac>>16) & 127]``   -- the wall-texture texel (0..255),
      * ``pix = colormap[t & 255]``          -- the light/colormap remap,
      * ``*dest = pix``                      -- one BYTE written to the framebuffer,
      * ``dest += stride`` (strided down the column), ``frac += fracstep``.

    All arithmetic is 32-bit; ``frac`` wraps mod 2**32 (the C ``int`` add).  The
    texel read is a signed-char ``LC`` in the VM but masked to ``& 255`` before
    the colormap index (matching the ported ``& 255`` in ``doom_run.c``), so the
    colormap index is an unsigned 0..255.  Mutates ``mem`` in place; leaves no
    return value (the fused op leaves AX at the ``count--`` chain's terminal 0).
    """
    d = dest & _MASK32
    src = source & _MASK32
    cmap = colormap & _MASK32
    n = _sx(count)
    st = _sx(stride)
    fr = frac & _MASK32
    fstep = fracstep & _MASK32
    for _ in range(n + 1 if n >= 0 else 0):
        texel = mem[(src + ((fr >> FRACBITS) & TEXHEIGHT_MASK)) & _MASK32] & 0xFF
        pix = mem[(cmap + texel) & _MASK32] & 0xFF
        mem[d] = pix
        d = (d + st) & _MASK32
        fr = (fr + fstep) & _MASK32


def draw_span(mem: bytearray, dest: int, source: int, colormap: int,
              count: int, xfrac: int, yfrac: int,
              xstep: int, ystep: int) -> None:
    """``DRAWSPANF(dest, source, colormap, count, xfrac, yfrac, xstep, ystep)``
    -- byte-exact to the ``R_DrawSpan`` texture-mapped floor/ceiling fill loop::

        do {
            spot = ((yfrac>>10)&(63*64)) + ((xfrac>>16)&63);
            *dest++ = colormap[source[spot]];   // CONTIGUOUS write
            xfrac += xstep;
            yfrac += ystep;
        } while (count--);

    Runs the body ``count+1`` times for ``count >= 0`` (post-test) and zero for
    ``count < 0``.  Each iteration folds ``(xfrac, yfrac)`` into a ``0..4095``
    index into a 64x64 flat tile (``spot``), reads the flat texel, remaps it
    through the colormap, and writes ONE byte to a CONTIGUOUS destination
    (``*dest++``).  All arithmetic is 32-bit with wraparound.  Mutates ``mem``.
    """
    d = dest & _MASK32
    src = source & _MASK32
    cmap = colormap & _MASK32
    n = _sx(count)
    xf = xfrac & _MASK32
    yf = yfrac & _MASK32
    xs = xstep & _MASK32
    ys = ystep & _MASK32
    for _ in range(n + 1 if n >= 0 else 0):
        spot = (((yf >> _SPAN_V_SHIFT) & _SPAN_V_MASK)
                + ((xf >> FRACBITS) & _SPAN_U_MASK)) & _MASK32
        texel = mem[(src + spot) & _MASK32] & 0xFF
        pix = mem[(cmap + texel) & _MASK32] & 0xFF
        mem[d] = pix
        d = (d + 1) & _MASK32
        xf = (xf + xs) & _MASK32
        yf = (yf + ys) & _MASK32


# Step-cost constants, from the compiled R_DrawColumn / R_DrawSpan inner loops on
# c4vm32.py (the ``do { ... } while (count--)`` bodies).  Each is the count of
# decoded instrs the INLINE per-pixel loop costs PER PIXEL (the body executed
# once), plus the final failed loop test.  Derived from the c4-ported disasm
# (see the loop disasm captured in ``verify_drawcol_onvm.py``); measured on-VM
# by the correctness harness.
_COL_LOOP_BODY = 30    # decoded instrs per wall pixel (R_DrawColumn body)
_COL_LOOP_EXIT = 11    # the final (false) count-- test that exits the loop
_SPAN_LOOP_BODY = 40   # decoded instrs per floor pixel (R_DrawSpan body)
_SPAN_LOOP_EXIT = 11   # the final (false) count-- test


def drawcol_step_cost_c(count: int) -> int:
    """Decoded VM STEPS the INLINE ``R_DrawColumn`` loop costs for ``count`` (the
    per-pixel loop the native DRAWCOL op replaces).  The body runs ``count+1``
    times for ``count >= 0``; the op replaces the whole run with ONE step."""
    n = _sx(count)
    if n < 0:
        return _COL_LOOP_EXIT
    return _COL_LOOP_BODY * (n + 1) + _COL_LOOP_EXIT


def drawspanf_step_cost_c(count: int) -> int:
    """Decoded VM STEPS the INLINE ``R_DrawSpan`` loop costs for ``count``."""
    n = _sx(count)
    if n < 0:
        return _SPAN_LOOP_EXIT
    return _SPAN_LOOP_BODY * (n + 1) + _SPAN_LOOP_EXIT


# =========================================================================== #
# 2. FUSED MEGABLOCK SCHEDULES (each native op's block sequence)               #
# =========================================================================== #
@dataclass
class MegablockSchedule:
    name: str
    blocks: List[str] = field(default_factory=list)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def drawcol_megablock() -> MegablockSchedule:
    """DRAWCOL fused schedule: pop operands -> guard -> recurrent
    texture-map-and-strided-write body -> return -> ax-mux.

    The ``col-fill`` body is ONE recurrent stored block (compute the texel index
    ``(frac>>16)&127``, read ``source[idx]``, remap through ``colormap``, write
    the running strided dst, bump ``dst`` by ``stride`` and ``frac`` by
    ``fracstep``, decrement the counter), iterated ``count+1`` times inside the
    single decoded step -- exactly the recurrent-body lever DRAWSPAN's
    ``span-copy`` and BLIT's ``blit-store`` use.  One decoded VM step.
    """
    return MegablockSchedule(
        name="DRAWCOL",
        blocks=[
            "alu-expand",   # pop dest,source,colormap,count,frac,fracstep,stride
            "col-guard",    # [count<0] -> skip the fill burst (early-return guard)
            "col-texidx",   # idx = (frac>>16)&127  (fixed-point texel index)
            "col-fill",     # recurrent: dst=cmap[src[idx]]; dst+=stride; frac+=step
            "col-ret",      # AX = 0 (the count-- chain terminal), frame locals set
            "ax-mux",       # write result -> AX (shared)
        ],
    )


def drawspanf_megablock() -> MegablockSchedule:
    """DRAWSPANF fused schedule: pop operands -> guard -> recurrent 2D-uv
    texture-map-and-contiguous-write body -> return -> ax-mux.

    The ``span-fill`` body folds ``(xfrac,yfrac)`` into ``spot``, reads the flat
    texel, remaps through the colormap, writes the running CONTIGUOUS dst, bumps
    ``dst`` by 1 and ``xfrac/yfrac`` by their steps, decrements the counter --
    iterated ``count+1`` times in one decoded step.  One decoded VM step.
    """
    return MegablockSchedule(
        name="DRAWSPANF",
        blocks=[
            "alu-expand",   # pop dest,source,colormap,count,xfrac,yfrac,xstep,ystep
            "span-guard",   # [count<0] -> skip (post-test guard)
            "span-spot",    # spot = ((yfrac>>10)&4032) + ((xfrac>>16)&63)
            "span-fill",    # recurrent: *dst++=cmap[src[spot]]; xfrac+=xs; yfrac+=ys
            "span-ret",     # AX = 0, frame locals set
            "ax-mux",       # write result -> AX (shared)
        ],
    )


# =========================================================================== #
# 3. INTRINSIC RECOGNITION -- bytecode peephole keyed on the loop-body SIGNATURE #
#                                                                             #
# The compiled ``R_DrawColumn`` / ``R_DrawSpan`` inner loops are INLINE (no JSR  #
# call site), so we key on the compiled OPCODE SIGNATURE, exactly as DRAWSPAN    #
# keys on the V_DrawPatch loop.  The c4-ported ``do..while(count--)`` idiom is    #
# lowered as ``while (__do_once0 || (count--))`` -> a ``LEA __do_once0; LI; BNZ`` #
# first-iter guard at the loop HEAD, the ``count--`` post-test, the body, and a   #
# ``JMP`` back-edge to the head.  The DISCRIMINATING fingerprints are:            #
#                                                                             #
#   DRAWCOL   : ``IMM 16; SHR`` then ``IMM 127; AND`` (the (frac>>16)&127 texel   #
#               index) + a STRIDED ``IMM 320; ADD`` dest bump + two ``LC`` (texel #
#               + colormap) and one ``SC``.                                       #
#   DRAWSPANF : the 2D-spot fold ``IMM 64; MUL`` + ``IMM 63; AND`` + a            #
#               CONTIGUOUS ``*dest++`` (NO IMM 320) + two ``LC`` and one ``SC``.  #
#                                                                             #
# We scan for these signatures within the loop span (head ``JMP``-closed) and     #
# rewrite the whole span to the native op + NOP padding (PRESERVING STREAM        #
# LENGTH, so every branch/JSR target stays valid) -- the DRAWSPAN discipline.     #
# ---------------------------------------------------------------------------


@dataclass
class IntrinsicMatch:
    """One recognised R_DrawColumn / R_DrawSpan inner-loop span.

    ``kind`` is ``"col"`` or ``"span"``; ``start`` is the loop-head index (the
    ``LEA __do_once0`` of the first-iter guard); ``length`` is the span length
    (head..JMP inclusive); ``stride`` is the decoded dest step (320 for a column,
    1 for a floor span)."""
    kind: str
    start: int
    length: int
    stride: int


def _op(ins, is_tuple):
    return ins[0] if is_tuple else ins.op


def _imm(ins, is_tuple):
    return ins[1] if is_tuple else ins.imm


def _span_has(code, lo, hi, is_tuple, op, imm=None) -> bool:
    """True if ``(op[, imm])`` appears in ``code[lo:hi]``."""
    for k in range(lo, hi):
        if _op(code[k], is_tuple) == op and (imm is None or _imm(code[k], is_tuple) == imm):
            return True
    return False


def find_drawcol_loops(code) -> List[IntrinsicMatch]:
    """Scan ``code`` for compiled R_DrawColumn / R_DrawSpan inner-loop spans by
    opcode signature.  ``code`` is a list of ``(op, imm)`` tuples (the ``c4vm32``
    decoded form) or a list of ``isa.Instr``.  Returns one :class:`IntrinsicMatch`
    per recognised texture-mapped fill loop.

    A span is delimited by a loop-head (``LEA; LI; BNZ`` first-iter guard) and the
    ``JMP`` back-edge that targets it.  Inside the span we require the fill
    fingerprints (two ``LC`` + one ``SC``) plus the DRAWCOL / DRAWSPANF
    discriminator; a coincidental opcode run that is NOT a real fill loop is
    rejected.
    """
    is_tuple = bool(code) and isinstance(code[0], tuple)
    O = isa
    n = len(code)
    out: List[IntrinsicMatch] = []
    i = 0
    while i < n:
        # loop head: LEA __do_once0 ; LI ; BNZ <body>
        if (i + 2 < n and _op(code[i], is_tuple) == O.LEA
                and _op(code[i + 1], is_tuple) == O.LI
                and _op(code[i + 2], is_tuple) == O.BNZ):
            # find the JMP back-edge that targets this head, within a bounded window
            jmp_idx = None
            for j in range(i + 3, min(n, i + 120)):
                if _op(code[j], is_tuple) == O.JMP and _imm(code[j], is_tuple) == i:
                    jmp_idx = j
                    break
            if jmp_idx is not None:
                lo, hi = i, jmp_idx + 1
                # both fills read a texel + colormap (two LC) and write one byte (SC)
                lc = sum(1 for k in range(lo, hi)
                         if _op(code[k], is_tuple) == O.LC)
                sc = sum(1 for k in range(lo, hi)
                         if _op(code[k], is_tuple) == O.SC)
                if lc >= 2 and sc >= 1:
                    # The mask/step VALUES are carried by the preceding IMM
                    # (AND/MUL/SHR themselves take the operand off the stack, so
                    # their own immediate is 0).  Discriminate on the IMM values:
                    #   DRAWCOL   : IMM 127 (the (frac>>16)&127 texel mask) AND a
                    #               STRIDED IMM 320 (dest += SCREENWIDTH).
                    #   DRAWSPANF : IMM 64 (the 63*64 v-fold MUL) + IMM 63 (u mask),
                    #               and NO IMM 320 (the write is *dest++, dest+=1).
                    is_col = (_span_has(code, lo, hi, is_tuple, O.IMM, TEXHEIGHT_MASK)
                              and _span_has(code, lo, hi, is_tuple, O.IMM, SCREENWIDTH))
                    is_span = (_span_has(code, lo, hi, is_tuple, O.IMM, 64)
                               and _span_has(code, lo, hi, is_tuple, O.IMM, _SPAN_U_MASK)
                               and _span_has(code, lo, hi, is_tuple, O.MUL, None)
                               and not _span_has(code, lo, hi, is_tuple, O.IMM, SCREENWIDTH))
                    if is_col:
                        out.append(IntrinsicMatch("col", lo, hi - lo, SCREENWIDTH))
                        i = hi
                        continue
                    if is_span:
                        out.append(IntrinsicMatch("span", lo, hi - lo, 1))
                        i = hi
                        continue
        i += 1
    return out


def substitute_intrinsics(code):
    """Peephole: replace each recognised R_DrawColumn / R_DrawSpan inner-loop span
    with its native op (DRAWCOL / DRAWSPANF), PRESERVING STREAM LENGTH.

    The loop-head instruction becomes ``<DRAWCOL>`` / ``<DRAWSPANF>`` (carrying
    the decoded dest stride as its immediate); EVERY other instruction in the
    span -- including the ``JMP`` back-edge -- becomes ``NOP``.  Stream LENGTH is
    UNCHANGED (span-in == span-out), so every branch / JSR target stays valid with
    NO re-resolution -- the DRAWSPAN discipline.  ``code`` is a list of
    ``(op, imm)`` tuples or a list of ``isa.Instr``.

    Returns ``(new_code, n_substitutions)``.
    """
    out = list(code)
    is_tuple = bool(out) and isinstance(out[0], tuple)
    matches = find_drawcol_loops(out)
    for m in matches:
        opc = DRAWCOL if m.kind == "col" else DRAWSPANF
        out[m.start] = (opc, m.stride) if is_tuple else isa.Instr(opc, m.stride)
        for j in range(m.start + 1, m.start + m.length):
            out[j] = (isa.NOP, 0) if is_tuple else isa.Instr(isa.NOP, 0)
    return out, len(matches)


# =========================================================================== #
# 4. VERIFICATION BATTERY -- count/frac/colormap/boundary edges + real cases   #
# =========================================================================== #
def battery_cases_col() -> List[Tuple[int, int, int, int, int, int, int]]:
    """R_DrawColumn battery of
    ``(dest, source, colormap, count, frac, fracstep, stride)`` cases,
    exercising:

      * COUNT edges: -1 (writes nothing, the early-return guard), 0 (one pixel),
        1, a full 200-row column, large,
      * FRAC / FRACSTEP: zero step (constant texel), 1:1 step, a real DDA scale,
        a step that wraps ``frac`` past 2**32, a texel index that walks the whole
        0..127 texture-column mask,
      * COLORMAP indirection: a colormap that is NOT the identity (so a
        'forgot the colormap' bug shows up),
      * STRIDE: 320 (a real screen column) and a small stride (adjacent rows).
    """
    SRC = 0x100000    # 128-byte wall-texture column
    CMAP = 0x110000   # 256-byte colormap
    DST = 0x200000
    cases: List[Tuple[int, int, int, int, int, int, int]] = []
    ONE = 1 << FRACBITS
    # (dest, source, colormap, count, frac, fracstep, stride)
    cases.append((DST, SRC, CMAP, -1, 0, ONE, 320))          # count<0 -> nothing
    cases.append((DST, SRC, CMAP, 0, 0, ONE, 320))           # one pixel
    cases.append((DST, SRC, CMAP, 1, 0, ONE, 320))           # two pixels
    cases.append((DST, SRC, CMAP, 199, 0, ONE, 320))         # full 200-row column
    cases.append((DST, SRC, CMAP, 500, 0, ONE, 320))         # large
    cases.append((DST, SRC, CMAP, 50, 0, 0, 320))            # zero step (const texel)
    cases.append((DST, SRC, CMAP, 127, 0, ONE, 320))         # walk the whole 0..127 mask
    cases.append((DST, SRC, CMAP, 40, 0x1234, 0x9ABC, 320))  # real DDA scale
    cases.append((DST, SRC, CMAP, 60, 0xFFFF0000, 0x40000, 320))  # frac wraps 2**32
    cases.append((DST, SRC, CMAP, 10, 0, ONE, 4))            # small stride
    cases.append((DST, SRC, CMAP, 8, 3 << FRACBITS, ONE, 320))    # nonzero start texel
    return cases


def battery_cases_span() -> List[Tuple[int, int, int, int, int, int, int, int]]:
    """R_DrawSpan battery of
    ``(dest, source, colormap, count, xfrac, yfrac, xstep, ystep)`` cases,
    exercising:

      * COUNT edges: -1 (nothing), 0 (one pixel), a full 320-wide span, large,
      * u/v walk: zero steps (constant spot), steps that walk the whole 64x64
        flat tile, steps that wrap ``xfrac`` / ``yfrac`` past 2**32,
      * COLORMAP indirection: same as the column battery,
      * SPOT fold: cases whose ``spot`` sweeps the tile.
    """
    SRC = 0x100000    # 4096-byte 64x64 flat tile
    CMAP = 0x110000
    DST = 0x200000
    cases: List[Tuple[int, int, int, int, int, int, int, int]] = []
    ONE = 1 << FRACBITS
    # (dest, source, colormap, count, xfrac, yfrac, xstep, ystep)
    cases.append((DST, SRC, CMAP, -1, 0, 0, ONE, ONE))       # count<0 -> nothing
    cases.append((DST, SRC, CMAP, 0, 0, 0, ONE, ONE))        # one pixel
    cases.append((DST, SRC, CMAP, 1, 0, 0, ONE, ONE))        # two pixels
    cases.append((DST, SRC, CMAP, 319, 0, 0, ONE, ONE))      # full 320-wide floor span
    cases.append((DST, SRC, CMAP, 800, 0, 0, ONE, ONE))      # large
    cases.append((DST, SRC, CMAP, 60, 0, 0, 0, 0))           # zero steps (const spot)
    cases.append((DST, SRC, CMAP, 63, 0, 0, ONE, 0))         # walk u 0..63
    cases.append((DST, SRC, CMAP, 63, 0, 0, 0, ONE << 4))    # walk v 0..63
    cases.append((DST, SRC, CMAP, 100, 0x1111, 0x2222, 0x8000, 0x4000))  # real u,v walk
    cases.append((DST, SRC, CMAP, 80, 0xFFFF0000, 0xFFFF0000, 0x80000, 0x80000))  # wrap
    return cases


def _seed_cmap(cmap_base: int, mem: bytearray) -> None:
    """Seed a NON-identity colormap so a 'forgot the colormap remap' bug shows
    up: ``colormap[i] = (i*7 + 13) & 255`` for ``i`` in 0..255."""
    for i in range(256):
        mem[(cmap_base + i) & _MASK32] = (i * 7 + 13) & 0xFF


def verify_byte_exact() -> Dict[str, object]:
    """Verify DRAWCOL / DRAWSPANF are byte-exact vs their reference loops over
    the batteries.

    For each case: seed a KNOWN source-texture pattern + a NON-identity colormap,
    PRE-POISON the whole destination footprint with a sentinel (so a 'wrote
    nothing' bug can never masquerade as a pass -- the ``doom_blit`` bug), run the
    op, and compare against an INDEPENDENT sequential re-implementation of the
    SAME live-memory C loop.  Returns ``{n, fail, detail}`` -- 0 fails ==
    byte-exact.  (The AUTHORITATIVE on-VM oracle -- the REAL compiled
    ``R_DrawColumn`` / ``R_DrawSpan`` run on ``c4vm32.py`` -- is the separate
    ``verify_drawcol_onvm.py`` harness.)
    """
    res: Dict[str, object] = {"n": 0, "fail": 0, "detail": []}

    # ---- R_DrawColumn ----
    for (dest, source, colormap, count, frac, fracstep, stride) in battery_cases_col():
        n = _sx(count)
        st = _sx(stride)
        iters = n + 1 if n >= 0 else 0
        top = max(source + 128, colormap + 256,
                  dest + max(0, iters - 1) * max(st, 1)) + 64
        base = bytearray(top)
        for k in range(128):
            base[(source + k) & _MASK32] = (k * 13 + 5) & 0xFF   # texture column
        _seed_cmap(colormap, base)
        sentinel = 0xEE
        d0 = dest & _MASK32
        for k in range(iters):
            base[(d0 + k * st) & _MASK32] = sentinel

        got = bytearray(base)
        draw_column(got, dest, source, colormap, count, frac, fracstep, stride)

        ref = bytearray(base)
        d = dest & _MASK32
        fr = frac & _MASK32
        fstep = fracstep & _MASK32
        for _ in range(iters):
            texel = ref[(source + ((fr >> FRACBITS) & TEXHEIGHT_MASK)) & _MASK32] & 0xFF
            ref[d] = ref[(colormap + texel) & _MASK32] & 0xFF
            d = (d + st) & _MASK32
            fr = (fr + fstep) & _MASK32

        res["n"] = int(res["n"]) + 1
        if got != ref:
            res["fail"] = int(res["fail"]) + 1
            ndiff = sum(1 for a, b in zip(got, ref) if a != b)
            res["detail"].append({"op": "DRAWCOL", "count": count,
                                  "frac": frac, "fracstep": fracstep,
                                  "diff_bytes": ndiff})

    # ---- R_DrawSpan ----
    for (dest, source, colormap, count, xfrac, yfrac, xstep, ystep) in battery_cases_span():
        n = _sx(count)
        iters = n + 1 if n >= 0 else 0
        top = max(source + 4096, colormap + 256, dest + max(0, iters)) + 64
        base = bytearray(top)
        for k in range(4096):
            base[(source + k) & _MASK32] = (k * 17 + 3) & 0xFF   # 64x64 flat tile
        _seed_cmap(colormap, base)
        sentinel = 0xEE
        d0 = dest & _MASK32
        for k in range(iters):
            base[(d0 + k) & _MASK32] = sentinel

        got = bytearray(base)
        draw_span(got, dest, source, colormap, count, xfrac, yfrac, xstep, ystep)

        ref = bytearray(base)
        d = dest & _MASK32
        xf = xfrac & _MASK32
        yf = yfrac & _MASK32
        xs = xstep & _MASK32
        ys = ystep & _MASK32
        for _ in range(iters):
            spot = (((yf >> _SPAN_V_SHIFT) & _SPAN_V_MASK)
                    + ((xf >> FRACBITS) & _SPAN_U_MASK)) & _MASK32
            texel = ref[(source + spot) & _MASK32] & 0xFF
            ref[d] = ref[(colormap + texel) & _MASK32] & 0xFF
            d = (d + 1) & _MASK32
            xf = (xf + xs) & _MASK32
            yf = (yf + ys) & _MASK32

        res["n"] = int(res["n"]) + 1
        if got != ref:
            res["fail"] = int(res["fail"]) + 1
            ndiff = sum(1 for a, b in zip(got, ref) if a != b)
            res["detail"].append({"op": "DRAWSPANF", "count": count,
                                  "xfrac": xfrac, "yfrac": yfrac,
                                  "diff_bytes": ndiff})

    return res
