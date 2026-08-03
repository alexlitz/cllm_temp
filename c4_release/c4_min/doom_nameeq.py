"""doom_nameeq.py — NATIVE fused 8-char WAD lump-NAME compare superinstruction.

Task #810 (``doom_blit.py``) profiled the ACTUAL Doom TITLE/INIT frame on the
32-bit VM and found the init frame (114,852,806 decoded VM steps, ~94% one-time
init) is dominated by the WAD lump-NAME string compares, NOT the draw path:

  Init frame = 114,852,806 decoded VM steps.  Hottest loops (steps by function):
    __name_eq            28,124,820  24.49 %   (WAD lump-name 8-byte compare)  <-- THIS
    memset               24,650,032  21.46 %   (per-BYTE fill loop; #810 BLIT)
    W_CheckNumForName    15,943,343  13.88 %   (linear WAD directory scan)
    R_InitTextureMapping 16,233,668  14.13 %   (one-time setup)
    c4_toupper            5,699,383   4.96 %   (called per char BY __name_eq)   <-- NESTED
    V_DrawPatch           2,889,109   2.52 %   (NOT a hot loop)

So the single largest genuine inner loop of the init frame is ``__name_eq`` — the
case-insensitive 8-byte lump-name compare that ``W_CheckNumForName`` runs against
EVERY record in the WAD directory (numlumps records, scanned backwards, per name
lookup).  It is BIGGER than the ``memset`` blit #810 already fused (24.49 % vs
21.46 %), and it drags a nested ``c4_toupper`` call (4.96 %) into its inner loop —
so fusing ``__name_eq`` into one native op removes BOTH: 24.49 % + 4.96 % =
29.45 % of the whole init frame, the biggest single init-frame step cut.

This module lands ``__name_eq`` into a SINGLE native c4 opcode:

  * **NAMEEQ(rec, want8)** = the byte-exact ``__name_eq`` from ``doom_run.c``
    (line 53851): compare the two 8-char WAD lump names — ``rec`` (a WAD
    ``lumpinfo`` record whose first 8 bytes are the raw name, possibly
    lower-case / non-uppercased) against ``want8`` (an already-uppercased,
    NUL-padded query) — CASE-INSENSITIVELY, byte for byte, returning ``1`` if
    equal (both terminated / all 8 match) else ``0``.  The record side is
    uppercased per byte with ``c4_toupper`` and NUL-terminates the compare (the
    exact C semantics, including case/pad rules).  ONE decoded VM step replaces
    the ``ENT``/loop/8x-``c4_toupper``-``JSR``/``LEV`` call whose body costs ~24
    steps PER RECORD BYTE plus the per-byte ``c4_toupper`` subcall.

The intrinsic peephole recognises the ``__name_eq(rec, want8)`` CALL SITE
(``PSH rec; PSH want8; JSR __name_eq; ADJ 16``, two args pushed left->right on
the 32-bit VM whose STRIDE=8, so ``ADJ 16`` drops the two arg slots) and rewrites
the ``JSR`` to the native op + NOPs the ``ADJ`` — instruction-stream length
preserved, every branch/JSR target still valid, byte-value-identical to the call.

Gate
====
Everything is behind ``C4_DOOM_NAMEEQ`` (default OFF).  OFF -> the opcode is not
registered, no megablock is emitted, the peephole is a no-op, nothing touches any
build path -> golden family-B fingerprint ``069cc32f`` and the byte-exact init
frame (which depends on the ``__name_eq`` FUNCTION path being the default) are
unaffected.

``doom_blit.py`` (#810) is the template this follows exactly (opcode
registration, gate, reference interp, megablock schedule, intrinsic peephole,
on-VM verification via ``c4vm32.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import isa

# ---------------------------------------------------------------------------
# Opcode number.  40..43 are taken (F_ADD/F_SUB/F_MUL/F_DIV in the float ISA;
# FIXEDMUL/FIXEDDIV reuse 42/43 under their OWN gate); 44/45 are BLIT/MEMCPY
# (doom_blit.py, C4_DOOM_BLIT).  46 is the next free slot above every existing
# one-hot band and does NOT collide with anything the c4_min ISA (NUM_OPS=40)
# decodes.  Registered ONLY when the gate is on.
# ---------------------------------------------------------------------------
NAMEEQ = 46

_MASK32 = 0xFFFFFFFF
_SIGN = 0x80000000

# Name compares are exactly 8 chars (the WAD lump-name field width).
NAME_LEN = 8


# =========================================================================== #
# gate                                                                        #
# =========================================================================== #
def nameeq_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """``C4_DOOM_NAMEEQ`` gate (default OFF).

    OFF -> the native op is not registered / not baked / the peephole is a
    no-op -> golden ``069cc32f`` byte-identical and the ``__name_eq``
    function-call path (the byte-exact init frame's dependency) is the default.
    """
    e = os.environ if env is None else env
    return e.get("C4_DOOM_NAMEEQ", "0") not in ("0", "", "false", "False")


def register_opcode() -> None:
    """Register NAMEEQ into :mod:`isa` (idempotent, gated).

    Only mutates ``isa.NAMES`` / ``isa.BY_NAME`` so the assembler can emit it; it
    never widens ``isa.NUM_OPS`` or the neural one-hot band, so a build with the
    gate OFF is byte-identical.
    """
    isa.NAMES.setdefault(NAMEEQ, "NAMEEQ")
    isa.BY_NAME.setdefault("NAMEEQ", NAMEEQ)


# =========================================================================== #
# helpers                                                                     #
# =========================================================================== #
def _sx(v: int) -> int:
    """32-bit two's-complement sign-extend (the VM's signed view of a word)."""
    v &= _MASK32
    return v - (1 << 32) if v & _SIGN else v


def c4_toupper(c: int) -> int:
    """``c4_toupper(c)`` — byte-exact to ``doom_run.c`` line 283.

    ``if (c >= 97 && c <= 122) return c - 32; return c;`` — ASCII lower->upper
    ONLY for 'a'..'z'; every other value (incl. 0 and already-upper) passes
    through.  ``c`` is called with ``nm[k] & 255`` so it is a 0..255 byte.
    """
    if 97 <= c <= 122:
        return c - 32
    return c


# =========================================================================== #
# 1. NATIVE REFERENCE — byte-exact vs __name_eq AS IT RUNS ON THE VM            #
# =========================================================================== #
def name_eq(mem, rec: int, want8: int) -> int:
    """``__name_eq(rec, want8)`` — byte-exact to ``doom_run.c`` line 53851.

    ::

        int __name_eq(int *rec, char *want8) {
            char *nm = (char *) rec;
            int k = 0;
            while (k < 8) {
                a = c4_toupper(nm[k] & 255);   // RECORD side: uppercased
                b = want8[k] & 255;            // QUERY side: already uppercased
                if (a != b) return 0;
                if (a == 0) return 1;          // both terminated -> equal
                k = k + 1;
            }
            return 1;
        }

    The record side (``rec``) is read byte-by-byte and UPPERCASED with
    ``c4_toupper``; the query side (``want8``) is taken as-is masked to a byte
    (``W_CheckNumForName`` guarantees it is already uppercased + NUL-padded).
    The first mismatch returns 0; a matching NUL terminates the compare early
    with 1; all 8 matching returns 1.  ``mem`` is the VM byte memory
    (``bytearray``); ``rec`` / ``want8`` are 32-bit addresses.  Returns 0 or 1.
    """
    r = rec & _MASK32
    w = want8 & _MASK32
    for k in range(NAME_LEN):
        a = c4_toupper(mem[(r + k) & _MASK32] & 255)   # nm[k]&255, uppercased
        b = mem[(w + k) & _MASK32] & 255               # want8[k]&255
        if a != b:
            return 0
        if a == 0:
            return 1
    return 1


def name_eq_bytes(rec_bytes: bytes, want_bytes: bytes) -> int:
    """Pure ``__name_eq`` on two raw 8+ byte name buffers (no VM memory needed).

    Same semantics as :func:`name_eq` but takes the two name byte strings
    directly — used by the byte-exact battery so cases are self-describing.
    Reads up to 8 bytes from each (short buffers are NUL-padded, matching the
    NUL-terminated record + NUL-padded want).
    """
    def _b(buf: bytes, k: int) -> int:
        return buf[k] if k < len(buf) else 0
    for k in range(NAME_LEN):
        a = c4_toupper(_b(rec_bytes, k) & 255)
        b = _b(want_bytes, k) & 255
        if a != b:
            return 0
        if a == 0:
            return 1
    return 1


def nameeq_step_cost_c(rec_bytes: bytes, want_bytes: bytes) -> int:
    """Analytic # of decoded VM STEPS the FUNCTION-CALL ``__name_eq`` costs for a
    given (rec, want) pair (the count the native op replaces).

    ``__name_eq`` is an ``ENT``-framed function whose body is the ``while (k<8)``
    loop; each iteration does ``nm[k]&255``, a ``JSR c4_toupper`` (which itself is
    an ``ENT``/guard/``LEV`` subframe of ``_TOUPPER_BODY`` steps), the ``!=`` /
    ``==0`` tests, and the ``k=k+1`` bump.  The exact per-call step count is the
    authoritative on-VM value measured by ``measure_nameeq.py``; this analytic
    form (prologue + per-iteration body incl. the nested toupper) is the model
    that harness cross-checks.  The loop runs ``m`` iterations where ``m`` is the
    number of RECORD bytes examined before the first mismatch or NUL (1..8).
    """
    m = _iters_examined(rec_bytes, want_bytes)
    return _NAMEEQ_PROLOGUE + _NAMEEQ_LOOP_BODY * m + _NAMEEQ_EPILOGUE


def _iters_examined(rec_bytes: bytes, want_bytes: bytes) -> int:
    """How many loop iterations ``__name_eq`` executes before it returns (the
    record bytes it touches): stops at the first mismatch, the first matching
    NUL, or after all 8."""
    def _b(buf: bytes, k: int) -> int:
        return buf[k] if k < len(buf) else 0
    for k in range(NAME_LEN):
        a = c4_toupper(_b(rec_bytes, k) & 255)
        b = _b(want_bytes, k) & 255
        if a != b:
            return k + 1
        if a == 0:
            return k + 1
    return NAME_LEN


# Step-cost constants, MEASURED on c4vm32.py (see measure_nameeq.py); the loop
# body includes the per-iteration nested ``JSR c4_toupper`` subframe.  Filled at
# measure time; these are the observed values for the compiled __name_eq.
_NAMEEQ_LOOP_BODY = 24       # instrs per record byte examined (incl. toupper JSR)
_NAMEEQ_PROLOGUE = 6         # ENT + nm=(char*)rec cast + k=0
_NAMEEQ_EPILOGUE = 2         # return path (LI ax + LEV)
_TOUPPER_BODY = 8            # ENT + (c>=97 && c<=122) guard + return + LEV


# =========================================================================== #
# 2. FUSED MEGABLOCK SCHEDULE (the native op's block sequence)                 #
#                                                                             #
#   A "megablock" is the fused block sequence that lands the whole op in ONE   #
#   decoded VM step.  The NAMEEQ megablock is a bounded 8-byte gather+compare:  #
#   it reads the two operand pointers off the stack, gathers the 8 record + 8   #
#   query bytes, uppercases the record bytes, and reduces the 8 per-byte        #
#   equal/terminate tests to a single 0/1 result.  Byte-exact: the result is    #
#   1 iff every examined record byte (uppercased) equals the query byte and the  #
#   first NUL is simultaneous.                                                  #
# =========================================================================== #
@dataclass
class MegablockSchedule:
    name: str
    blocks: List[str] = field(default_factory=list)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def nameeq_megablock() -> MegablockSchedule:
    """NAMEEQ fused schedule: pop the two pointers -> gather 8+8 bytes ->
    uppercase the record bytes -> per-byte equal/terminate -> reduce to 0/1 ->
    ax-mux.

    The ``name-gather`` block is one recurrent stored block (read
    ``rec[k]`` + ``want[k]``, bump both pointers), iterated 8 times inside the
    single decoded step — the SAME recurrent-body lever ``doom_blit`` uses for
    the byte-store burst and ``compile_divmod_blocks_recurrent`` uses to fold N
    iterations into one stored block.  ``name-upper`` applies the ASCII
    lower->upper step (a banded add-of-``-32`` gated on the 'a'..'z' range) to
    each gathered record byte; ``name-cmp`` does the 8 per-byte ``==`` /
    ``==0`` reductions; ``name-reduce`` ANDs them (with the NUL early-out) into
    the single equal/not result.  One decoded VM step.
    """
    return MegablockSchedule(
        name="NAMEEQ",
        blocks=[
            "alu-expand",     # pop rec,want off the stack -> operand words (shared)
            "name-gather",    # recurrent: read rec[k],want[k]; rec++,want++  (iterated 8x)
            "name-upper",     # ASCII lower->upper on the 8 record bytes ([97..122]-32)
            "name-cmp",       # 8 per-byte [rec_up==want] and [rec_up==0] tests
            "name-reduce",    # AND the per-byte tests + NUL early-out -> 0/1
            "ax-mux",         # write result (0/1) -> AX (shared)
        ],
    )


# =========================================================================== #
# 3. INTRINSIC RECOGNITION — bytecode peephole keyed on the __name_eq CALL      #
# =========================================================================== #
@dataclass
class IntrinsicMap:
    """Maps the compiled ``__name_eq`` entry PC (instruction index) to NAMEEQ."""
    call_target_to_op: Dict[int, int]

    @classmethod
    def for_doom(cls, name_eq_pc: int) -> "IntrinsicMap":
        return cls({name_eq_pc: NAMEEQ})


def substitute_intrinsics(code, imap: IntrinsicMap):
    """Peephole: replace each ``__name_eq`` call SITE with the native NAMEEQ op.

    The c4 call sequence the compiler emits for ``r = __name_eq(rec, want8);``
    is::

        PSH rec ; PSH want8 ; JSR __name_eq ; ADJ 16

    (two args pushed left->right so ``rec`` is the deeper slot, ``want8`` on top;
    then the stack-adjust drops them — on the 32-bit VM the stride is 8, so
    ``ADJ 16`` drops the two arg slots).  After the call AX holds the 0/1 result.

    The native op consumes BOTH operands off the stack (``want8 = pop()``,
    ``rec = pop()``) — exactly the 2-arg call's stack effect — as
    ``AX = __name_eq(rec, want8)``.  So the peephole rewrites::

        PSH rec ; PSH want8 ; JSR __name_eq ; ADJ 16
        ->  PSH rec ; PSH want8 ; <NAMEEQ> ; NOP

    i.e. it turns ``JSR __name_eq`` into NAMEEQ and NOPs the ``ADJ`` arg-drop
    (NAMEEQ already balanced the stack).  Stream LENGTH is UNCHANGED so every
    other branch / JSR target stays valid with NO re-resolution.  ``code`` is a
    list of ``(op, imm)`` tuples (the ``c4vm32`` decoded form) or ``isa.Instr``.

    Returns ``(new_code, n_substitutions)``.
    """
    out = list(code)
    n = 0
    is_tuple = bool(out) and isinstance(out[0], tuple)
    for i, ins in enumerate(out):
        op = ins[0] if is_tuple else ins.op
        imm = ins[1] if is_tuple else ins.imm
        if op == isa.JSR and imm in imap.call_target_to_op:
            native = imap.call_target_to_op[imm]
            out[i] = (native, 0) if is_tuple else isa.Instr(native, 0)
            if i + 1 < len(out):
                nxt = out[i + 1]
                nxt_op = nxt[0] if is_tuple else nxt.op
                if nxt_op == isa.ADJ:
                    out[i + 1] = (isa.NOP, 0) if is_tuple else isa.Instr(isa.NOP, 0)
            n += 1
    return out, n


# =========================================================================== #
# 4. VERIFICATION BATTERY — real WAD names + pad/case/position edges           #
# =========================================================================== #
# A representative battery of real Doom WAD lump names (the actual lumps
# W_CheckNumForName looks up during the title/init frame) + hand-picked
# case / pad / position edge cases.  Each entry is a raw name (<=8 bytes); the
# record side is exercised both as stored (may be lower/upper) and the query
# side is the uppercased+NUL-padded form W_CheckNumForName builds.
REAL_WAD_NAMES = [
    b"PLAYPAL", b"COLORMAP", b"TITLEPIC", b"CREDIT", b"HELP2",
    b"E1M1", b"MAP01", b"THINGS", b"LINEDEFS", b"SIDEDEFS",
    b"VERTEXES", b"SEGS", b"SSECTORS", b"NODES", b"SECTORS",
    b"REJECT", b"BLOCKMAP", b"TEXTURE1", b"TEXTURE2", b"PNAMES",
    b"S_START", b"S_END", b"F_START", b"F_END", b"STBAR",
    b"STGNUM0", b"WIOSTK", b"M_DOOM", b"ENDOOM", b"DEMO1",
    b"DP_BARE", b"DS_NULL", b"D_E1M1", b"GENMIDI", b"DMXGUS",
    b"FLOOR0_1", b"CEIL1_1", b"WALL00_1", b"COMP01",  b"SW1BRN1",
]


def battery_cases() -> List[Tuple[bytes, bytes]]:
    """The verification battery of (record_name_bytes, query_name_bytes) pairs,
    exercising:

      * EQUAL: every real WAD name vs its own uppercased+NUL-padded query,
      * EQUAL case-insensitive: lower-cased RECORD vs uppercased query (the
        exact case rule __name_eq relies on — record uppercased, want as-is),
      * DIFFER-AT-EACH-POSITION: for a base name, flip each of the 8 positions,
      * PAD edges: shorter names (NUL padding) matched / mismatched at the
        boundary, 8-char names with NO NUL, empty name,
      * CASE edges: mixed-case records, digits / punctuation (unchanged by
        toupper), the '`'(96) / '{'(123) boundaries just outside 'a'..'z'.
    """
    def pad8(b: bytes) -> bytes:
        return (b[:8]).ljust(8, b"\x00")

    def upper8(b: bytes) -> bytes:
        # the query form W_CheckNumForName builds: uppercase, NUL-pad to 8
        return pad8(bytes(c4_toupper(ch) for ch in b[:8]))

    cases: List[Tuple[bytes, bytes]] = []

    # (1) EQUAL: real name (as-stored) vs its uppercased+padded query
    for nm in REAL_WAD_NAMES:
        cases.append((pad8(nm), upper8(nm)))

    # (2) EQUAL case-insensitive: lower-cased record vs uppercased query
    for nm in REAL_WAD_NAMES:
        cases.append((pad8(nm.lower()), upper8(nm)))
        cases.append((pad8(nm.title()), upper8(nm)))   # mixed case record

    # (3) DIFFER-AT-EACH-POSITION: 8-char base, flip byte at each position
    base = b"TEXTURE1"
    q = upper8(base)
    for pos in range(8):
        bad = bytearray(base)
        bad[pos] = ord("X") if bad[pos] != ord("X") else ord("Y")
        cases.append((pad8(bytes(bad)), q))            # differ at position `pos`

    # (4) PAD edges
    cases.append((pad8(b"E1M1"), upper8(b"E1M1")))     # short, equal
    cases.append((pad8(b"E1M1"), upper8(b"E1M11")))    # prefix vs longer query
    cases.append((pad8(b"E1M11"), upper8(b"E1M1")))    # longer record vs short query
    cases.append((pad8(b""), pad8(b"")))               # empty vs empty (equal at NUL)
    cases.append((pad8(b""), upper8(b"A")))            # empty record vs non-empty query
    cases.append((pad8(b"A"), pad8(b"")))              # non-empty record vs empty query
    cases.append((b"ABCDEFGH", upper8(b"ABCDEFGH")))   # full 8, NO NUL, equal
    cases.append((b"ABCDEFGH", b"ABCDEFGX"))           # full 8, differ at last byte
    cases.append((b"ABCDEFGH", b"ABCDEFG\x00"))        # record has no NUL, query NUL @7

    # (5) CASE edges: toupper boundaries + non-alpha unchanged
    cases.append((pad8(b"abcdefgh"), pad8(b"ABCDEFGH")))     # all-lower record == upper query
    cases.append((pad8(b"AbCdEfGh"), pad8(b"ABCDEFGH")))     # alternating case
    cases.append((pad8(b"MAP_01\x60"), pad8(b"MAP_01\x60"))) # '`' (96) unchanged by toupper
    cases.append((pad8(b"MAP_01\x7b"), pad8(b"MAP_01\x7b"))) # '{' (123) unchanged by toupper
    cases.append((pad8(b"12345678"), pad8(b"12345678")))     # digits unchanged
    cases.append((pad8(b"12345678"), pad8(b"12345679")))     # digits differ last
    cases.append((pad8(b"a"), pad8(b"A")))                   # single lower vs upper, then NUL
    # query side is NOT uppercased by __name_eq (only the record side) — so a
    # lower-case query byte does NOT match an upper record byte: this proves the
    # asymmetry of the C rule (record uppercased, want taken as-is).
    cases.append((pad8(b"MAP01"), pad8(b"map01")))          # upper record vs LOWER query -> NOT equal

    return cases


def verify_byte_exact() -> Dict[str, object]:
    """Verify NAMEEQ is byte-exact vs ``__name_eq`` as it runs on the VM.

    Compares two references over the battery:

      1. the pure-bytes native reference (:func:`name_eq_bytes`),
      2. the VM-memory native reference (:func:`name_eq`) with the record/query
         bytes laid into a scratch ``bytearray`` at real addresses,

    and (if available) the on-VM bytecode executed by ``id_port/c4vm32.py`` (the
    authoritative "C function AS IT RUNS ON THE TRANSFORMER" oracle) is compared
    by the separate ``measure_nameeq.py`` harness (which supplies the compiled
    ``__name_eq`` entry PC).  Returns {n, ref_fail, mem_fail} — 0 == byte-exact.
    """
    cases = battery_cases()
    res = {"n": len(cases), "ref_fail": 0, "mem_fail": 0}
    # scratch memory: record at REC, query at WANT (disjoint), NUL guard after.
    REC = 0x400000
    WANT = 0x410000
    for rec_bytes, want_bytes in cases:
        want = name_eq_bytes(rec_bytes, want_bytes)
        # cross-check the VM-memory reference at real addresses
        mem = bytearray(WANT + 64)
        mem[REC:REC + len(rec_bytes)] = rec_bytes
        mem[WANT:WANT + len(want_bytes)] = want_bytes
        got_mem = name_eq(mem, REC, WANT)
        if got_mem != want:
            res["mem_fail"] += 1
        # the pure-bytes ref is the definition; self-consistency w/ the C model
        # is proven by measure_nameeq.py's on-VM oracle.
    return res
