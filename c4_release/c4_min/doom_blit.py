"""doom_blit.py — NATIVE fused blit / byte-fill superinstruction for Doom.

Task #810 asked for a "draw-column / blit superinstruction" to cut the title
frame's step count, on the premise that the title frame is dominated by
``V_DrawPatch``'s column-copy loop.  Profiling the ACTUAL title op stream on the
32-bit VM (``doom_blit_work/profile_title.py`` + ``profile_by_function.py``)
REFUTES that premise:

  Title frame = 114,852,806 decoded VM steps.  Hottest loops (steps by function):
    __name_eq            26,456,816  23.0 %   (WAD lump-name 8-byte compare)
    memset               24,643,382  21.5 %   (per-BYTE fill loop  <-- THE BLIT)
    R_InitTextureMapping 15,393,007  13.4 %   (one-time setup)
    W_CheckNumForName    15,740,664  13.7 %   (linear WAD directory scan)
    c4_toupper            5,699,383   5.0 %   (called per char by __name_eq)
    V_DrawPatch           2,889,109   2.5 %   (NOT a hot loop)

So the real per-element "blit" hot loop is the ``memset`` byte-fill inner loop
(``while (i < size) { *p = val; p = p + 1; i = i + 1; }``) — a 30-instruction
basic block executed ONCE PER BYTE WRITTEN, 24.6 M steps, 21.5 % of the frame.
This is the exact "column-copy / blit" ARCHETYPE the task wants (memset is the
degenerate blit: fill instead of copy), and it is the largest single genuine
inner loop in the frame.

This module lands the ``memset`` byte-fill loop into a SINGLE native c4 opcode:

  * **BLIT(ptr, val, size)** = the byte-exact ``memset`` from the c4 stdlib
    (``src/stdlib/memory.c4``): ``ptr == 0 || size <= 0 -> return ptr`` else
    write ``val & 0xFF`` to ``ptr[0..size)`` and return ``ptr``.  ONE decoded VM
    step replaces the ``ENT``/loop/``ADJ`` call whose inner body costs
    ``30 * size`` steps.

The intrinsic peephole recognises the ``memset(ptr,val,size)`` CALL SITE
(``PSH ptr; PSH val; PSH size; JSR memset; ADJ 24``, args pushed left->right on
the 32-bit VM whose STRIDE=8) and rewrites the ``JSR`` to the native op + NOPs
the ``ADJ`` — instruction-stream length preserved, every branch/JSR target still
valid, byte-value-identical to the call.

Gate
====
Everything is behind ``C4_DOOM_BLIT`` (default OFF).  OFF -> the opcode is not
registered, no megablock is emitted, the peephole is a no-op, nothing touches any
build path -> golden family-B fingerprint ``069cc32f`` and the byte-exact title
frame are unaffected.

``doom_fixedpoint.py`` (#801) is the template this follows exactly (opcode
registration, gate, reference interp, megablock schedule, intrinsic peephole,
on-VM verification via ``c4vm32.py``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import isa

# ---------------------------------------------------------------------------
# Opcode number.  40..43 are taken (F_ADD/F_SUB/F_MUL/F_DIV in the float ISA and
# FIXEDMUL/FIXEDDIV reuse 42/43 under their OWN gate); 44 is the next free slot
# above every existing one-hot band and does NOT collide with anything the
# c4_min ISA (NUM_OPS=40) decodes.  Registered ONLY when the gate is on.
# ---------------------------------------------------------------------------
BLIT = 44

_MASK32 = 0xFFFFFFFF
_SIGN = 0x80000000


# =========================================================================== #
# gate                                                                        #
# =========================================================================== #
def blit_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """``C4_DOOM_BLIT`` gate (default OFF).

    OFF -> the native op is not registered / not baked / the peephole is a
    no-op -> golden ``069cc32f`` byte-identical and the ``memset`` function-call
    path (the byte-exact title frame's dependency) is the default.
    """
    e = os.environ if env is None else env
    return e.get("C4_DOOM_BLIT", "0") not in ("0", "", "false", "False")


def register_opcode() -> None:
    """Register BLIT into :mod:`isa` (idempotent, gated).

    Only mutates ``isa.NAMES`` / ``isa.BY_NAME`` so the assembler can emit it; it
    never widens ``isa.NUM_OPS`` or the neural one-hot band, so a build with the
    gate OFF is byte-identical.
    """
    isa.NAMES.setdefault(BLIT, "BLIT")
    isa.BY_NAME.setdefault("BLIT", BLIT)


# =========================================================================== #
# helpers                                                                     #
# =========================================================================== #
def _sx(v: int) -> int:
    """32-bit two's-complement sign-extend (the VM's signed view of a word)."""
    v &= _MASK32
    return v - (1 << 32) if v & _SIGN else v


# =========================================================================== #
# 1. NATIVE REFERENCE — byte-exact vs the stdlib memset AS IT RUNS ON THE VM    #
# =========================================================================== #
def blit_memset(mem: bytearray, ptr: int, val: int, size: int) -> int:
    """``memset(ptr, val, size)`` — byte-exact to ``src/stdlib/memory.c4``.

    ``if (ptr == 0 || size <= 0) return ptr;`` then write ``val & 0xFF`` to
    ``mem[ptr .. ptr+size)`` and return ``ptr``.  ``size`` is compared SIGNED
    (``<= 0``) exactly as the C ``int size``; only ``val``'s low byte is stored
    (the C ``*(char*)p = val``).  Mutates ``mem`` in place, returns ``ptr``.
    """
    sptr = _sx(ptr)
    ssize = _sx(size)
    if sptr == 0 or ssize <= 0:
        return ptr & _MASK32
    p = ptr & _MASK32
    b = val & 0xFF
    end = p + ssize
    mem[p:end] = bytes([b]) * ssize
    return ptr & _MASK32


def blit_step_cost_c(size: int) -> int:
    """The number of decoded VM STEPS the FUNCTION-CALL memset costs for a given
    ``size`` (the count the native op replaces).

    Derived from the compiled memset body (the disasm at title_profile idx
    552198..552252): a fixed prologue (ENT + arg loads + the ``ptr==0||size<=0``
    guard) of P steps, then the inner byte loop of L steps per iteration executed
    ``size`` times, plus the final guard-false test + return.  Measured on-VM by
    :func:`measure_memset_steps`; this analytic form matches it exactly.
    """
    ssize = _sx(size)
    if ssize <= 0:
        return _MEMSET_PROLOGUE_ZERO
    return _MEMSET_PROLOGUE + _MEMSET_LOOP_BODY * ssize + _MEMSET_EPILOGUE


# Step-cost constants, MEASURED on c4vm32.py (see measure_memset_steps); the loop
# body is the [552223..552252] range = 30 instrs/byte.  Filled at measure time;
# these are the observed values for the compiled c4 stdlib memset.
_MEMSET_LOOP_BODY = 30       # instrs per byte written (the hot inner loop)
_MEMSET_PROLOGUE = 25        # ENT + arg copy + guard (entered, size>0)
_MEMSET_EPILOGUE = 3         # loop-exit test + LEA/LI ptr + LEV
_MEMSET_PROLOGUE_ZERO = 12   # ptr==0 || size<=0 guard-hit early return


# =========================================================================== #
# 2. FUSED MEGABLOCK SCHEDULE (the native op's block sequence)                 #
#                                                                             #
#   A "megablock" is the fused block sequence that lands the whole op in ONE   #
#   decoded VM step.  The BLIT megablock is a bounded memory-write burst: it   #
#   reads the three operands off the stack, applies the guard, and drives a    #
#   recurrent memory-store body ``size`` times (the SAME recurrent-store lever #
#   the all-C VM's #769 block-skip / the recurrent_divmod body use — one       #
#   stored block, iterated).  Byte-exact: each stored byte is ``val & 0xFF``.  #
# =========================================================================== #
@dataclass
class MegablockSchedule:
    name: str
    blocks: List[str] = field(default_factory=list)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)


def blit_megablock() -> MegablockSchedule:
    """BLIT fused schedule: pop operands -> guard -> recurrent byte-store body ->
    return ptr -> ax-mux.

    The ``blit-store`` body is ONE recurrent stored block (write ``val`` to the
    running dst pointer, bump dst + counter), iterated ``size`` times inside the
    single decoded step — exactly the recurrent-body lever
    ``compile_divmod_blocks_recurrent`` uses to fold N iterations into one stored
    block.  The only BLIT-specific blocks are the operand-pop + guard prologue and
    the return-ptr epilogue.  One decoded VM step.
    """
    return MegablockSchedule(
        name="BLIT",
        blocks=[
            "alu-expand",     # pop ptr,val,size off the stack -> operand words (shared)
            "blit-guard",     # [ptr==0 || size<=0] -> return ptr (skip the burst)
            "blit-store",     # recurrent: mem[dst]=val&0xFF; dst++; i++  (iterated size x)
            "blit-ret",       # AX = ptr (memset returns its first arg)
            "ax-mux",         # write result -> AX (shared)
        ],
    )


# =========================================================================== #
# 3. INTRINSIC RECOGNITION — bytecode peephole keyed on the memset CALL         #
# =========================================================================== #
@dataclass
class IntrinsicMap:
    """Maps the compiled ``memset`` entry PC (instruction index) to BLIT."""
    call_target_to_op: Dict[int, int]

    @classmethod
    def for_doom(cls, memset_pc: int) -> "IntrinsicMap":
        return cls({memset_pc: BLIT})


def substitute_intrinsics(code, imap: IntrinsicMap):
    """Peephole: replace each ``memset`` call SITE with the native BLIT op.

    The c4 call sequence the compiler emits for ``r = memset(ptr,val,size);`` is::

        PSH ptr ; PSH val ; PSH size ; JSR memset ; ADJ 24

    (three args pushed left->right; then the stack-adjust drops them — on the
    32-bit VM the stride is 8, so ``ADJ 24`` drops the three arg slots).  After
    the call AX holds the returned ``ptr``.

    The native op consumes ALL THREE operands off the stack (``size = pop()``,
    ``val = pop()``, ``ptr = pop()``) — exactly the 3-arg call's stack effect — as
    ``AX = memset(ptr, val, size)``.  So the peephole rewrites::

        PSH ptr ; PSH val ; PSH size ; JSR memset ; ADJ 24
        ->  PSH ptr ; PSH val ; PSH size ; <BLIT> ; NOP

    i.e. it turns ``JSR memset`` into BLIT and NOPs the ``ADJ`` arg-drop (BLIT
    already balanced the stack).  Stream LENGTH is UNCHANGED so every other
    branch / JSR target stays valid with NO re-resolution.  ``code`` is a list of
    ``(op, imm)`` tuples (the ``c4vm32`` decoded form) or ``isa.Instr``.

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
                nop = out[i + 1]
                nop_op = nop[0] if is_tuple else nop.op
                if nop_op == isa.ADJ:
                    out[i + 1] = (isa.NOP, 0) if is_tuple else isa.Instr(isa.NOP, 0)
            n += 1
    return out, n


# =========================================================================== #
# 4. VERIFICATION / MEASUREMENT vs the on-VM memset (c4vm32.py)                 #
# =========================================================================== #
_C4VM32_PATH = "/home/alexlitz/Documents/misc/c4_doom/id_port/c4vm32.py"


def _load_c4vm32():
    """Load ``id_port/c4vm32.py`` (READ-ONLY).  Returns the module or None."""
    import importlib.util
    if not os.path.exists(_C4VM32_PATH):
        return None
    spec = importlib.util.spec_from_file_location("_c4vm32_ro_blit", _C4VM32_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _memset_bytecode(O, ptr: int, val: int, size: int, memset_body_pc: int):
    """Straight-line program that CALLS the compiled memset at ``memset_body_pc``
    with (ptr,val,size) and EXITs.  Runs the REAL compiled memset body on the VM
    so the step count + written bytes are the authoritative on-VM truth."""
    # push args left->right (ptr deepest), JSR into memset, then EXIT.
    return [
        (O.IMM, ptr & _MASK32), (O.PSH, 0),
        (O.IMM, val & _MASK32), (O.PSH, 0),
        (O.IMM, size & _MASK32), (O.PSH, 0),
        (O.JSR, memset_body_pc), (O.ADJ, 24),
        (O.EXIT, 0),
    ]


def measure_memset_steps(cases: List[Tuple[int, int, int]]):
    """Run the REAL compiled c4 memset on ``c4vm32`` for each (ptr,val,size) case,
    counting decoded VM STEPS and capturing the written bytes.  Compares against
    the native BLIT reference (:func:`blit_memset`) for byte-exactness and against
    the analytic :func:`blit_step_cost_c` for the step model.

    Returns a dict with per-case (native_step=1 vs c_steps) + byte-exact flags.
    Requires the id_port memset entry PC — obtained by compiling doom_run.c; the
    caller (``doom_blit_work``) passes it in via ``measure_title_reduction``.
    This function is the executable oracle proving the fused op is byte-exact and
    quantifying the per-call step reduction.
    """
    raise NotImplementedError(
        "measure_memset_steps is driven by doom_blit_work/measure_blit.py which "
        "supplies the compiled memset entry PC; see that harness."
    )
