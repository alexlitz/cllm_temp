"""codegen_fuse.py — the MEMORY-OPERAND-ALU codegen peephole (C4_CODEGEN_FUSE).

Why this module exists
======================
``nibble_exact_steps`` proved the model can execute a single fused ``<OP>M [addr]``
opcode (ADDM/SUBM/MULM/DIVM/MODM = 41-45) that computes ``AX = mem[addr] <op> AX``
in ONE ``model.forward`` — replacing the base-ISA four-instruction

    <load a into AX> ; PSH ; <compute b into AX> ; <OP>

lowering the ``src.compiler`` c4 codegen emits for every ``a <op> b`` whose LEFT
operand ``a`` is a memory read.  But the compiler NEVER emits ``<OP>M`` — it is a
strict single-pass recursive-descent emitter with no IR, so no real program
benefits.  This module is the missing codegen pass: a POST-EMISSION peephole over
the flat bytecode word list that folds the pattern to the fused opcode, gated
behind ``C4_CODEGEN_FUSE`` (default OFF) so the default compilation — and the
neural golden — are byte-identical.

The fold (byte-exact of RESULT)
==============================
c4 lowers ``a <op> b`` as:  compile ``a`` -> AX ; ``PSH`` ; compile ``b`` -> AX ;
``<OP>``.  The ``<OP>`` (``isa.interpret``) computes ``pop() <op> AX`` = ``a <op>
b`` (popped = left ``a``, AX = right ``b``).  ``<OP>M [addr]`` computes
``mem[addr] <op> AX`` — the SAME operand order, with ``mem[addr]`` playing the role
of the popped left operand.  So when ``a`` is a PURE absolute-address load

    IMM addr ; LI/LC ; PSH        # push a = mem[addr]

and the stack-balanced code that computes ``b`` into AX is followed by ``<OP>``, we
delete the three-instruction left-operand ``IMM addr; LI/LC; PSH`` and replace the
``<OP>`` with ``<OP>M addr``.  AX at the ``<OP>`` time is exactly ``b`` (the right
operand), so ``<OP>M`` computes ``mem[addr] <op> b`` = ``a <op> b`` — identical.

Scope / honest limits (see MEASURE_CODEGEN_FUSE.md)
==================================================
* ABSOLUTE address only.  ``<OP>M``'s ``imm`` is an ABSOLUTE memory address
  (``mem.get(imm)`` in the ref / the frame-carried address in the model).  A
  frame-relative ``LEA off`` local's absolute address is ``bp + off`` — NOT known at
  compile time — so a ``LEA off; LI; PSH; ...; <OP>`` left operand is NOT foldable
  without a NEW frame-relative ``<OP>M`` addressing mode (assessed, not built: it
  needs a model change, out of scope for a pure-codegen, byte-neutral pass).  Only
  GLOBAL / string-literal operands (``IMM addr``) fold.
* The intervening ``b`` code must be STACK-BALANCED and STRAIGHT-LINE between the
  PSH and its matching ``<OP>`` (the ``<OP>`` that fires when the stack returns to
  the pushed operand's level).  A branch/call/label crossing the span aborts the
  fold (the pushed value could be consumed on another control-flow path).
* NO STORE in the span.  ``PSH`` snapshots the VALUE ``mem[addr]`` at push time;
  ``<OP>M`` RE-READS ``mem[addr]`` at op time.  A store (``SI``/``SC``) in the ``b``
  code could alias ``addr`` (``r = g + (g = 2)``), diverging the snapshot from the
  re-read, so ANY store in the span conservatively aborts the fold.
* The left operand must be a PURE load: ``IMM addr`` (a constant address) directly
  feeding ``LI``/``LC`` then ``PSH``.  A computed / pointer-dereferenced address
  (``... ; LI`` where AX is a runtime value) is NOT an ``IMM``-addressed load.
* ADDRESS MASKING.  In both the byte-masked census/model VM and the ``<OP>M`` ref,
  ``IMM addr`` FIRST masks the constant to the value width (0xFF) before ``LI``
  reads ``mem[AX]`` — so ``IMM addr; LI`` loads ``mem[addr & 0xFF]``, NOT
  ``mem[addr]``.  ``<OP>M imm`` reads ``mem[imm]`` at full width.  To be byte-exact
  the fold therefore emits ``<OP>M (addr & IMM_ADDR_MASK)`` — the SAME effective
  load address the deleted ``IMM; LI`` produced.  ``IMM_ADDR_MASK`` defaults to 0xFF
  (the value mask the c4_min VMs use); a full-width VM would set it to 0xFFFFFFFF.

Relocation
==========
Each fold deletes 3 instructions, so every downstream PC target (JMP/JSR/BZ/BNZ
immediate) shifts.  The pass rebuilds the code with a relocation map and re-patches
every control-flow immediate, so the folded program is control-flow-identical and
byte-exact of result vs the original compilation.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

# c4 ``Op`` values (match ``src.compiler.Op`` / ``c4_min.isa``).
LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
OR, XOR, AND = 14, 15, 16
EQ, NE, LT, GT, LE, GE = 17, 18, 19, 20, 21, 22
SHL, SHR = 23, 24
ADD, SUB, MUL, DIV, MOD = 25, 26, 27, 28, 29
JSR_LIKE = frozenset({JMP, JSR, BZ, BNZ})   # ops whose imm is a PC target

# The folded memory-operand-ALU opcodes (match ``nibble_exact_steps``).
ADDM, SUBM, MULM, DIVM, MODM = 41, 42, 43, 44, 45
_FOLD_OP = {ADD: ADDM, SUB: SUBM, MUL: MULM, DIV: DIVM, MOD: MODM}
_ALU = frozenset(_FOLD_OP.keys())

# Per-op net stack-slot effect (for matching a PSH to its consuming <OP>).  A
# straight-line window only; branches/calls abort the span so their effects never
# matter here.  PSH pushes; every binary consumer pops one; SI/SC pop the address.
_BINARY_CONSUMERS = _ALU | {OR, XOR, AND, SHL, SHR, EQ, NE, LT, GT, LE, GE}
_STACK_POP1 = _BINARY_CONSUMERS | {SI, SC}
# ops that break a straight-line stack-balanced span (control flow / frame exit).
_SPAN_BREAKERS = frozenset({JMP, JSR, BZ, BNZ, LEV})

# ``IMM addr`` masks the constant to the VALUE width before ``LI`` reads ``mem[AX]``
# (both the byte-masked census/model VM and the ``<OP>M`` ref use 0xFF), so a folded
# ``<OP>M`` must read ``mem[addr & IMM_ADDR_MASK]`` to load the SAME cell the deleted
# ``IMM addr; LI`` did.  Overridable via C4_CODEGEN_FUSE_ADDR_MASK for a full-width VM.
_DEFAULT_ADDR_MASK = 0xFF


def _addr_mask() -> int:
    return int(os.environ.get("C4_CODEGEN_FUSE_ADDR_MASK", str(_DEFAULT_ADDR_MASK)), 0)


def codegen_fuse_enabled() -> bool:
    """The memory-operand-ALU codegen peephole (C4_CODEGEN_FUSE).  DEFAULT OFF.

    OFF -> ``compile_c`` output is unchanged (the golden / all runners see the
    identical bytecode).  ON -> ``fuse_bytecode`` rewrites the emitted words,
    folding ``IMM addr; LI/LC; PSH; <b>; <OP>`` -> ``<b>; <OP>M addr``."""
    return os.environ.get("C4_CODEGEN_FUSE", "0") != "0"


def _decode(word: int) -> Tuple[int, int]:
    return int(word) & 0xFF, int(word) >> 8


def _encode(op: int, imm: int) -> int:
    return int(op) + (int(imm) << 8)


def _stack_delta(op: int) -> int:
    if op == PSH:
        return +1
    if op in _STACK_POP1:
        return -1
    return 0


def find_folds(code: List[int]) -> List[Tuple[int, int, int, int]]:
    """Return the foldable sites as ``(imm_idx, li_idx, psh_idx, op_idx)`` tuples.

    A site is:  ``IMM addr`` @ imm_idx ; ``LI``/``LC`` @ li_idx=imm_idx+1 ;
    ``PSH`` @ psh_idx=imm_idx+2 ; ... stack-balanced straight-line ``b`` ... ;
    ``<OP>`` @ op_idx (the ALU op that pops THIS pushed value).  Non-overlapping,
    scanned left-to-right; the ``b`` span may itself contain OTHER (nested) folds,
    which a later fixpoint pass picks up."""
    n = len(code)
    # PC targets that land inside a span abort it (the pushed value could be
    # consumed on another control-flow path).
    targets = set()
    for w in code:
        op, imm = _decode(w)
        if op in JSR_LIKE:
            targets.add(imm)

    sites: List[Tuple[int, int, int, int]] = []
    i = 0
    while i + 2 < n:
        op0, _ = _decode(code[i])
        op1, _ = _decode(code[i + 1])
        op2, _ = _decode(code[i + 2])
        if op0 == IMM and op1 in (LI, LC) and op2 == PSH:
            # scan forward for the matching <OP> (stack returns to this level).
            depth = 0
            j = i + 3
            matched = -1
            while j < n:
                if j in targets:
                    break                     # a label lands mid-span -> abort
                opj, _ = _decode(code[j])
                if opj in _SPAN_BREAKERS:
                    break                     # control flow / frame exit -> abort
                if opj in (SI, SC):
                    break                     # a store could alias mem[addr] -> abort
                if opj in _ALU and depth == 0:
                    matched = j               # this ALU pops our pushed operand
                    break
                depth += _stack_delta(opj)
                if depth < 0:
                    break                     # underflow -> our push already gone
                j += 1
            if matched >= 0:
                sites.append((i, i + 1, i + 2, matched))
                # skip past this site's push to avoid overlapping the same PSH;
                # nested folds inside the b-span are found on the next fixpoint pass.
                i += 3
                continue
        i += 1
    return sites


def _apply_folds_once(code: List[int]) -> Tuple[List[int], int]:
    """Apply every NON-OVERLAPPING fold found in one pass, rebuild with a
    relocation map, re-patch control-flow immediates.  Returns ``(new_code,
    n_folded)``."""
    sites = find_folds(code)
    if not sites:
        return list(code), 0

    # choose a non-overlapping subset: a site owns [imm_idx .. op_idx].  Greedy
    # left-to-right by imm_idx; skip any site whose range overlaps an accepted one.
    sites.sort(key=lambda s: s[0])
    chosen: List[Tuple[int, int, int, int]] = []
    last_end = -1
    for s in sites:
        imm_idx, _, _, op_idx = s
        if imm_idx > last_end:
            chosen.append(s)
            last_end = op_idx

    amask = _addr_mask()
    delete = set()
    fold_at: Dict[int, int] = {}          # op_idx -> addr to fold in
    for imm_idx, li_idx, psh_idx, op_idx in chosen:
        _, addr = _decode(code[imm_idx])
        delete.update((imm_idx, li_idx, psh_idx))
        # the deleted ``IMM addr; LI`` loaded ``mem[addr & value_mask]`` (IMM masks
        # the constant to the value width first) -> fold the SAME effective address.
        fold_at[op_idx] = addr & amask

    # build old-index -> new-index relocation (deleted indices map to the next
    # surviving index, so a branch that (never legally) targets a deleted slot
    # still lands on the following live instruction).
    n = len(code)
    new_index = [0] * (n + 1)
    k = 0
    for old in range(n):
        new_index[old] = k
        if old not in delete:
            k += 1
    new_index[n] = k                       # one-past-end (JMP to code end)

    out: List[int] = []
    for old in range(n):
        if old in delete:
            continue
        op, imm = _decode(code[old])
        if old in fold_at:
            op = _FOLD_OP[op]
            imm = fold_at[old]             # ABSOLUTE operand address rides imm
        elif op in JSR_LIKE:
            imm = new_index[imm] if 0 <= imm <= n else imm
        out.append(_encode(op, imm))
    return out, len(chosen)


def fuse_bytecode(code: List[int]) -> Tuple[List[int], int]:
    """Fold ``code`` (a c4 word list) to a fixpoint.  Returns ``(folded_code,
    total_folds)``.  Byte-exact of RESULT vs ``code`` (control-flow-preserving).

    Idempotent when no site remains.  Runs to a fixpoint so a fold inside another
    fold's ``b``-span (nested ``a <op> (c <op> d)``) is also caught."""
    total = 0
    cur = list(code)
    while True:
        cur, n = _apply_folds_once(cur)
        if n == 0:
            break
        total += n
    return cur, total


def compile_c_fused(source: str, link_stdlib: bool = True):
    """``src.compiler.compile_c`` + the memory-operand fold when C4_CODEGEN_FUSE is
    on.  Returns ``(code, data)`` exactly like ``compile_c``; the code is the folded
    word list (or the unchanged one when the flag is OFF)."""
    from src.compiler import compile_c
    code, data = compile_c(source, link_stdlib=link_stdlib)
    if codegen_fuse_enabled():
        code, _ = fuse_bytecode(code)
    return code, data
