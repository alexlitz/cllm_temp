"""Faithful full-word c4 interpreter — the ``native ./c4`` reference for the battery.

The repo's shipped reference interpreters are each PARTIAL:
  * ``nibble_pure_forward_complete.ref_interpret`` — 8-bit-masked (LEA/PC in the
    stack wrap at 0xFF), so function-call frames and >255 PC targets alias.
  * ``nibble_runtime.ref_interpret_words`` — full 32-bit but has NO JSR/ENT/ADJ/LEV.
  * ``isa.interpret`` — 8-bit, no call frames.

None of them runs a general function-calling, string-using c4 program the way the
real ``./c4`` VM (and gcc) does.  This module is the missing faithful reference:
a byte-addressed, 32-bit-word c4 VM with the FULL control set (JSR/ENT/ADJ/LEV),
a data segment seeded from the compiler's string/global literals, and a bump-heap
so ``malloc`` works even when the program does NOT link the stdlib .c4 (we execute
MALC/FREE/MSET/MCMP as the same intrinsics the baked stdlib compiles to).

Semantics match c4.c / the repo COMPILER's ABI (NOT x86 exactly): ``sizeof(int)==8``,
pointers are 8 bytes, the stack cell is 8 bytes (PSH/JSR/ENT move SP by 8), and
``LEA``/``ENT``/``ADJ`` immediates are BYTE offsets (no ``*4`` scale) — this is the
ABI the compiler emits (verified: arg offsets ``16 + (n-1-idx)*8`` and locals
``-8k`` are byte offsets, LEA reads them directly).  The ALU wraps mod 2^32, DIV/MOD
are C-truncating with a 0-guard, SHR is arithmetic, LC is a signed-char load.  Where
these differ from gcc-on-x86 (only ``sizeof`` and the 8-byte int cell) the battery
records it as an EXPECTED c4-vs-x86 semantic gap, not a VM bug.

NOTE (fix-backlog signal): the production TRANSFORMER build uses a 4-byte cell with
a ``*4`` LEA/ENT scale, which is self-consistent for locals-only frames but does NOT
match this compiler's arg passing — so multi-arg function CALLS on the transformer
hit an ABI wall (classified in the battery, not fixed here).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from c4_min import isa

CELL = 8            # bytes per stack cell (PSH/JSR/ENT SP delta) — matches the compiler
SCALE = 1           # LEA/ENT/ADJ immediate is a byte offset (no *4)
SP_INIT = 1 << 20          # 1 MiB stack top (well clear of data_base 0x10000)
HEAP_BASE = 1 << 22        # bump heap above the stack
_MASK = 0xFFFFFFFF


def run(code: List[isa.Instr], data: Optional[List[int]] = None,
        data_base: int = 65536, max_steps: int = 2_000_000) -> Tuple[int, int]:
    """Execute ``code``; return ``(final_ax_32bit, steps)``.

    ``data`` is the compiler's data byte-array; ``data[i]`` seeds mem[data_base+i].
    """
    mem: Dict[int, int] = {}
    if data:
        for i, b in enumerate(data):
            if b:
                mem[data_base + i] = b & 0xFF
    heap = HEAP_BASE
    ax = pc = 0
    sp = bp = SP_INIT
    steps = 0

    def loadw(a: int) -> int:
        return sum(mem.get(a + i, 0) << (8 * i) for i in range(4))

    def storew(a: int, v: int) -> None:
        v &= _MASK
        for i in range(4):
            mem[a + i] = (v >> (8 * i)) & 0xFF

    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm & _MASK
        elif op == isa.LEA:
            ax = (bp + SCALE * imm) & _MASK
        elif op == isa.PSH:
            sp -= CELL; storew(sp, ax)
        elif op == isa.ADD:
            sp0 = loadw(sp); sp += CELL; ax = (sp0 + ax) & _MASK
        elif op == isa.SUB:
            sp0 = loadw(sp); sp += CELL; ax = (sp0 - ax) & _MASK
        elif op == isa.MUL:
            sp0 = loadw(sp); sp += CELL; ax = (sp0 * ax) & _MASK
        elif op == isa.DIV:
            sp0 = loadw(sp); sp += CELL; ax = ((sp0 // ax) if ax else 0) & _MASK
        elif op == isa.MOD:
            sp0 = loadw(sp); sp += CELL; ax = ((sp0 % ax) if ax else 0) & _MASK
        elif op in (isa.OR, isa.XOR, isa.AND):
            sp0 = loadw(sp); sp += CELL
            ax = (sp0 | ax) if op == isa.OR else (sp0 ^ ax) if op == isa.XOR else (sp0 & ax)
            ax &= _MASK
        elif op == isa.SHL:
            sp0 = loadw(sp); sp += CELL; ax = (sp0 << ax) & _MASK
        elif op == isa.SHR:
            sp0 = loadw(sp); sp += CELL
            sv = sp0 - (1 << 32) if sp0 & 0x80000000 else sp0     # arithmetic (signed)
            ax = (sv >> ax) & _MASK
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = loadw(sp); sp += CELL
            sv = v - (1 << 32) if v & 0x80000000 else v
            sa = ax - (1 << 32) if ax & 0x80000000 else ax
            r = {isa.EQ: v == ax, isa.NE: v != ax, isa.LT: sv < sa,
                 isa.GT: sv > sa, isa.LE: sv <= sa, isa.GE: sv >= sa}[op]
            ax = 1 if r else 0
        elif op == isa.LI:
            ax = loadw(ax)
        elif op == isa.LC:
            b = mem.get(ax, 0) & 0xFF
            ax = (b - 0x100 if b & 0x80 else b) & _MASK          # signed char load
        elif op == isa.SI:
            addr = loadw(sp); sp += CELL; storew(addr, ax)
        elif op == isa.SC:
            addr = loadw(sp); sp += CELL; mem[addr] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            sp -= CELL; storew(sp, i + 1); pc = imm
        elif op == isa.ENT:
            sp -= CELL; storew(sp, bp); bp = sp; sp -= SCALE * imm
        elif op == isa.ADJ:
            sp += SCALE * imm
        elif op == isa.LEV:
            sp = bp; bp = loadw(sp); pc = loadw(sp + CELL); sp += 2 * CELL
        elif op == isa.MALC:
            n = loadw(sp); sp += CELL
            n = (n + 7) & ~7                                     # 8-byte align
            ax = heap; heap += n
        elif op == isa.FREE:
            sp += CELL                                           # bump heap: free is a no-op
        elif op == isa.MSET:
            n = ax; c = loadw(sp); sp += CELL; p = loadw(sp); sp += CELL
            for k in range(n):
                mem[p + k] = c & 0xFF
            ax = p & _MASK
        elif op == isa.MCMP:
            n = ax; pb = loadw(sp); sp += CELL; pa = loadw(sp); sp += CELL
            ax = 0
            for k in range(n):
                d = (mem.get(pa + k, 0) - mem.get(pb + k, 0)) & _MASK
                if d:
                    ax = d; break
        elif op == isa.PRTF:
            pass                                                 # no visible-output channel here
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(f"native_c4: op {isa.NAMES.get(op, op)} @pc={i}")
    return ax & _MASK, steps
