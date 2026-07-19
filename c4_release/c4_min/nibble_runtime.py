"""c4_min NIBBLE RUNTIME LIBRARY — malloc / free / memset / memcmp.

The C4 standard-library routines are **compiled from C into VM bytecode** and run
ENTIRELY NEURALLY — they are NOT tool calls (BLOG_SPEC §"Memory Allocation and
Freeing" (687), §"Memset, Memcmp and Memcpy" (747)):

    Notably, runtime library functions like malloc, free, memset, and memcmp are
    NOT tool calls.  They're compiled from C into the VM's bytecode and execute
    entirely neurally — malloc is just LEA/LI/ADD/SI, memset is a loop of SC
    instructions, and so on.

So each routine is a **baked bytecode subroutine**: a straight-line / looping
snippet of the base 8-bit ISA ops (IMM, LEA, PSH, ADD, SUB, MUL, LI, SI, SC, LC,
LT, BZ, JMP) that runs on the same neural VM step-block the rest of ``c4_min``
compiles.  "Baking" is literally: get the bytecode, place it in the code segment,
and let the network's opcode-fetch FFNs execute it (§849-853).

This module is the port of the original ``nibble_runtime`` (branch ``c4min-final``)
onto the ONE canonical unified full-op interpreter
(``nibble_pure_forward_complete.build_pure_forward_complete_model``).  Two things
change vs the original reference-only port:

  1. **Addressing width.**  The unified §Memory KV head keys stores/loads on the
     FULL 32-bit address (``blogspec_memory.ADDR_BITS = 32``).  The heap base
     ``0x30008`` and bump cell ``0x30000`` therefore ride the neural CAM directly
     — PROVIDED the LI/LC address QUERY is expanded from all 8 register nibbles
     (32 bits), not just the low byte.  ``lib_neural.compile_mem_prep_addr32``
     supplies that widening; ``build_lib_model`` there builds the model with it.

  2. **Value width.**  The neural memory carries a full 32-bit VALUE per cell (the
     store frame's 4 ``MEM_VAL`` bytes; the CAM writes 8 value nibbles into AX),
     so a bump pointer stored with SI and reloaded with LI round-trips WHOLE when
     the driver runs at ``mask=0xFFFFFFFF``.  The byte-truncating ``ref_interpret``
     is NOT the oracle for the library; :func:`ref_interpret_words` (word-width
     SI/SC/LI/LC memory) is.

Heap layout (unchanged from the reference; chosen to clear the stack, which
descends from ``SP_INIT = 0x10000``, and the data segment):

  * ``BUMP_CELL = 0x30000`` — fixed cell holding the next free address.
  * ``HEAP_BASE = 0x30008`` — first 4-aligned address the bump allocator hands out.
  * scratch cells ``0x2FFxx`` — the loop subroutines' C locals.

The five ISA syscall aliases MALC/FREE/MSET/MCMP (ops 34-37) are added to
``c4_min.isa`` so a word-width reference VM (:func:`ref_interpret_words`) can
execute them as intrinsics and prove the baked bytecode is EQUIVALENT to the
"compiled from C" contract — exactly as the original reference did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import isa

# ---------------------------------------------------------------------------
# Opcode ids we compose from (base 8-bit ISA + MUL for index math).  All are
# ALREADY decoded + dispatched by the unified model (base ops + LI/LC/SI/SC +
# cmp LT + MUL), so the library needs NO new neural op — it is pure bytecode.
# ---------------------------------------------------------------------------
LEA, IMM, JMP, JSR, BZ, BNZ = isa.LEA, isa.IMM, isa.JMP, isa.JSR, isa.BZ, isa.BNZ
LI, LC, SI, SC, PSH = isa.LI, isa.LC, isa.SI, isa.SC, isa.PSH
LT = isa.LT
ADD, SUB, MUL = isa.ADD, isa.SUB, isa.MUL
EXIT = isa.HALT

#: Data segment base a program's static literals load at.
DATA_BASE = 0x10000

#: Bump allocator layout.  Above the stack (descends from SP_INIT = 0x10000) and
#: any data segment so MALC and static data never collide.  The bump cell holds
#: the next free address; the heap begins 4-aligned just above it.
BUMP_CELL = 0x30000          #: fixed cell holding the next free address
HEAP_BASE = 0x30008          #: first address the bump allocator hands out (aligned)
ALIGN = 4                    #: §688 "incrementing addresses by 4"

#: Scratch (spill) cells the loop subroutines use for their C locals.
_SC_I = 0x2FFE8
_SC_RET = 0x2FFD0            #: malloc's saved return value


def _align_up(n: int, a: int) -> int:
    """Round ``n`` up to the next multiple of ``a`` (a power of two here)."""
    return (n + a - 1) & ~(a - 1)


# ---------------------------------------------------------------------------
# Tiny label-resolving assembler.  A branch target (JMP/BZ/BNZ) is a *label
# name* (str) that resolves to the target instruction index; the reference VM
# accepts an instruction index directly as a branch target, so index targets
# are byte-exact.  ``assemble_instrs`` yields ``isa.Instr`` for the unified
# model's overlay (which reads ``(op, imm)`` per code slot).
# ---------------------------------------------------------------------------

Instr = Tuple[int, object]          # (opcode, imm | label-name)


@dataclass
class Asm:
    """Accumulates (op, imm|label) instructions with named labels."""

    code: List[Instr] = field(default_factory=list)
    labels: Dict[str, int] = field(default_factory=dict)

    def label(self, name: str) -> "Asm":
        if name in self.labels:
            raise ValueError(f"duplicate label {name!r}")
        self.labels[name] = len(self.code)
        return self

    def emit(self, op: int, imm: object = 0) -> "Asm":
        self.code.append((op, imm))
        return self

    def splice(self, body: "Asm") -> "Asm":
        """Append another Asm body, RESOLVING its internal labels to absolute
        instruction indices at the current offset (so a subroutine's internal
        branches stay correct when placed mid-program).  Label-free instructions
        pass through unchanged.  The body's label NAMES are not merged (they were
        internal); build the composite's own labels with :meth:`label` as needed."""
        off = len(self.code)
        for op, imm in body.code:
            if isinstance(imm, str):
                if imm not in body.labels:
                    raise KeyError(f"unresolved label {imm!r} in spliced body")
                imm = body.labels[imm] + off
            self.code.append((int(op), imm))
        return self

    # Readable wrappers.
    def imm(self, v: int):  return self.emit(IMM, v)
    def lea(self, v: int):  return self.emit(LEA, v)
    def psh(self):          return self.emit(PSH)
    def add(self):          return self.emit(ADD)
    def sub(self):          return self.emit(SUB)
    def mul(self):          return self.emit(MUL)
    def li(self):           return self.emit(LI)
    def lc(self):           return self.emit(LC)
    def si(self):           return self.emit(SI)
    def sc(self):           return self.emit(SC)
    def lt(self):           return self.emit(LT)
    def jmp(self, lbl):     return self.emit(JMP, lbl)
    def bz(self, lbl):      return self.emit(BZ, lbl)
    def bnz(self, lbl):     return self.emit(BNZ, lbl)
    def exit_(self):        return self.emit(EXIT)

    def resolve(self) -> List[Tuple[int, int]]:
        """Resolve labels -> instruction indices; return ``[(op, imm), ...]``."""
        out: List[Tuple[int, int]] = []
        for op, imm in self.code:
            if isinstance(imm, str):
                if imm not in self.labels:
                    raise KeyError(f"unresolved label {imm!r}")
                imm = self.labels[imm]
            out.append((int(op), int(imm) & 0xFFFFFFFF))
        return out

    def instrs(self) -> List[isa.Instr]:
        """Resolve to a list of ``isa.Instr`` for the unified model / oracle."""
        return [isa.Instr(op, imm) for op, imm in self.resolve()]


# ---------------------------------------------------------------------------
# The universal store idiom + a load helper.  These are the only two primitives
# every subroutine is built from.
# ---------------------------------------------------------------------------


def _store(a: Asm, addr: int, value: Callable[[Asm], None], *, byte: bool = False) -> None:
    """Bake ``*(addr) = value`` — the C4 store idiom.

    Push the constant ADDRESS first, then let ``value(a)`` produce the value into
    AX in a *stack-neutral* way, then SI (word) / SC (byte).  Because the address
    is pushed first and the value computation nets zero stack change, the address
    is exactly on top when SI/SC pops it — no swaps, no spills.
    """
    a.imm(addr).psh()
    value(a)
    a.emit(SC if byte else SI)


def _load(a: Asm, addr: int) -> None:
    """AX = ``*(addr)`` (word).  ``IMM addr; LI``."""
    a.imm(addr).li()


def _add_const(a: Asm, addr: int, delta: int) -> None:
    """``*(addr) += delta`` (word)."""
    _store(a, addr, lambda a: (_load(a, addr), a.psh(), a.imm(delta), a.add()))


# ---------------------------------------------------------------------------
# MALC — bump allocator (§687-688).  Returns increasing 4-aligned addresses.
# ---------------------------------------------------------------------------


def emit_malloc(size: int) -> Asm:
    """Emit MALC(size): a bump-allocator gadget (LEA/LI/ADD/SI, §688).

    C equivalent::

        int malloc(int n) {          // n already rounded up to a multiple of 4
            if (*bump == 0)          // first use -> initialise heap base
                *bump = HEAP_BASE;
            ret   = *bump;           // the address we hand out
            *bump = *bump + n;       // bump the pointer
            return ret;
        }
    """
    n = _align_up(int(size), ALIGN)
    a = Asm()
    _load(a, BUMP_CELL)                 # AX = *bump
    a.bnz("have")                       # already initialised -> skip
    _store(a, BUMP_CELL, lambda a: a.imm(HEAP_BASE))
    a.label("have")
    _store(a, _SC_RET, lambda a: _load(a, BUMP_CELL))     # ret = *bump (spill)
    _add_const(a, BUMP_CELL, n)                            # *bump = *bump + n
    _load(a, _SC_RET)                                      # return ret
    return a


# ---------------------------------------------------------------------------
# FREE — zero-overwrite = eviction under softmax1 (§689-691).
# ---------------------------------------------------------------------------


def emit_free(ptr: int) -> Asm:
    """Emit FREE(ptr): ``*(int*)ptr = 0`` — a single store of 0.

    §689-691: overwriting with zero *is* the free; under softmax1 a zero value
    embedding is the attention default (ZFOD), so a later load of a freed cell
    reads 0.  At the ISA level this is a single SI of 0.
    """
    a = Asm()
    _store(a, ptr, lambda a: a.imm(0))
    return a


# ---------------------------------------------------------------------------
# MSET — memset as a loop of SC (§747-749).
# ---------------------------------------------------------------------------


def emit_memset(p: int, c: int, n: int) -> Asm:
    """Emit MSET(p, c, n): ``for i in 0..n: *(char*)(p+i) = c; return p``."""
    c &= 0xFF
    a = Asm()
    _store(a, _SC_I, lambda a: a.imm(0))            # i = 0
    a.label("top")
    _load(a, _SC_I); a.psh(); a.imm(n); a.lt()      # AX = (i < n)
    a.bz("done")
    a.imm(p).psh(); _load(a, _SC_I); a.add()        # AX = p + i
    a.psh()                                          # [p+i]  (SC address)
    a.imm(c).emit(SC)                                # *(p+i) = c
    _add_const(a, _SC_I, 1)                          # i += 1
    a.jmp("top")
    a.label("done")
    a.imm(p).exit_()                                 # return p
    return a


# ---------------------------------------------------------------------------
# MCMP — memcmp as a loop of LC + subtract (§747-749).
# ---------------------------------------------------------------------------


def emit_memcmp(pa: int, pb: int, n: int) -> Asm:
    """Emit MCMP(pa, pb, n): first differing ``a[i]-b[i]`` (u32), else 0."""
    a = Asm()
    _store(a, _SC_I, lambda a: a.imm(0))            # i = 0
    a.label("top")
    _load(a, _SC_I); a.psh(); a.imm(n); a.lt()      # AX = (i < n)
    a.bz("equal")
    a.imm(pa).psh(); _load(a, _SC_I); a.add(); a.lc()   # AX = a[i]
    a.psh()                                          # [a[i]]
    a.imm(pb).psh(); _load(a, _SC_I); a.add(); a.lc()   # AX = b[i]
    a.sub()                                          # AX = a[i] - b[i]
    a.bz("next")                                     # equal -> continue
    a.exit_()                                        # differ -> return a[i]-b[i]
    a.label("next")
    _add_const(a, _SC_I, 1)                          # i += 1
    a.jmp("top")
    a.label("equal")
    a.imm(0).exit_()                                 # return 0
    return a


# ---------------------------------------------------------------------------
# WORD-WIDTH reference oracle (the library's oracle).  Unlike
# ``nibble_pure_forward_complete.ref_interpret`` (byte-valued SI/SC/LI/LC to
# match the 8-bit value path), the library uses full 32-bit VALUES in memory —
# a bump pointer is a 32-bit word.  The unified NEURAL model carries a 32-bit
# value per cell (4 MEM_VAL bytes / 8 value nibbles), so word-width memory is
# the faithful oracle when the driver runs at ``mask = 0xFFFFFFFF``.  LC/SC are
# still BYTE accesses (loading/storing one char) — that matches both the C
# semantics and the neural LC/SC, which the memset/memcmp loops rely on.
# ---------------------------------------------------------------------------


def ref_interpret_words(code: List[isa.Instr], max_steps: int = 5000,
                        data: Optional[Dict[int, int]] = None) -> Tuple[int, Dict[int, int]]:
    """Word-width reference VM.  Returns ``(final_AX, memory)``.

    SI/LI are 32-bit word store/load; SC/LC are byte store/load.  The stack is a
    byte-addressed region descending from ``SP_INIT`` (matching the neural VM);
    ``data`` seeds the read-only data segment (``{byte_addr: byte}``).  MALC/FREE/
    MSET/MCMP (ops 34-37) are ALSO executed as intrinsics so the baked bytecode
    can be diffed against the "compiled from C" contract.
    """
    from .nibble_pure_forward import SP_INIT
    mem: Dict[int, int] = dict(data or {})
    stack: Dict[int, int] = {}
    ax = pc = 0
    sp = bp = SP_INIT
    steps = 0

    def load_word(addr: int) -> int:
        return sum(mem.get(addr + i, 0) << (8 * i) for i in range(4))

    def store_word(addr: int, v: int) -> None:
        for i in range(4):
            mem[addr + i] = (v >> (8 * i)) & 0xFF

    def push(v):
        nonlocal sp
        sp -= 4; stack[sp] = v & 0xFFFFFFFF

    def pop():
        nonlocal sp
        v = stack.get(sp, 0); sp += 4; return v

    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        op, imm = code[pc].op, code[pc].imm
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFFFFFFFF
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFFFFFFFF
        elif op == isa.PSH:
            push(ax)
        elif op == isa.ADD:
            ax = (pop() + ax) & 0xFFFFFFFF
        elif op == isa.SUB:
            ax = (pop() - ax) & 0xFFFFFFFF
        elif op == isa.MUL:
            ax = (pop() * ax) & 0xFFFFFFFF
        elif op == isa.LT:
            ax = 1 if pop() < ax else 0
        elif op == isa.LI:
            ax = load_word(ax)
        elif op == isa.LC:
            ax = mem.get(ax, 0) & 0xFF
        elif op == isa.SI:
            store_word(pop(), ax)
        elif op == isa.SC:
            mem[pop()] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == MALC:
            n = _align_up(pop(), ALIGN)
            if load_word(BUMP_CELL) == 0:
                store_word(BUMP_CELL, HEAP_BASE)
            ax = load_word(BUMP_CELL)
            store_word(BUMP_CELL, ax + n)
        elif op == FREE:
            store_word(pop(), 0)
        elif op == MSET:
            n = ax; c = pop() & 0xFF; p = pop()
            for k in range(n):
                mem[p + k] = c
            ax = p
        elif op == MCMP:
            n = ax; pb = pop(); pa = pop()
            ax = 0
            for k in range(n):
                d = (mem.get(pa + k, 0) - mem.get(pb + k, 0)) & 0xFFFFFFFF
                if d:
                    ax = d; break
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
    return ax & 0xFFFFFFFF, mem


# ---------------------------------------------------------------------------
# ISA syscall aliases (intrinsics) — for the equivalence diff only.  MALC/FREE/
# MSET/MCMP execute the SAME memory effect as the baked bytecode; proving the
# two agree demonstrates the "compiled from C, not a tool call" contract.
# ---------------------------------------------------------------------------
MALC = isa.MALC if hasattr(isa, "MALC") else 34
FREE = isa.FREE if hasattr(isa, "FREE") else 35
MSET = isa.MSET if hasattr(isa, "MSET") else 36
MCMP = isa.MCMP if hasattr(isa, "MCMP") else 37


# ---------------------------------------------------------------------------
# Chain helper: place N subroutine bodies back-to-back in one image, shifting
# each body's branch targets by the running offset, then EXIT.
# ---------------------------------------------------------------------------


def chain(*bodies: Asm, halt: bool = True) -> List[isa.Instr]:
    """Concatenate ``bodies`` into one image (each internally label-resolved
    relative to its own start, shifted by the running offset), optional trailing
    HALT.  Returns a list of ``isa.Instr``."""
    out: List[Tuple[int, int]] = []
    for b in bodies:
        off = len(out)
        for op, imm in b.code:
            if isinstance(imm, str):
                imm = b.labels[imm] + off
            out.append((int(op), int(imm) & 0xFFFFFFFF))
    if halt:
        out.append((EXIT, 0))
    return [isa.Instr(op, imm) for op, imm in out]
