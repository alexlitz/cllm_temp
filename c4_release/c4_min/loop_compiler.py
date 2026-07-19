"""A VARIABLE-LENGTH loop compiler in c4_min bytecode (uses the moving-pointer
EMITP from nibble_fetch_dedup).

Grammar: a single-operator left-associative integer chain over single ASCII
digits, ``d0 op d1 op d2 ... op dk``, null-terminated (a 0 byte ends the source).
``op`` is ``+`` (ADD) or ``*`` (MUL); the compiler reads the operator ONCE (the
first operator char) and applies it to the whole chain (c4's expr for a single
precedence level, left-assoc).  It LOOPS over the source, so the number of
produced instructions grows with the input — the compiler bytecode is FIXED-size,
the produced program is variable-size, written at the runtime cursor OUT_PTR.

Produced code (byte-identical to real c4's arithmetic core for a same-op chain):
    IMM d0 ; [ PSH ; IMM di ; <OP> ]*  ; HALT
e.g. ``2+3+4+5`` -> IMM2;PSH;IMM3;ADD;PSH;IMM4;ADD;PSH;IMM5;ADD;HALT.

State layout used by the compiler bytecode:
    MEM[0] = source cursor i (index into SRC).
The cursor advances by 2 per operand (skip the digit + the operator).

Stack discipline: every helper is stack-BALANCED (net SP change 0) except the
produced-value PSH inside EMITP (which EMITP itself pops).
"""
from __future__ import annotations

from typing import List, Tuple

from . import isa
from . import nibble_compiler as C
from . import nibble_fetch_dedup as D

EMIT = C.EMIT
EMITP = D.EMITP
_PLUS = ord("+")   # 43
_STAR = ord("*")   # 42


class _Asm:
    def __init__(self):
        self.code: List[list] = []
        self.labels: dict = {}

    def emit(self, op: int, imm=0) -> int:
        self.code.append([op, imm]); return len(self.code) - 1

    def label(self, name: str):
        self.labels[name] = len(self.code)

    def resolve(self) -> List[Tuple[str, int]]:
        out = []
        for op, imm in self.code:
            if isinstance(imm, str):
                imm = self.labels[imm]
            name = ("EMIT" if op == EMIT else
                    "EMITP" if op == EMITP else isa.NAMES[op])
            out.append((name, imm))
        return out


# ---- stack-balanced primitives over MEM[0] (the source cursor) ----

def _load_cursor(a: _Asm):
    """AX := MEM[0]  (the cursor).  Stack-neutral."""
    a.emit(isa.IMM, 0); a.emit(isa.LI, 0)


def _bump_cursor(a: _Asm, k: int):
    """MEM[0] += k.  Stack-neutral.  Uses the SI(addr=STACK0, val=AX) form:
    push the address 0, compute value = MEM[0]+k into AX (stack-neutral over the
    pushed address), SI, then drop the address with a dummy pop."""
    a.emit(isa.IMM, 0); a.emit(isa.PSH, 0)          # STACK0 := 0 (the address); sp+1
    a.emit(isa.IMM, 0); a.emit(isa.LI, 0)           # AX := MEM[0]
    a.emit(isa.PSH, 0); a.emit(isa.IMM, k); a.emit(isa.ADD, 0)   # AX := MEM[0]+k (sp back)
    a.emit(isa.SI, 0)                                # MEM[STACK0=0] := AX
    # drop the address 0 still on the stack: pop via ADD into a scratch (AX+=0 pops it)
    a.emit(isa.ADD, 0)                              # AX := 0 + AX ; sp-1  (stack balanced)


def _read_src_at_cursor(a: _Asm):
    """AX := src[MEM[0]].  Stack-neutral."""
    _load_cursor(a)               # AX = cursor
    a.emit(isa.LC, 0)             # AX = src[cursor]


def _emit_imm_digit(a: _Asm):
    """Emit produced ``IMM d`` where d = src[cursor]-'0'.  Stack-balanced (EMITP
    pops the produced-imm it consumes)."""
    _read_src_at_cursor(a)                          # AX = src[cursor] (ascii digit)
    a.emit(isa.PSH, 0); a.emit(isa.IMM, 48); a.emit(isa.SUB, 0)   # AX = digit value
    a.emit(isa.PSH, 0)                              # STACK0 := produced imm (digit)
    a.emit(isa.IMM, isa.IMM)                        # AX := opcode IMM(1)
    a.emit(EMITP, 0)                                # OUT_WORD[OUT_PTR++] = IMM d


def _emit_simple(a: _Asm, opcode: int):
    """Emit a produced instruction with immediate 0 (PSH / ADD / MUL / HALT)."""
    a.emit(isa.IMM, 0); a.emit(isa.PSH, 0)          # STACK0 := 0 (produced imm)
    a.emit(isa.IMM, opcode)                         # AX := produced opcode
    a.emit(EMITP, 0)


def chain_compiler_bytecode() -> List[Tuple[str, int]]:
    """The variable-length single-operator chain compiler as c4_min bytecode.

    Layout of the loop (cursor MEM[0], starts at 0):
        MEM[0] := 0
        emit IMM d0                  # produced: first operand
        MEM[0] += 1                  # cursor -> first operator
      loop:
        op := src[cursor]
        if op == 0: goto done        # null terminator -> end
        emit PSH                     # produced
        if op == '*' : emit_add_or_mul via branch
        MEM[0] += 1                  # cursor -> next digit
        emit IMM d(cursor)           # produced operand
        (emit the OP decided above)
        MEM[0] += 1                  # cursor -> next operator
        goto loop
      done:
        emit HALT
        JMP OUT_PTR_start (handoff handled by the run driver via OUT region)
    We restructure so the OP is emitted AFTER the operand (c4 order PSH;IMM d;OP):
        emit PSH ; MEM[0]+=1 ; emit IMM d ; [branch] emit ADD|MUL ; MEM[0]+=1 ; loop
    """
    a = _Asm()
    # cursor MEM[0] := 0
    a.emit(isa.IMM, 0); a.emit(isa.PSH, 0)          # STACK0 := 0 (addr)
    a.emit(isa.IMM, 0)                              # AX := 0 (value)
    a.emit(isa.SI, 0)                               # MEM[0] := 0
    a.emit(isa.ADD, 0)                              # drop pushed addr (balanced)

    _emit_imm_digit(a)                              # emit IMM d0
    _bump_cursor(a, 1)                              # cursor -> first operator

    a.label("loop")
    _read_src_at_cursor(a)                          # AX = src[cursor] = op or 0
    a.emit(isa.BZ, "done")                          # null terminator -> done
    _emit_simple(a, isa.PSH)                        # emit PSH
    _bump_cursor(a, 1)                              # cursor -> next digit
    _emit_imm_digit(a)                              # emit IMM d(next)
    # decide the operator: re-read src[cursor-1] (the operator) and branch.
    _load_cursor(a); a.emit(isa.PSH, 0); a.emit(isa.IMM, 1); a.emit(isa.SUB, 0)  # AX=cursor-1
    a.emit(isa.LC, 0)                               # AX = src[cursor-1] = the operator
    a.emit(isa.PSH, 0); a.emit(isa.IMM, _STAR); a.emit(isa.SUB, 0)  # AX = op - '*'
    a.emit(isa.BZ, "op_mul")                        # op == '*' -> MUL
    _emit_simple(a, isa.ADD)                        # else emit ADD
    a.emit(isa.JMP, "after_op")
    a.label("op_mul")
    _emit_simple(a, isa.MUL)                        # emit MUL
    a.label("after_op")
    _bump_cursor(a, 1)                              # cursor -> next operator
    a.emit(isa.JMP, "loop")

    a.label("done")
    _emit_simple(a, isa.HALT)                       # emit HALT (terminates produced prog)
    # HANDOFF: jump to the start of the produced code (the OUT region base).  The
    # OUT region begins right after the compiler, at PC == gen_size == len(code).
    # We patch this JMP target after the code length is known.
    handoff = a.emit(isa.JMP, 0)
    resolved = a.resolve()
    gen_size = len(resolved)
    resolved[handoff] = ("JMP", gen_size)           # JMP to OUT base == run produced
    return resolved


def run_chain(expr: str, out_size: int = 64, src_size: int = 24, mem_size: int = 8,
              stack_depth: int = 16, max_steps: int = 20000, base=None):
    """Build the LoopCompilerMachine with this chain compiler baked into the
    weights, feed the null-terminated C expression as source, and run it: the
    model reads the source, LOOPS emitting the produced program at the moving
    OUT_PTR cursor, hands off (JMP), and runs it.  Returns ``(result, produced)``.
    """
    prog = chain_compiler_bytecode()
    machine = D.LoopCompilerMachine(prog, out_size=out_size, src_size=src_size,
                                    mem_size=mem_size, stack_depth=stack_depth,
                                    base=base)
    src = [ord(c) for c in expr] + [0]              # null-terminated
    trace, words = machine.run(src, max_steps=max_steps, return_code=True)
    # produced program = OUT_WORD up to the first HALT (inclusive)
    produced = []
    for w in words:
        produced.append(w)
        if (w & 0xFF) == isa.HALT:
            break
    return trace[-1], produced, machine


def produced_disasm(produced):
    out = []
    for w in produced:
        op, imm = w & 0xFF, w >> 8
        nm = ("EMIT" if op == EMIT else "EMITP" if op == EMITP
              else isa.NAMES.get(op, str(op)))
        out.append(nm if imm == 0 else f"{nm} {imm}")
    return "; ".join(out)
