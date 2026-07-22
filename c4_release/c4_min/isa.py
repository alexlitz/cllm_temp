"""c4_min ISA: 8-bit-only C4 opcode subset, encoding, and a reference interpreter.

Opcode numeric values match the reference ``neural_vm.embedding.Opcode`` so the
semantics are shared, but only the 8-bit subset is defined here. Clean-room: no
import from ``neural_vm``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

MASK = 0xFF          # 8-bit value mask
WIDTH = 2            # cells per instruction slot: [opcode, imm]
NUM_OPS = 40        # size of the opcode one-hot band (covers all values below)

# Opcode values (subset of the C4 ISA). Values match neural_vm.embedding.Opcode.
LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
OR, XOR, AND = 14, 15, 16
EQ, NE, LT, GT, LE, GE = 17, 18, 19, 20, 21, 22
SHL, SHR = 23, 24
ADD, SUB = 25, 26
MUL, DIV, MOD = 27, 28, 29
# File / system opcodes (§File Operations). Values match the BLOG_SPEC opcode
# table (30-33). These cross the computation/outside-world boundary, so they are
# dispatched via the TOOL_CALL-token protocol (see ``nibble_filesys``) rather
# than being computed neurally like the ALU ops above. PRTF (op 33) is the
# printf visible-output byte channel, shared by the tool-IO dispatch and the
# think-tag string quine (§System / §Printing).
OPEN, READ, CLOS, PRTF = 30, 31, 32, 33
# Runtime-library syscall aliases (§"Memory Allocation and Freeing" 687, §"Memset,
# Memcmp and Memcpy" 747). These are NOT neural ops: malloc/free/memset/memcmp are
# "compiled from C into the VM's bytecode and execute entirely neurally" (§687) —
# baked base-ISA subroutines (see ``nibble_runtime``). The op numbers exist ONLY so
# a word-width reference VM (``nibble_runtime.ref_interpret_words``) can run them as
# INTRINSICS to prove the baked bytecode is equivalent to the "compiled from C"
# contract. The unified neural model never decodes them — it runs the bytecode.
MALC, FREE, MSET, MCMP = 34, 35, 36, 37
NOP = 39
HALT = 38  # alias EXIT

NAMES = {
    LEA: "LEA", IMM: "IMM", JMP: "JMP", JSR: "JSR", BZ: "BZ", BNZ: "BNZ",
    ENT: "ENT", ADJ: "ADJ", LEV: "LEV", LI: "LI", LC: "LC", SI: "SI",
    SC: "SC", PSH: "PSH", OR: "OR",
    XOR: "XOR", AND: "AND", EQ: "EQ", NE: "NE", LT: "LT", GT: "GT",
    LE: "LE", GE: "GE", SHL: "SHL", SHR: "SHR", ADD: "ADD", SUB: "SUB",
    MUL: "MUL", DIV: "DIV", MOD: "MOD",
    OPEN: "OPEN", READ: "READ", CLOS: "CLOS", PRTF: "PRTF",
    MALC: "MALC", FREE: "FREE", MSET: "MSET", MCMP: "MCMP",
    NOP: "NOP",
    HALT: "HALT",
}
BY_NAME = {v: k for k, v in NAMES.items()}


@dataclass
class Instr:
    op: int
    imm: int = 0

    def __repr__(self) -> str:
        return f"{NAMES.get(self.op, self.op)} {self.imm}"


def assemble(prog: List[Tuple[str, int]]) -> List[Instr]:
    """Turn [(name, imm), ...] into a code table of Instr.

    The immediate is kept at its full 32-bit width: a value literal (``IMM``/
    ``LEA``) can be a multi-byte constant and a jump/branch/call immediate is a
    PC target that may exceed one byte. The per-op mod-fold (if any) is applied
    at execution in ``_apply_op``, not here.
    """
    out = []
    for entry in prog:
        name, imm = (entry if isinstance(entry, tuple) else (entry, 0))
        out.append(Instr(BY_NAME[name], imm & 0xFFFFFFFF))
    return out


def interpret(code: List[Instr], mem_size: int = 256, max_steps: int = 256,
              out: list = None, stdin=None):
    """Reference 8-bit interpreter. Returns list of AX values emitted per step.

    Stack grows downward from ``mem_size`` (top). ``pop`` reads stack[SP] then SP+=1.
    Emits the value of AX after each executed step. If ``out`` is a list, PRTF
    appends ``AX & 0xFF`` to it — the byte-stream a real stdout would see (the
    ``printf("%c", AX)`` visible-output channel, §System / op 33). This is the
    single-char printf form the classic string quine uses.

    ``stdin`` (a ``nibble_filesys.InputKVStream`` or any object with
    ``read(n) -> bytes``) is the neural-stdin / input-KV byte source a
    ``READ(fd=0, buf, n)`` pulls from (§Tool Use Mode: argv/stdin are "read
    exactly as user input is"). READ marshalling matches
    ``nibble_filesys.dispatch_file_op``: ``fd = pop()``, ``buf = pop()``,
    ``n = AX``; it lays the read bytes into ``mem`` at ``buf`` and sets AX to the
    number of bytes read. This is the SAME semantics the pure-forward driver
    services via the TOOL_CALL runner — so this clean-room VM is a value-faithful
    golden for the argv READ path.
    """
    ax = sp = bp = 0
    sp = mem_size          # empty stack
    pc = 0
    mem = [0] * mem_size
    stack = [0] * (mem_size + 1)
    emitted = []

    def push(v):
        nonlocal sp
        sp -= 1
        stack[sp] = v & MASK

    def pop():
        nonlocal sp
        v = stack[sp]
        sp += 1
        return v & MASK

    steps = 0
    while pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        pc += 1
        if op == IMM:
            ax = imm & MASK
        elif op == LEA:
            ax = (bp + imm) & MASK
        elif op == PSH:
            push(ax)
        elif op == ADD:
            ax = (pop() + ax) & MASK
        elif op == SUB:
            ax = (pop() - ax) & MASK
        elif op == MUL:
            ax = (pop() * ax) & MASK
        elif op == DIV:
            v = pop(); ax = (v // ax if ax else 0) & MASK
        elif op == MOD:
            v = pop(); ax = (v % ax if ax else 0) & MASK
        elif op == AND:
            ax = pop() & ax
        elif op == OR:
            ax = pop() | ax
        elif op == XOR:
            ax = pop() ^ ax
        elif op == SHL:
            ax = (pop() << ax) & MASK
        elif op == SHR:
            ax = (pop() >> ax) & MASK
        elif op == EQ:
            ax = 1 if pop() == ax else 0
        elif op == NE:
            ax = 1 if pop() != ax else 0
        elif op == LT:
            # C4's ordering comparisons are SIGNED, but the neural model's signed
            # gadget uses the 32-bit word's sign bit (2^31).  In this 8-bit slice
            # every value is masked to [0,255] < 2^31, so the sign bit is never set
            # and the SIGNED order coincides with the unsigned byte order — kept as
            # a plain unsigned compare so this reference matches the model at the
            # 8-bit fold (32-bit signed is exercised via ``ref_interpret`` /
            # ``C4_VM_WIDTH32``).
            ax = 1 if pop() < ax else 0
        elif op == GT:
            ax = 1 if pop() > ax else 0
        elif op == LE:
            ax = 1 if pop() <= ax else 0
        elif op == GE:
            ax = 1 if pop() >= ax else 0
        elif op == LI:
            ax = mem[ax] & MASK
        elif op == LC:
            ax = mem[ax] & MASK        # byte load (mem is byte-addressed here)
        elif op == SI:
            mem[pop()] = ax & MASK
        elif op == SC:
            mem[pop()] = ax & MASK     # byte store
        elif op == READ:
            # READ(fd=pop, buf=pop, n=AX) -> n_read.  fd 0 is neural stdin: the
            # bytes are pulled from ``stdin`` (the input-KV stream) and laid into
            # ``mem`` at ``buf`` — argv "read exactly as user input is".
            n = ax
            buf = pop()
            fd = pop()
            if fd == 0 and stdin is not None:
                chunk = stdin.read(n)
            else:
                chunk = b""
            for i, byte in enumerate(chunk):
                mem[(buf + i) % mem_size] = byte & MASK
            ax = len(chunk) & MASK
        elif op == PRTF:
            if out is not None:
                out.append(ax & 0xFF)   # printf visible output byte; AX unchanged
        elif op == JMP:
            pc = imm
        elif op == BZ:
            pc = imm if ax == 0 else pc
        elif op == BNZ:
            pc = imm if ax != 0 else pc
        elif op == HALT:
            emitted.append(ax)
            break
        else:
            raise NotImplementedError(f"op {op} not in slice ISA")
        emitted.append(ax)
    return emitted
