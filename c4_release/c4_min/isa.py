"""c4_min ISA: 8-bit-only C4 opcode subset, encoding, and a reference interpreter.

Opcode numeric values match the reference ``neural_vm.embedding.Opcode`` so the
semantics are shared, but only the 8-bit subset is defined here. Clean-room: no
import from ``neural_vm``.
"""
from __future__ import annotations

import os
import struct
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

# ---------------------------------------------------------------------------
# NATIVE FLOATING-POINT (c4+FP ISA extension, C4_FLOAT_OPS, DEFAULT OFF).
# IEEE-754 single (binary32) elementary ops as SINGLE opcodes, so the
# efficient full-C90 float path is ONE VM op instead of the hundreds-of-steps
# soft-float routine.  A float32 value is carried as its raw 32-bit int bit
# pattern (``struct.pack('<f')`` bits); the ops decode/encode the bit pattern
# and are correctly-rounded round-to-nearest-even, matching gcc.  These opcode
# VALUES (40..43) sit ABOVE the golden NUM_OPS=40 opcode one-hot band, so the
# baseline OP_IS layout is UNCHANGED when the flag is off; the flag widens the
# band to ``NUM_OPS_FLOAT`` only when on (golden byte-identical off).  This is
# the GENERAL-C90 lever — Doom uses ZERO float (fixed-point), so it is NOT a
# Doom-capstone op.
F_ADD, F_SUB, F_MUL, F_DIV = 40, 41, 42, 43
NUM_OPS_FLOAT = 44   # OP_IS band width when C4_FLOAT_OPS is on (covers 0..43)

# ---------------------------------------------------------------------------
# EMIT (in-transformer JIT, ``C4_CFM_EMIT``, DEFAULT OFF).  The CFM-path analog
# of ``nibble_compiler.EMIT`` (value 30 on the bespoke handoff ISA — a value that
# is OPEN here).  EMIT APPENDS / rewrites a CODE frame into the SAME KV §Memory
# the fetch@PC CAM reads: the produced instruction's op rides AX, its imm rides
# STACK0, and the EMIT instruction's OWN immediate is the TARGET code address to
# write — ``code[IMM] := Instr(op=AX, imm=STACK0)`` (mirror of
# ``nibble_compiler.compile_emit_store``: ``CODE_WORD[IMM] := AX + 256*STACK0``).
# Because ``_bake_code_cam`` is an address-keyed CAM over ALL code frames, a
# runtime-appended frame becomes fetchable by PC with NO residual-width (D)
# growth — the program-length-INDEPENDENT JIT the bespoke fixed-width ``CODE_WORD``
# band (program size capped by D) lacks.  EMIT is NOT a neural dispatch op (no FFN
# rule decodes it): the driver services it (like JSR/ENT/LEV/SI/SC) by reading the
# model's AX/STACK0 and mutating the runtime code list.  Its VALUE (45) sits ABOVE
# both NUM_OPS (40) and NUM_OPS_FLOAT (44), so the OP_IS one-hot band width
# (``num_ops_effective``) is UNCHANGED whether or not EMIT is used -> the golden
# 069cc32f layout/build stays byte-identical (EMIT only ever appears in the driver
# + this reference oracle, never in the baked weights).
EMIT = 45


def float_ops_enabled() -> bool:
    """``C4_FLOAT_OPS`` (DEFAULT OFF): add the gated IEEE-754 single F_ADD/F_SUB/
    F_MUL/F_DIV opcodes.  OFF -> the OP_IS band stays ``NUM_OPS`` wide and every
    downstream layout dim is byte-identical to the golden 069cc32f build."""
    return os.environ.get("C4_FLOAT_OPS", "0") not in ("0", "", "false", "False")


def num_ops_effective() -> int:
    """Width of the OP_IS opcode one-hot band for the CURRENT flag state:
    ``NUM_OPS`` (40, golden) when C4_FLOAT_OPS is off, ``NUM_OPS_FLOAT`` (44)
    when on.  The layout reads THIS so a flag-off build is byte-identical."""
    return NUM_OPS_FLOAT if float_ops_enabled() else NUM_OPS

NAMES = {
    LEA: "LEA", IMM: "IMM", JMP: "JMP", JSR: "JSR", BZ: "BZ", BNZ: "BNZ",
    ENT: "ENT", ADJ: "ADJ", LEV: "LEV", LI: "LI", LC: "LC", SI: "SI",
    SC: "SC", PSH: "PSH", OR: "OR",
    XOR: "XOR", AND: "AND", EQ: "EQ", NE: "NE", LT: "LT", GT: "GT",
    LE: "LE", GE: "GE", SHL: "SHL", SHR: "SHR", ADD: "ADD", SUB: "SUB",
    MUL: "MUL", DIV: "DIV", MOD: "MOD",
    OPEN: "OPEN", READ: "READ", CLOS: "CLOS", PRTF: "PRTF",
    MALC: "MALC", FREE: "FREE", MSET: "MSET", MCMP: "MCMP",
    F_ADD: "F_ADD", F_SUB: "F_SUB", F_MUL: "F_MUL", F_DIV: "F_DIV",
    EMIT: "EMIT",
    NOP: "NOP",
    HALT: "HALT",
}
BY_NAME = {v: k for k, v in NAMES.items()}


# ---------------------------------------------------------------------------
# IEEE-754 single (binary32) bit-exact reference — the byte-exact oracle for the
# F_ADD/F_SUB/F_MUL/F_DIV megablocks.  A float32 VALUE is its raw 32-bit int bit
# pattern (``struct.pack('<f')``); these helpers decode the bits, compute the op
# with hardware round-to-nearest-even (via the CPU's own binary32 arithmetic,
# which is correctly-rounded and matches gcc), and re-encode the result bits.
# Python floats are binary64; casting the operands UP to binary64 and the result
# DOWN to binary32 (``struct '<f'`` pack/unpack) reproduces the binary32 op
# EXACTLY for the elementary ops (each is correctly-rounded, and the double
# round double->... is not an issue because +,-,*,/ on two binary32 inputs are
# computed exactly in binary64 then rounded once to binary32 — the classic
# "double rounding is harmless for a single elementary op" result).
# ---------------------------------------------------------------------------
FLOAT_OPS = (F_ADD, F_SUB, F_MUL, F_DIV)


def f32_from_bits(bits: int) -> float:
    """Decode a raw 32-bit IEEE-754 single bit pattern into a Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def bits_from_f32(x: float) -> int:
    """Encode a Python float as the raw 32-bit IEEE-754 single bit pattern
    (round-to-nearest-even to binary32, the hardware default).  A binary64 value
    whose magnitude exceeds the largest finite binary32 rounds to a signed inf —
    ``struct.pack('<f')`` raises ``OverflowError`` on that (and on inf/nan it is
    fine), so we handle the overflow-to-inf case explicitly."""
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        # magnitude above FLT_MAX rounds to signed inf (round-to-nearest-even:
        # anything strictly past the FLT_MAX..inf midpoint, which pack already
        # would have rounded to FLT_MAX below; OverflowError only fires when the
        # rounded result IS inf).
        import math
        sign = 0x80000000 if math.copysign(1.0, x) < 0 else 0
        return sign | 0x7F800000


def f32_op_bits(op: int, a_bits: int, b_bits: int) -> int:
    """Bit-exact IEEE-754 single ``a OP b`` on RAW bit-pattern operands, returning
    the RAW result bit pattern.  ``a`` is the popped stack operand (first), ``b``
    is AX (second) — matching the integer ALU convention ``ax = pop() OP ax``.
    NaN is canonicalised to the standard quiet NaN 0x7FC00000 (as gcc/x86 do for
    a produced NaN); division by zero yields the correctly-signed inf (or the
    canonical NaN for 0/0)."""
    a = f32_from_bits(a_bits)
    b = f32_from_bits(b_bits)
    try:
        if op == F_ADD:
            r = a + b
        elif op == F_SUB:
            r = a - b
        elif op == F_MUL:
            r = a * b
        elif op == F_DIV:
            r = a / b
        else:
            raise ValueError(f"not a float op: {op}")
    except ZeroDivisionError:
        # F_DIV by zero (Python raises instead of producing the IEEE result):
        #   NaN / 0  -> propagate NaN (canonicalised);  0 / 0 -> canonical NaN;
        #   finite x / 0 -> IEEE inf with sign(a)^sign(b).
        import math
        if a != a:                       # a is NaN
            return 0x7FC00000
        if a == 0.0:
            return 0x7FC00000            # 0/0 -> NaN
        sign = (a_bits >> 31) ^ (b_bits >> 31)
        return (sign << 31) | 0x7F800000
    out = bits_from_f32(r)
    # canonicalise any produced NaN (payload/sign) to the x86/gcc quiet NaN.
    if (out & 0x7F800000) == 0x7F800000 and (out & 0x007FFFFF) != 0:
        return 0x7FC00000
    return out & 0xFFFFFFFF


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
              out: list = None, stdin=None, mem_init: dict = None):
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
    # ``mem_init`` pre-seeds the data §Memory (the input "file" a compiler reads via
    # LI/LC) — the reference analog of the CFM driver's pre-seeded store_log.
    if mem_init:
        for a, v in mem_init.items():
            mem[a % mem_size] = v & MASK
    stack = [0] * (mem_size + 1)
    emitted = []
    # EMIT mutates ``code`` in place (writes the just-produced bytecode into the
    # code array a later JMP runs), so work on a private copy — the caller's list
    # stays pristine and re-runnable, matching the model driver's own runtime copy.
    code = list(code)

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
            # c4: ``a = *sp++ >> a`` on a SIGNED ``long long`` -> ARITHMETIC
            # (sign-extending) right shift.  Interpret the popped value as signed
            # at the value width (sign bit = top bit of MASK); Python's ``>>`` on a
            # negative int already sign-fills, then re-mask to the value width.
            _sign = (MASK >> 1) + 1                     # 0x80 at MASK=0xFF
            v = pop()
            v = v - (MASK + 1) if v & _sign else v      # signed interpretation
            ax = (v >> ax) & MASK
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
            # c4: ``a = *(char *)a`` -> SIGNED char load (byte >= 0x80 is negative,
            # sign-extended to the register width).  At MASK=0xFF the register IS
            # one byte, so the sign-extended value re-masks back to the same byte;
            # the signedness becomes observable only under a wider register (see
            # ``ref_interpret`` / ``RefVM``, which sign-extend the byte to 32 bits).
            b = mem[ax] & 0xFF
            ax = (b - 0x100 if b & 0x80 else b) & MASK
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
        elif op == EMIT:
            # In-transformer JIT: rewrite the code slot at ``imm`` to the produced
            # instruction whose op rides AX and whose imm rides the stack top
            # (STACK0), then POP that immediate.  ``code`` is mutated in place, so
            # a later JMP to the emitted region runs the freshly-produced bytecode
            # (self-modifying / just-compiled code).  Byte-exact analog of the CFM
            # driver's EMIT service + ``nibble_compiler.compile_emit_store``.
            produced_imm = pop()                  # STACK0 -> produced instruction imm
            produced_op = ax & MASK               # AX     -> produced instruction op
            while imm >= len(code):
                code.append(Instr(NOP, 0))        # grow into the reserved JIT region
            code[imm] = Instr(produced_op, produced_imm)
        elif op == HALT:
            emitted.append(ax)
            break
        else:
            raise NotImplementedError(f"op {op} not in slice ISA")
        emitted.append(ax)
    return emitted
