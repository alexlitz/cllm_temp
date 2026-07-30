#!/usr/bin/env python3
"""_agent_io_reference.py — a clean-room 8-bit reference VM that runs the MODE-1
(strict) and MODE-2 (burst) I/O programs from ``_agent_io_progs`` and returns the
visible stdout bytes.  This is the byte-exact ORACLE the all-C binary is checked
against (mode1 == mode2 == reference).

It faithfully models §Memory as a byte array, the descending stack, and the I/O
syscalls:
  * strict PRTF  -> append AX & 0xFF                (program walks §Memory itself)
  * burst  PRTF  -> printf(mem[AX .. NUL])          (runtime walks §Memory)
  * READ(fd,buf,n) -> inject stdin bytes into §Memory[buf..], AX = n_read

``io_burst`` selects the PRTF semantics, exactly as the C runtime's C4_IO_BURST.
"""
from __future__ import annotations

from typing import Tuple

from c4_min import isa
from c4_min.selfhost._agent_io_progs import IOProg


def run_reference(prog: IOProg, io_burst: bool, max_steps: int = 50000
                  ) -> Tuple[bytes, int]:
    code = isa.assemble(prog.code)
    MASK = 0xFF
    ax = 0
    sp = 256                       # descending stack top (data seg is 0..251)
    pc = 0
    mem = [0] * 258
    stack = [0] * 260
    for a, v in prog.seed_mem.items():
        mem[a & 0xFF] = v & MASK
    stdin = prog.stdin
    spin = 0
    out = bytearray()

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
        if op == isa.IMM:
            ax = imm & MASK
        elif op == isa.PSH:
            push(ax)
        elif op == isa.ADD:
            ax = (pop() + ax) & MASK
        elif op == isa.SUB:
            ax = (pop() - ax) & MASK
        elif op == isa.EQ:
            ax = 1 if (pop() & MASK) == (ax & MASK) else 0
        elif op == isa.NE:
            ax = 1 if (pop() & MASK) != (ax & MASK) else 0
        elif op == isa.LI:
            ax = mem[ax] & MASK
        elif op == isa.LC:
            b = mem[ax] & 0xFF
            ax = (b - 0x100 if b & 0x80 else b) & MASK
        elif op == isa.SI:
            mem[pop()] = ax & MASK
        elif op == isa.SC:
            mem[pop()] = ax & MASK
        elif op == isa.READ:
            n = ax
            buf = pop()
            fd = pop()
            chunk = stdin[spin:spin + n] if fd == 0 else b""
            spin += len(chunk)
            for i, bb in enumerate(chunk):
                mem[(buf + i) & 0xFF] = bb & MASK
            ax = len(chunk) & MASK
        elif op == isa.PRTF:
            if io_burst:
                # runtime walks §Memory mem[AX..NUL] -> stdout (one op)
                p = ax & 0xFF
                guard = 0
                while guard < 512:
                    b = mem[p] & 0xFF
                    if b == 0:
                        break
                    out.append(b)
                    p = (p + 1) & 0xFF
                    guard += 1
            else:
                out.append(ax & 0xFF)      # program supplied one byte in AX
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(f"reference VM: op {op} unhandled")
    return bytes(out), steps


if __name__ == "__main__":
    from c4_min.selfhost import _agent_io_progs as P
    cases = [
        ("echo", {"text": "hello\n"}),
        ("yes",  {"text": "y\n", "n": 4}),
        ("cat",  {"stdin_text": "meow world\n"}),
    ]
    allok = True
    for name, args in cases:
        for io in ("strict", "burst"):
            prog = P.build(name, io, **args)
            got, steps = run_reference(prog, io_burst=(io == "burst"))
            ok = got == prog.expected
            allok = allok and ok
            print(f"{name:5s} {io:6s}: OK={ok} steps={steps:5d} "
                  f"got={got!r} exp={prog.expected!r}")
    print("ALL OK" if allok else "SOME FAILED")
