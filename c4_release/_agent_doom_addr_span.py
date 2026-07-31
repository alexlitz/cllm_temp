#!/usr/bin/env python3
"""WALL #1 instrumentation: measure doom's LIVE memory-address span.

Runs a full-opcode pure-Python c4 reference interpreter over doom.c for the
first N executed steps (default 4000) and records the DISTINCT addresses touched
by every memory op (LI/SI/LC/SC) and by the descending stack (PSH/SI/SC targets,
ENT/LEV frames).  Reports:
  * min / max address, count of distinct addresses
  * how many bits are needed to distinguish them without aliasing (bit-width)
  * how badly they alias under the current 8-bit CAM (addr & 0xFF collision map)
  * a suggested MEM_ADDR_BITS width

This sets the required CAM width for _bake_memory_cam.  No model build; pure
Python, ~seconds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DOOM_C = Path("/home/alexlitz/Documents/misc/c4_doom/doom.c")


def run_ref(code, data_seg, max_steps):
    from c4_min import isa
    MASK = 0xFFFFFFFF
    ax = 0
    SP0 = 0x10000                     # SP_INIT (matches the model driver)
    sp = bp = SP0
    pc = 0
    mem = dict(data_seg)
    call = []

    load_addrs = set()                # LI/LC query addresses
    store_addrs = set()               # SI/SC store addresses
    stack_addrs = set()               # every stack word address written
    all_mem_addrs = set()

    def signed(v):
        v &= MASK
        return v - (1 << 32) if v & 0x80000000 else v

    def ld(a, n=4):
        val = 0
        for i in range(n):
            val |= mem.get(a + i, 0) << (8 * i)
        return val

    def st(a, v, n=4):
        for i in range(n):
            mem[a + i] = (v >> (8 * i)) & 0xFF

    steps = 0
    while 0 <= pc < len(code) and steps < max_steps:
        ins = code[pc]
        op, imm = ins.op, ins.imm
        pc += 1
        if op == isa.IMM:
            ax = imm & MASK
        elif op == isa.LEA:
            ax = (bp + imm * 4) & MASK
        elif op == isa.PSH:
            sp -= 4; st(sp, ax); stack_addrs.add(sp); all_mem_addrs.add(sp)
        elif op == isa.LI:
            load_addrs.add(ax); all_mem_addrs.add(ax); ax = ld(ax)
        elif op == isa.LC:
            load_addrs.add(ax); all_mem_addrs.add(ax); ax = mem.get(ax, 0)
        elif op == isa.SI:
            a = ld(sp); sp += 4; st(a, ax); store_addrs.add(a); all_mem_addrs.add(a)
        elif op == isa.SC:
            a = ld(sp); sp += 4; mem[a] = ax & 0xFF; store_addrs.add(a); all_mem_addrs.add(a)
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD, isa.AND, isa.OR,
                    isa.XOR, isa.SHL, isa.SHR, isa.EQ, isa.NE, isa.LT, isa.GT,
                    isa.LE, isa.GE):
            v = signed(ld(sp)); sp += 4; a = signed(ax)
            if op == isa.ADD: ax = (v + a) & MASK
            elif op == isa.SUB: ax = (v - a) & MASK
            elif op == isa.MUL: ax = (v * a) & MASK
            elif op == isa.DIV: ax = (int(v / a) if a else 0) & MASK
            elif op == isa.MOD: ax = (v - int(v / a) * a if a else 0) & MASK
            elif op == isa.AND: ax = (v & a) & MASK
            elif op == isa.OR: ax = (v | a) & MASK
            elif op == isa.XOR: ax = (v ^ a) & MASK
            elif op == isa.SHL: ax = (v << a) & MASK
            elif op == isa.SHR: ax = (v >> a) & MASK
            elif op == isa.EQ: ax = 1 if v == a else 0
            elif op == isa.NE: ax = 1 if v != a else 0
            elif op == isa.LT: ax = 1 if v < a else 0
            elif op == isa.GT: ax = 1 if v > a else 0
            elif op == isa.LE: ax = 1 if v <= a else 0
            elif op == isa.GE: ax = 1 if v >= a else 0
        elif op == isa.JMP: pc = imm
        elif op == isa.BZ: pc = imm if (ax & MASK) == 0 else pc
        elif op == isa.BNZ: pc = imm if (ax & MASK) != 0 else pc
        elif op == isa.JSR: sp -= 4; st(sp, pc); stack_addrs.add(sp); all_mem_addrs.add(sp); pc = imm
        elif op == isa.ENT:
            sp -= 4; st(sp, bp); stack_addrs.add(sp); all_mem_addrs.add(sp); bp = sp; sp -= imm * 4
        elif op == isa.ADJ: sp += imm * 4
        elif op == isa.LEV: sp = bp; bp = ld(sp); sp += 4; pc = ld(sp); sp += 4
        elif op in (isa.OPEN, isa.READ, isa.CLOS, isa.PRTF):
            break
        elif op == isa.HALT:
            break
        elif op == isa.NOP:
            pass
        else:
            break
        steps += 1

    return steps, load_addrs, store_addrs, stack_addrs, all_mem_addrs


def bits_needed(addrs):
    if not addrs:
        return 0
    return max(a.bit_length() for a in addrs)


def alias_report(addrs, bits):
    """How many distinct addrs collide under a `bits`-bit CAM (addr & mask)."""
    mask = (1 << bits) - 1
    buckets = {}
    for a in addrs:
        buckets.setdefault(a & mask, set()).add(a)
    collisions = {k: v for k, v in buckets.items() if len(v) > 1}
    n_aliased = sum(len(v) for v in collisions.values())
    return len(collisions), n_aliased


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()

    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    src = DOOM_C.read_text()
    bytecode, data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    DATA_BASE = 0x10000
    data_seg = {DATA_BASE + i: int(b) & 0xFF for i, b in enumerate(data or [])}
    print(f"doom: {len(code)} instrs, {len(data)} data bytes, DATA_BASE={hex(DATA_BASE)}")

    steps, la, sa, ka, allm = run_ref(code, data_seg, args.steps)
    print(f"executed {steps} steps (stops at first I/O / HALT)\n")

    for name, s in [("LOAD (LI/LC)", la), ("STORE (SI/SC)", sa),
                    ("STACK words", ka), ("ALL mem addrs", allm)]:
        if not s:
            print(f"{name:16s}: (none)")
            continue
        mn, mx = min(s), max(s)
        nb = bits_needed(s)
        print(f"{name:16s}: {len(s):5d} distinct  min={hex(mn)} max={hex(mx)}  "
              f"need {nb} bits (max addr {mx})")

    print("\nALIASING under N-bit CAM (distinct addrs that collide to same key):")
    print(f"  {'bits':>5} {'keyspace':>10} {'colliding-keys':>15} {'aliased-addrs':>15}")
    for b in [8, 12, 16, 18, 20, 24, 28, 32]:
        nk, na = alias_report(allm, b)
        print(f"  {b:5d} {2**b:10d} {nk:15d} {na:15d}"
              + ("   <-- current CAM" if b == 8 else ""))

    nb_all = bits_needed(allm)
    print(f"\n==> doom needs {nb_all} address bits to distinguish ALL live memory "
          f"addresses without aliasing.")
    # find minimal width with zero aliasing
    lo = 8
    while lo <= 32:
        nk, na = alias_report(allm, lo)
        if na == 0:
            print(f"==> MINIMUM zero-alias width in this trace: {lo} bits")
            break
        lo += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
