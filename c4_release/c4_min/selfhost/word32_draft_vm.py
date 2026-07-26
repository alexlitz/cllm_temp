#!/usr/bin/env python3
"""word32_draft_vm.py — a 32-BIT-MEMORY draft VM (``ref_interpret_word32``).

Why a new VM
============
The byte draft VM (``nibble_pure_forward_complete.ref_interpret`` /
``_analyze_opclass_steps.census``) masks the memory ops to a BYTE:

    * ``SI`` / ``SC``   ->  ``mem[addr] = ax & 0xFF``   (line 1610)
    * ``LI`` / ``LC``   ->  ``ax = mem[ax] & 0xFF``     (line 1607)
    * ``IMM``           ->  ``ax = imm & 0xFF``          (line 1553)
    * ``LEA``           ->  ``ax = (bp + 4*imm) & 0xFF`` (line 1555)
    * ``JSR``  ret addr ->  ``(i + 1) & 0xFF``           (line 1618)

That TRUNCATES any value, loop counter, or address that exceeds 255.  A real
self-emulated forward (dims like 896) has partial-sum accumulators, loop
counters, and array offsets far above 255 — they spill to memory through
``SI`` and are then read back with ``LI`` — so the byte VM computes GARBAGE for
a real-dimension forward (verified: an 8-term paged dot with data > a byte
returns ``[0]``, the accumulator masked away).

``refword_interpret`` (in ``_nonmatmul_ops_src``) is a fully UNMASKED VM (no
wrap at all — Python bigints).  That over-corrects: the neural model's ISA is a
32-bit machine (ADD/SUB/MUL wrap mod 2^32, unsigned DIV/MOD, per ISA_SPEC 4.2 /
``nibble_alu32``), so an unmasked VM diverges from the real model on any 32-bit
overflow.

``ref_interpret_word32`` is the CORRECT draft VM for grounding a real forward:
EVERY value path (IMM, LEA, ALU, PSH, SI/SC, LI/LC, JSR ret, ENT/LEV frame)
wraps to 32 bits (``& 0xFFFFFFFF``), matching the neural model's 32-bit ISA, so
values and loop counters up to 2^32-1 do NOT truncate.  The CONTROL FLOW
(instruction count / step count) is identical to the byte VM for any program
whose values fit a byte (masking is value-only, never control-flow) — so it is
byte-identical on the fits-a-byte cases and CORRECT (matches numpy 32-bit) on
the cases the byte VM truncated.

This is TOOLING only (new file; not on any build path); golden unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from c4_min import isa
import c4_min.nibble_pure_forward_complete as PF

WORD = 0xFFFFFFFF
SIGN = 1 << 31
ADJ = isa.ADJ


def ref_interpret_word32(code: List[isa.Instr], max_steps: int = 20_000_000,
                         out: List[int] = None,
                         seed_mem: Dict[int, int] = None,
                         count_steps: bool = True) -> Tuple[List[int], int]:
    """32-bit-memory SP-addressed draft VM.  Returns ``(ax_trace, steps)``.

    Semantics are the c4 32-bit ISA (``nibble_alu32`` / ISA_SPEC 4.2):
      * every stored / loaded / computed value wraps mod 2^32 (``& WORD``),
      * ADD/SUB/MUL wrap 32-bit; DIV/MOD are C trunc-toward-zero on the SIGNED
        32-bit interpretation (matching the model's signed-divide gadget);
      * ordering compares (LT/GT/LE/GE) are SIGNED 32-bit; EQ/NE bit-equal;
      * OR/XOR/AND/SHL/SHR are 32-bit;
      * addresses (LEA / JSR ret / ENT-BP / LEV) are 32-bit — NO byte wrap, so
        a frame's array walk and a call past instruction 255 are correct.

    ``out`` (if a list) receives ``ax & 0xFF`` on PRTF (the visible stdout
    byte).  ``seed_mem`` pre-seeds memory (word values).  ``count_steps`` is
    always the executed-instruction count (== the byte VM's step count for any
    program, because masking is value-only)."""
    mem: Dict[int, int] = dict(seed_mem or {})
    sp = bp = PF.SP_INIT
    ax = pc = 0
    trace: List[int] = []
    steps = 0

    def s32(v: int) -> int:
        v &= WORD
        return v - (1 << 32) if v & SIGN else v

    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm & WORD
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & WORD
        elif op == isa.PSH:
            sp -= 4
            mem[sp] = ax & WORD
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0) & WORD
            sp += 4
            if op == isa.ADD:
                ax = (v + ax) & WORD
            elif op == isa.SUB:
                ax = (v - ax) & WORD
            elif op == isa.MUL:
                ax = (v * ax) & WORD
            elif op == isa.DIV:
                # signed 32-bit trunc-toward-zero (C semantics), then wrap
                a, b = s32(v), s32(ax)
                ax = (int(a / b) if b else 0) & WORD
            else:
                a, b = s32(v), s32(ax)
                ax = ((a - b * int(a / b)) if b else 0) & WORD
        elif op in (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR):
            v = mem.get(sp, 0) & WORD
            sp += 4
            if op == isa.OR:
                ax = (v | ax) & WORD
            elif op == isa.XOR:
                ax = (v ^ ax) & WORD
            elif op == isa.AND:
                ax = (v & ax) & WORD
            elif op == isa.SHL:
                ax = (v << (ax & 31)) & WORD
            else:
                ax = (v >> (ax & 31)) & WORD       # logical (unsigned) SHR
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0) & WORD
            sp += 4
            av = ax & WORD
            sv, sax = s32(v), s32(av)
            r = {isa.EQ: v == av, isa.NE: v != av, isa.LT: sv < sax,
                 isa.GT: sv > sax, isa.LE: sv <= sax, isa.GE: sv >= sax}[op]
            ax = 1 if r else 0
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0) & WORD             # 32-bit load (was & 0xFF)
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0)
            sp += 4
            mem[addr] = ax & WORD                  # 32-bit store (was & 0xFF)
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            sp -= 4
            mem[sp] = (i + 1) & WORD               # 32-bit ret addr (was & 0xFF)
            pc = imm
        elif op == isa.ENT:
            mem[sp - 4] = bp & WORD
            sp -= 4
            bp = sp
            sp -= 4 * imm
        elif op == ADJ:
            sp += 4 * imm
        elif op == isa.LEV:
            sp = bp
            bp = mem.get(sp, 0)
            pc = mem.get(sp + 4, 0)
            sp += 8
        elif op == isa.PRTF:
            if out is not None:
                out.append(ax & 0xFF)              # visible byte (stdout is bytes)
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            trace.append(ax & WORD)
            break
        else:
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in word32 ISA")
        trace.append(ax & WORD)
    return trace, steps
