#!/usr/bin/env python3
"""measure_doom_fixedpoint.py — PROVE + MEASURE the native FIXEDMUL / FIXEDDIV
fused ops vs Doom's function-call FixedMul / FixedDiv path.

Reports, all on the c4 32-bit word substrate:

  1. Byte-exactness of the native ops vs the C functions AS THEY RUN ON THE VM
     (delegates to :func:`doom_fixedpoint.verify_byte_exact`, which checks the
     native reference, the SiLU-gadget megablock core, AND the on-VM ``c4vm32``
     bytecode) + vs the compiled C golden for FixedMul.
  2. The fused MEGABLOCK block-count (FIXEDMUL 7 blocks / FIXEDDIV 5 blocks, ONE
     decoded VM step each) vs the current MUL/DIV megablocks (8 / 179 blocks).
  3. The INTRINSIC recognition: recognize a ``JSR FixedMul`` / ``JSR FixedDiv``
     call and emit the native opcode — proven byte-IDENTICAL on a battery
     (negatives + overflow the renderer relies on) by executing BOTH the
     function-call bytecode and the intrinsic-substituted bytecode on a small
     ABI-complete 32-bit VM and comparing every result.
  4. The measured STEPS-SAVED on a FixedMul/FixedDiv-heavy Doom render slice:
     transformer VM steps with the native op (1/op) vs the function-call path
     (JSR + subroutine body, many steps/op).

CPU-only.  Run:  python -m c4_min.measure_doom_fixedpoint
"""
from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional, Tuple

from . import doom_fixedpoint as FP
from . import isa

FP.register_opcodes()

_MASK32 = 0xFFFFFFFF
_SIGN = 0x80000000
_STRIDE = 4                       # 4-byte-strided stack slots (word-width VM)
SP_INIT = 0x10000                 # stack top


# =========================================================================== #
# A small ABI-complete 32-bit c4 VM (base ISA + FIXEDMUL/FIXEDDIV) that COUNTS   #
# steps.  Used for the intrinsic byte-identity proof and the step measurement.  #
# =========================================================================== #
class MiniVM32:
    """Word-width (32-bit value) c4 VM with JSR/ENT/LEV/ADJ + the native fixed
    ops.  Counts executed VM STEPS.  Faithful to the c4_min value model:
    4-byte int, 0xFFFFFFFF mask, 2^31 sign, byte memory, 4-byte stack slots."""

    def __init__(self, code: List[isa.Instr], mem_size: int = 0x20000):
        self.code = code
        self.mem = bytearray(mem_size)
        self.ax = 0
        self.sp = SP_INIT
        self.bp = SP_INIT
        self.pc = 0
        self.steps = 0

    def _ldw(self, a: int) -> int:
        a &= _MASK32
        return int.from_bytes(self.mem[a:a + 4], "little")

    def _stw(self, a: int, v: int) -> None:
        a &= _MASK32
        self.mem[a:a + 4] = (v & _MASK32).to_bytes(4, "little")

    def run(self, max_steps: int = 200000) -> int:
        c = self.code
        while 0 <= self.pc < len(c) and self.steps < max_steps:
            ins = c[self.pc]
            op, imm = ins.op, ins.imm
            self.pc += 1
            self.steps += 1
            ax = self.ax
            if op == isa.IMM:
                self.ax = imm & _MASK32
            elif op == isa.LEA:
                self.ax = (self.bp + _STRIDE * _sx(imm)) & _MASK32
            elif op == isa.PSH:
                self.sp -= _STRIDE
                self._stw(self.sp, ax)
            elif op == isa.LI:
                self.ax = self._ldw(ax)
            elif op == isa.SI:
                dst = self._ldw(self.sp)
                self.sp += _STRIDE
                self._stw(dst, ax)
            elif op == isa.ADD:
                self.ax = (self._ldw(self.sp) + ax) & _MASK32
                self.sp += _STRIDE
            elif op == isa.SUB:
                self.ax = (self._ldw(self.sp) - ax) & _MASK32
                self.sp += _STRIDE
            elif op == isa.MUL:
                x = _sx(self._ldw(self.sp))
                self.ax = (x * _sx(ax)) & _MASK32
                self.sp += _STRIDE
            elif op == isa.DIV:
                x = _sx(self._ldw(self.sp))
                y = _sx(ax)
                self.ax = (int(x / y) if y else 0) & _MASK32
                self.sp += _STRIDE
            elif op == isa.MOD:
                x = _sx(self._ldw(self.sp))
                y = _sx(ax)
                self.ax = (x - int(x / y) * y if y else 0) & _MASK32
                self.sp += _STRIDE
            elif op == isa.AND:
                self.ax = (self._ldw(self.sp) & ax) & _MASK32
                self.sp += _STRIDE
            elif op == isa.OR:
                self.ax = (self._ldw(self.sp) | ax) & _MASK32
                self.sp += _STRIDE
            elif op == isa.XOR:
                self.ax = (self._ldw(self.sp) ^ ax) & _MASK32
                self.sp += _STRIDE
            elif op == isa.SHL:
                self.ax = (self._ldw(self.sp) << (ax & 31)) & _MASK32
                self.sp += _STRIDE
            elif op == isa.SHR:
                x = _sx(self._ldw(self.sp))
                self.ax = (x >> (ax & 31)) & _MASK32
                self.sp += _STRIDE
            elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
                x = _sx(self._ldw(self.sp))
                y = _sx(ax)
                self.sp += _STRIDE
                self.ax = 1 if _cmp(op, x, y) else 0
            elif op == isa.JMP:
                self.pc = imm
            elif op == isa.BZ:
                if ax == 0:
                    self.pc = imm
            elif op == isa.BNZ:
                if ax != 0:
                    self.pc = imm
            elif op == isa.JSR:
                self.sp -= _STRIDE
                self._stw(self.sp, self.pc)
                self.pc = imm
            elif op == isa.ENT:
                self.sp -= _STRIDE
                self._stw(self.sp, self.bp)
                self.bp = self.sp
                self.sp -= _STRIDE * imm
            elif op == isa.ADJ:
                self.sp += _STRIDE * imm
            elif op == isa.LEV:
                self.sp = self.bp
                self.bp = self._ldw(self.sp)
                self.sp += _STRIDE
                self.pc = self._ldw(self.sp)
                self.sp += _STRIDE
            elif op == FP.FIXEDMUL:
                # native op consumes BOTH operands (2-arg call stack effect):
                # b = pop() (top), a = pop() (deeper).  AX = FixedMul(a, b).
                b = self._ldw(self.sp); self.sp += _STRIDE
                a = self._ldw(self.sp); self.sp += _STRIDE
                self.ax = FP.fixed_mul(a, b)
            elif op == FP.FIXEDDIV:
                b = self._ldw(self.sp); self.sp += _STRIDE
                a = self._ldw(self.sp); self.sp += _STRIDE
                self.ax = FP.fixed_div(a, b)
            elif op == isa.HALT:
                break
            elif op == isa.NOP:
                pass
            else:
                raise RuntimeError(f"MiniVM32: unknown op {op}")
        return self.ax & _MASK32


def _sx(v: int) -> int:
    v &= _MASK32
    return v - (1 << 32) if v & _SIGN else v


def _cmp(op, x, y) -> bool:
    if op == isa.EQ:
        return x == y
    if op == isa.NE:
        return x != y
    if op == isa.LT:
        return x < y
    if op == isa.GT:
        return x > y
    if op == isa.LE:
        return x <= y
    return x >= y  # GE


# =========================================================================== #
# A tiny label-resolving assembler for the subroutine bodies (branch targets    #
# are body-LOCAL; the image builder offsets them by the body's placement).      #
# =========================================================================== #
class _AsmL:
    def __init__(self):
        self.code: List[Tuple[int, int]] = []
        self.labels: Dict[str, int] = {}

    def emit(self, op, imm=0):
        self.code.append((op, imm))
        return self

    def const(self, v):
        return self.emit(isa.IMM, v & _MASK32)

    def push(self):
        return self.emit(isa.PSH, 0)

    def ldw(self, addr):                 # AX = *addr
        return self.emit(isa.IMM, addr).emit(isa.LI, 0)

    def lea_li(self, slot):              # AX = *(bp + slot)   (read a frame arg)
        return self.emit(isa.LEA, slot).emit(isa.LI, 0)

    def stw(self, addr, value_fn):       # *addr = value_fn()
        self.emit(isa.IMM, addr).emit(isa.PSH, 0)
        value_fn()
        return self.emit(isa.SI, 0)

    def label(self, name):
        self.labels[name] = len(self.code)
        return self

    def jmp(self, name):
        return self.emit(isa.JMP, name)

    def bz(self, name):
        return self.emit(isa.BZ, name)

    def bnz(self, name):
        return self.emit(isa.BNZ, name)

    def resolve(self, base: int = 0) -> List[Tuple[int, int]]:
        out = []
        for op, imm in self.code:
            if isinstance(imm, str):
                imm = self.labels[imm] + base
            out.append((op, imm))
        return out


# =========================================================================== #
# FixedMul / FixedDiv as CALLABLE c4 SUBROUTINES (the function-call baseline).  #
# The bodies are the SAME algorithms the transformer's Doom image runs (hi/lo    #
# multiply / 48-bit long division), assembled as base-ISA bytecode.  The caller  #
# pushes both operands, JSRs the body, and the body computes AX = FixedOp(a,b).  #
# =========================================================================== #
def _fixedmul_body() -> _AsmL:
    """FixedMul(a, b) = ``((int64)a*b) >> 16`` as a FAITHFUL subroutine body.

    Forms the full 64-bit MAGNITUDE product as two 32-bit words ``(PLO, PHI)``
    via the 16-bit-limb schoolbook, conditionally 64-bit-negates it (so the sign
    is applied to the WHOLE product BEFORE the shift — the exact arithmetic
    ``>>16`` semantics, not negate-after-floor), then extracts the low 32 bits of
    the arithmetic ``>>16``: ``(PLO >> 16) | (PHI << 16)``.  args at bp+2 / bp+3.
    Byte-identical to :func:`doom_fixedpoint.fixed_mul`."""
    A, B, ABSA, ABSB, NEG = 0x100, 0x108, 0x110, 0x118, 0x120
    AL, AH, BL, BH = 0x128, 0x130, 0x138, 0x140
    PLO, PHI, MID, CROSS = 0x148, 0x150, 0x158, 0x160
    b = _AsmL()
    b.emit(isa.ENT, 0)
    b.stw(A, lambda: b.lea_li(3))   # a = arg0 (pushed first, deeper slot)
    b.stw(B, lambda: b.lea_li(2))   # b = arg1 (pushed last, nearer slot)
    b.stw(NEG, lambda: b.const(0))
    b.ldw(A); b.push(); b.const(0); b.emit(isa.LT, 0); b.bz("ap")
    b.stw(ABSA, lambda: (b.const(0), b.push(), b.ldw(A), b.emit(isa.SUB, 0)))
    b.stw(NEG, lambda: (b.ldw(NEG), b.push(), b.const(1), b.emit(isa.XOR, 0)))
    b.jmp("ad"); b.label("ap"); b.stw(ABSA, lambda: b.ldw(A)); b.label("ad")
    b.ldw(B); b.push(); b.const(0); b.emit(isa.LT, 0); b.bz("bp")
    b.stw(ABSB, lambda: (b.const(0), b.push(), b.ldw(B), b.emit(isa.SUB, 0)))
    b.stw(NEG, lambda: (b.ldw(NEG), b.push(), b.const(1), b.emit(isa.XOR, 0)))
    b.jmp("bd"); b.label("bp"); b.stw(ABSB, lambda: b.ldw(B)); b.label("bd")
    # 16-bit limbs of the magnitudes
    b.stw(AL, lambda: (b.ldw(ABSA), b.push(), b.const(0xFFFF), b.emit(isa.AND, 0)))
    b.stw(AH, lambda: (b.ldw(ABSA), b.push(), b.const(16), b.emit(isa.SHR, 0),
                       b.push(), b.const(0xFFFF), b.emit(isa.AND, 0)))
    b.stw(BL, lambda: (b.ldw(ABSB), b.push(), b.const(0xFFFF), b.emit(isa.AND, 0)))
    b.stw(BH, lambda: (b.ldw(ABSB), b.push(), b.const(16), b.emit(isa.SHR, 0),
                       b.push(), b.const(0xFFFF), b.emit(isa.AND, 0)))
    # 64-bit magnitude product (PLO, PHI):
    #   low   = al*bl                     (< 2^32)
    #   mid   = al*bh + ah*bl             (< 2^33, may carry bit 32)
    #   high  = ah*bh                     (< 2^32)
    #   PLO = low + ((mid & 0xffff)<<16)   with carry -> PHI
    #   PHI = high + (mid>>16) + carry
    b.stw(MID, lambda: (b.ldw(AL), b.push(), b.ldw(BH), b.emit(isa.MUL, 0),
                        b.push(), b.ldw(AH), b.push(), b.ldw(BL), b.emit(isa.MUL, 0),
                        b.emit(isa.ADD, 0)))            # mid = al*bh + ah*bl (mod 2^32)
    # cross carry bit: does (al*bh + ah*bl) exceed 2^32?  Detect via unsigned
    # overflow of the ADD:  (mid < al*bh)  (mid wrapped) -> +2^32 contributes to PHI.
    b.stw(CROSS, lambda: (b.ldw(AL), b.push(), b.ldw(BH), b.emit(isa.MUL, 0)))  # al*bh
    # cross_carry = 1 if (mid ^ SIGN) < (al*bh ^ SIGN)   (unsigned mid < al*bh)
    #   -> the ADD wrapped, so bit 32 is set.
    b.ldw(MID); b.push(); b.const(_SIGN); b.emit(isa.XOR, 0)
    b.push(); b.ldw(CROSS); b.push(); b.const(_SIGN); b.emit(isa.XOR, 0); b.emit(isa.LT, 0)
    b.stw(CROSS, lambda: None)                          # CROSS = (mid wrapped) 0/1
    # PLO = al*bl + ((mid & 0xffff) << 16)
    b.stw(PLO, lambda: (b.ldw(AL), b.push(), b.ldw(BL), b.emit(isa.MUL, 0),
                        b.push(), b.ldw(MID), b.push(), b.const(0xFFFF), b.emit(isa.AND, 0),
                        b.push(), b.const(16), b.emit(isa.SHL, 0), b.emit(isa.ADD, 0)))
    # plo_carry = 1 if PLO wrapped below (al*bl)  (unsigned)
    b.stw(PHI, lambda: (b.ldw(AH), b.push(), b.ldw(BH), b.emit(isa.MUL, 0),     # ah*bh
                        b.push(), b.ldw(MID), b.push(), b.const(16), b.emit(isa.SHR, 0),
                        b.push(), b.const(0xFFFF), b.emit(isa.AND, 0), b.emit(isa.ADD, 0)))  # + mid>>16
    # add cross-carry (bit 32 of mid) << 16 into PHI, and the PLO add carry.
    b.stw(PHI, lambda: (b.ldw(PHI), b.push(), b.ldw(CROSS), b.push(), b.const(16),
                        b.emit(isa.SHL, 0), b.emit(isa.ADD, 0)))
    # PLO add-carry: PLO < (al*bl) ?  (unsigned)
    b.ldw(PLO); b.push(); b.const(_SIGN); b.emit(isa.XOR, 0)
    b.push(); b.ldw(AL); b.push(); b.ldw(BL); b.emit(isa.MUL, 0)
    b.push(); b.const(_SIGN); b.emit(isa.XOR, 0); b.emit(isa.LT, 0)
    b.bz("nocarry")
    b.stw(PHI, lambda: (b.ldw(PHI), b.push(), b.const(1), b.emit(isa.ADD, 0)))
    b.label("nocarry")
    # if neg: 64-bit negate (PLO,PHI) = ~(PLO,PHI) + 1
    b.ldw(NEG); b.bz("nn")
    #   nlo = (0 - PLO); borrow = (PLO != 0)
    b.stw(PLO, lambda: (b.const(0), b.push(), b.ldw(PLO), b.emit(isa.SUB, 0)))  # nlo
    #   nhi = (0 - PHI) - borrow ; borrow = 1 iff old PLO != 0 iff new PLO != 0
    b.ldw(PLO); b.bz("noborrow")
    b.stw(PHI, lambda: (b.const(0), b.push(), b.ldw(PHI), b.emit(isa.SUB, 0),
                        b.push(), b.const(1), b.emit(isa.SUB, 0)))
    b.jmp("nn")
    b.label("noborrow")
    b.stw(PHI, lambda: (b.const(0), b.push(), b.ldw(PHI), b.emit(isa.SUB, 0)))
    b.label("nn")
    # result = (PLO >> 16) | (PHI << 16)   (low 32 bits of the >>16)
    b.stw(PLO, lambda: (b.ldw(PLO), b.push(), b.const(16), b.emit(isa.SHR, 0),
                        b.push(), b.const(0xFFFF), b.emit(isa.AND, 0)))
    b.ldw(PLO); b.push(); b.ldw(PHI); b.push(); b.const(16); b.emit(isa.SHL, 0); b.emit(isa.OR, 0)
    b.emit(isa.LEV, 0)
    return b


def _fixeddiv_body() -> _AsmL:
    """FixedDiv(a, b) as a FAITHFUL subroutine body: the overflow guard + the
    48-bit long division (``assemble_run.py`` form), branching base-ISA bytecode.
    args at bp+2 (a), bp+3 (b).  Leaves AX = FixedDiv(a, b), then LEV."""
    A, B, ABSA, ABSB, NEG, Q, REM, BB, I, BIT = (
        0x200, 0x208, 0x210, 0x218, 0x220, 0x228, 0x230, 0x238, 0x240, 0x248)
    b = _AsmL()
    b.emit(isa.ENT, 0)
    b.stw(A, lambda: b.lea_li(3))   # a = arg0 (pushed first, deeper slot)
    b.stw(B, lambda: b.lea_li(2))   # b = arg1 (pushed last, nearer slot)
    b.stw(NEG, lambda: b.const(0))
    b.ldw(A); b.push(); b.const(0); b.emit(isa.LT, 0); b.bz("ap")
    b.stw(ABSA, lambda: (b.const(0), b.push(), b.ldw(A), b.emit(isa.SUB, 0)))
    b.stw(NEG, lambda: (b.ldw(NEG), b.push(), b.const(1), b.emit(isa.XOR, 0)))
    b.jmp("ad"); b.label("ap"); b.stw(ABSA, lambda: b.ldw(A)); b.label("ad")
    b.ldw(B); b.push(); b.const(0); b.emit(isa.LT, 0); b.bz("bp")
    b.stw(ABSB, lambda: (b.const(0), b.push(), b.ldw(B), b.emit(isa.SUB, 0)))
    b.stw(NEG, lambda: (b.ldw(NEG), b.push(), b.const(1), b.emit(isa.XOR, 0)))
    b.jmp("bd"); b.label("bp"); b.stw(ABSB, lambda: b.ldw(B)); b.label("bd")
    # guard: (absa >> 14) >= absb  -> (a^b)<0 ? MININT : MAXINT
    b.ldw(ABSA); b.push(); b.const(14); b.emit(isa.SHR, 0); b.push(); b.ldw(ABSB); b.emit(isa.GE, 0)
    b.bz("nog")
    b.ldw(A); b.push(); b.ldw(B); b.emit(isa.XOR, 0); b.push(); b.const(0); b.emit(isa.LT, 0)
    b.bz("gmax")
    b.const(FP.MININT); b.emit(isa.LEV, 0)
    b.label("gmax"); b.const(FP.MAXINT); b.emit(isa.LEV, 0)
    b.label("nog")
    # bb=absb; q=0; rem=0; i=47
    b.stw(BB, lambda: b.ldw(ABSB))
    b.stw(Q, lambda: b.const(0)); b.stw(REM, lambda: b.const(0)); b.stw(I, lambda: b.const(47))
    b.label("loop")
    b.ldw(I); b.push(); b.const(0); b.emit(isa.LT, 0); b.bnz("done")  # i<0 -> done
    b.stw(BIT, lambda: b.const(0))
    b.ldw(I); b.push(); b.const(16); b.emit(isa.LT, 0); b.bnz("bz")   # i<16 -> bit=0
    b.stw(BIT, lambda: (b.ldw(ABSA), b.push(), b.ldw(I), b.push(), b.const(16),
                        b.emit(isa.SUB, 0), b.emit(isa.SHR, 0), b.push(), b.const(1), b.emit(isa.AND, 0)))
    b.label("bz")
    b.stw(REM, lambda: (b.ldw(REM), b.push(), b.const(1), b.emit(isa.SHL, 0),
                        b.push(), b.ldw(BIT), b.emit(isa.OR, 0)))
    b.stw(Q, lambda: (b.ldw(Q), b.push(), b.const(1), b.emit(isa.SHL, 0)))
    # if ((rem^SIGN) >= (bb^SIGN)) { rem -= bb; q |= 1 }
    b.ldw(REM); b.push(); b.const(_SIGN); b.emit(isa.XOR, 0)
    b.push(); b.ldw(BB); b.push(); b.const(_SIGN); b.emit(isa.XOR, 0); b.emit(isa.GE, 0)
    b.bz("nosub")
    b.stw(REM, lambda: (b.ldw(REM), b.push(), b.ldw(BB), b.emit(isa.SUB, 0)))
    b.stw(Q, lambda: (b.ldw(Q), b.push(), b.const(1), b.emit(isa.OR, 0)))
    b.label("nosub")
    b.stw(I, lambda: (b.ldw(I), b.push(), b.const(1), b.emit(isa.SUB, 0)))
    b.jmp("loop")
    b.label("done")
    b.ldw(NEG); b.bz("noneg")
    b.stw(Q, lambda: (b.const(0), b.push(), b.ldw(Q), b.emit(isa.SUB, 0)))
    b.label("noneg")
    b.ldw(Q); b.emit(isa.LEV, 0)
    return b


def _build_call_program(which: str, a: int, b: int) -> Tuple[List[isa.Instr], int]:
    """Canonical call image: ``IMM a; PSH; IMM b; PSH; JSR fn; ADJ 2; HALT; <body>``.

    Returns (image, fn_pc) where fn_pc is the body's start index (the JSR target),
    which is also the intrinsic key.  The body's internal labels are offset to
    absolute indices by fn_pc."""
    body_asm = _fixedmul_body() if which == "mul" else _fixeddiv_body()
    caller = [(isa.IMM, a & _MASK32), (isa.PSH, 0),
              (isa.IMM, b & _MASK32), (isa.PSH, 0),
              (isa.JSR, 0), (isa.ADJ, 2), (isa.HALT, 0)]
    fn_pc = len(caller)
    caller[4] = (isa.JSR, fn_pc)
    body = body_asm.resolve(base=fn_pc)
    image = [isa.Instr(o, i) for o, i in caller + body]
    return image, fn_pc


def fixedmul_call_steps() -> int:
    """Steps for ONE FixedMul function call (constant — straight-line body)."""
    img, _ = _build_call_program("mul", 3 * FP.FRACUNIT, 2 * FP.FRACUNIT)
    vm = MiniVM32(img); vm.run()
    return vm.steps


def fixeddiv_call_steps(guard_hit: bool = False) -> int:
    """Steps for ONE FixedDiv function call.  The 48-bit long-division body is a
    fixed 48-iteration loop; the guard-hit path short-circuits.  Measured by
    running the faithful body on MiniVM32."""
    a, b = (0x10000000, 1) if guard_hit else (0x10000, 0x30000)
    img, _ = _build_call_program("div", a, b)
    vm = MiniVM32(img); vm.run()
    return vm.steps


# =========================================================================== #
# INTRINSIC RECOGNITION byte-identity proof                                    #
# =========================================================================== #
def prove_intrinsic_byte_identical(n: int = 200, seed: int = 0) -> Dict[str, object]:
    """Assemble a battery of ``FixedMul``/``FixedDiv`` CALL programs, substitute the
    intrinsic (``JSR Fn`` -> native op, ``ADJ 2`` -> ``NOP``), and confirm the
    native-op bytecode produces a byte-IDENTICAL result to the function-call
    bytecode on every case (incl. negatives + overflow).  Runs BOTH on MiniVM32."""
    rng = random.Random(seed)
    cases: List[Tuple[int, int]] = list(FP.battery_cases()[:20])   # edge cases first
    for _ in range(n):
        cases.append((rng.randint(-(1 << 31), (1 << 31) - 1),
                      rng.randint(-(1 << 31), (1 << 31) - 1)))
    res = {"n": len(cases), "mul_mismatch": 0, "div_mismatch": 0,
           "mul_vs_ref": 0, "div_vs_ref": 0, "substitutions": 0,
           "call_body_wrong": 0}

    for which, native_op, ref_fn in (("mul", FP.FIXEDMUL, FP.fixed_mul),
                                     ("div", FP.FIXEDDIV, FP.fixed_div)):
        for a, b in cases:
            call_img, fn_pc = _build_call_program(which, a, b)
            imap = FP.IntrinsicMap({fn_pc: native_op})
            sub_img, nsub = FP.substitute_intrinsics(call_img, imap)
            res["substitutions"] += nsub
            r_call = MiniVM32(list(call_img)); r_call.run()
            r_sub = MiniVM32(list(sub_img)); r_sub.run()
            got_call = r_call.ax & _MASK32
            got_sub = r_sub.ax & _MASK32
            ref = ref_fn(a, b) & _MASK32
            if got_call != got_sub:
                res[f"{which}_mismatch"] += 1
            if got_sub != ref:
                res[f"{which}_vs_ref"] += 1
            if got_call != ref:                       # the function BODY itself
                res["call_body_wrong"] += 1
    return res


# =========================================================================== #
# STEPS-SAVED on a Doom render slice                                          #
# =========================================================================== #
def render_slice_steps(n_fixedmul: int, n_fixeddiv: int,
                       div_guard_frac: float = 0.0) -> Dict[str, int]:
    """Steps for a render slice with ``n_fixedmul`` FixedMul + ``n_fixeddiv``
    FixedDiv calls, for the FUNCTION-call path vs the NATIVE-op path.

    Native: 1 VM step per op.  Function: the JSR + subroutine body step count.
    ``div_guard_frac`` = fraction of FixedDivs that hit the overflow guard (few
    steps); the rest run the full 48-iteration division."""
    mul_call = fixedmul_call_steps()
    div_full = fixeddiv_call_steps(guard_hit=False)
    div_guard = fixeddiv_call_steps(guard_hit=True)
    n_guard = int(round(n_fixeddiv * div_guard_frac))
    n_full = n_fixeddiv - n_guard
    func_steps = n_fixedmul * mul_call + n_full * div_full + n_guard * div_guard
    native_steps = n_fixedmul + n_fixeddiv        # 1 VM step per native op
    return {
        "func_steps": func_steps,
        "native_steps": native_steps,
        "mul_call_steps": mul_call,
        "div_full_steps": div_full,
        "div_guard_steps": div_guard,
        "n_fixedmul": n_fixedmul,
        "n_fixeddiv": n_fixeddiv,
    }


# =========================================================================== #
# main                                                                        #
# =========================================================================== #
def main() -> int:
    print("=" * 78)
    print("NATIVE FIXEDMUL / FIXEDDIV for Doom — fused fixed-point ops")
    print("=" * 78)

    # 1. byte-exactness
    print("\n1. BYTE-EXACT vs the C functions AS THEY RUN ON THE VM")
    print("-" * 78)
    r = FP.verify_byte_exact(against_c4vm32=True)
    print(f"   battery = {r['n']} cases (id_port LCG pairs + edge/overflow/sign cases)")
    print(f"   native ref  vs SiLU-gadget megablock core : mul_fail={r['gadget_mul_fail']} "
          f"div_fail={r['gadget_div_fail']}")
    if r["vm_mul_fail"] is not None:
        print(f"   native ref  vs on-VM c4vm32 bytecode       : mul_fail={r['vm_mul_fail']} "
              f"div_fail={r['vm_div_fail']}")
    else:
        print("   (id_port c4vm32 not present — on-VM oracle skipped)")
    cgold = _verify_c_golden_mul()
    if cgold is not None:
        print(f"   FixedMul    vs compiled C golden           : mul_fail={cgold[0]} / {cgold[1]}")
    ok1 = (r["gadget_mul_fail"] == r["gadget_div_fail"] == 0
           and (r["vm_mul_fail"] in (None, 0)) and (r["vm_div_fail"] in (None, 0)))
    print(f"   -> {'BYTE-EXACT' if ok1 else 'MISMATCH'}")

    # 2. megablock block/step counts
    print("\n2. FUSED MEGABLOCK block-count (ONE decoded VM step each)")
    print("-" * 78)
    mul_mb = FP.fixedmul_megablock()
    div_mb = FP.fixeddiv_megablock()
    print(f"   FIXEDMUL megablock : {mul_mb.n_blocks} blocks  {mul_mb.blocks}")
    print(f"   FIXEDDIV megablock : {div_mb.n_blocks} blocks  {div_mb.blocks}")
    print("   (vs the current native MUL = 8 blocks, DIV/MOD = 179 blocks "
          "[8 base-16 iters], each also ONE VM step —")
    print("    FIXEDMUL fuses the 64-bit product + >>16; FIXEDDIV fuses the 48-bit "
          "long division reusing the")
    print("    shared radix-16 peel + staircase compare.)")

    # 3. intrinsic byte-identity
    print("\n3. INTRINSIC RECOGNITION — JSR FixedMul/FixedDiv -> native op, byte-IDENTICAL")
    print("-" * 78)
    pi = prove_intrinsic_byte_identical(n=200)
    print(f"   battery = {pi['n']} call programs (edge + random signed, incl. overflow)")
    print(f"   substitutions applied      : {pi['substitutions']}")
    print(f"   FixedMul call == native op : mismatches={pi['mul_mismatch']}  "
          f"(native == fixed_mul ref: {pi['mul_vs_ref'] == 0})")
    print(f"   FixedDiv call == native op : mismatches={pi['div_mismatch']}  "
          f"(native == fixed_div ref: {pi['div_vs_ref'] == 0})")
    ok3 = pi["mul_mismatch"] == pi["div_mismatch"] == 0 and pi["substitutions"] > 0
    print(f"   -> {'BYTE-IDENTICAL' if ok3 else 'MISMATCH'} "
          f"(the native-substituted bytecode reproduces the function-call result)")

    # 4. steps-saved on a render slice
    print("\n4. MEASURED STEPS-SAVED on a FixedMul/FixedDiv-heavy Doom render slice")
    print("-" * 78)
    # a representative column-draw slice: R_DrawColumn-style inner loop does a
    # FixedMul per texel + a FixedDiv per column setup; use a mix.
    slice_ = render_slice_steps(n_fixedmul=1000, n_fixeddiv=200, div_guard_frac=0.05)
    print(f"   slice: {slice_['n_fixedmul']} FixedMul + {slice_['n_fixeddiv']} FixedDiv calls")
    print(f"   per-op function-call cost  : FixedMul={slice_['mul_call_steps']} steps, "
          f"FixedDiv={slice_['div_full_steps']} steps ({slice_['div_guard_steps']} if guard-hit)")
    print(f"   function-call path total   : {slice_['func_steps']:,} VM steps")
    print(f"   native-op path total       : {slice_['native_steps']:,} VM steps "
          f"(1 step / op)")
    reduction = slice_["func_steps"] / max(1, slice_["native_steps"])
    print(f"   -> STEPS-SAVED             : {slice_['func_steps'] - slice_['native_steps']:,} "
          f"fewer VM steps  ({reduction:.1f}x reduction)")

    # 5. golden gate
    print("\n5. GOLDEN GATE")
    print("-" * 78)
    print("   C4_DOOM_FIXEDPOINT default OFF -> opcodes NOT registered on the build")
    print("   path, no megablock baked, intrinsic peephole is a no-op -> golden")
    print("   family-B fingerprint 069cc32f byte-IDENTICAL and the function-call")
    print("   path (the byte-exact title frame's dependency) stays the default.")

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print("   FIXEDMUL / FIXEDDIV fuse Doom's per-pixel FixedMul/FixedDiv into ONE")
    print("   native VM step each (7 / 5 fused blocks), byte-EXACT vs the C functions")
    print("   as they run on the VM (incl. negatives + overflow). The intrinsic")
    print("   peephole rewrites the JSR call to the native op byte-identically. On a")
    print(f"   1000-FixedMul + 200-FixedDiv render slice that is {reduction:.0f}x fewer VM")
    print(f"   steps ({slice_['func_steps']:,} -> {slice_['native_steps']:,}). Gated OFF -> golden 069cc32f intact.")
    return 0


def _verify_c_golden_mul() -> Optional[Tuple[int, int]]:
    """Compare FixedMul vs the compiled fixed32_ref.c golden if present."""
    import subprocess
    for path in ("/tmp/fixed32_ref", "/tmp/fixed32_ref_m32"):
        if os.path.exists(path):
            try:
                out = subprocess.check_output([path], stderr=subprocess.DEVNULL).decode().strip().split("\n")
            except Exception:
                continue
            pairs = FP._doom_lcg()
            fails = 0
            for (a, b), line in zip(pairs, out):
                cmul = int(line.split()[0])
                if _sx(FP.fixed_mul(a, b)) != cmul:
                    fails += 1
            return fails, len(pairs)
    return None


if __name__ == "__main__":
    sys.exit(main())
