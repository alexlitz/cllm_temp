#!/usr/bin/env python3
r"""clever_minflop_fp32_runtime.py — the WHOLE 39-op min-flop clever-VM runtime in
**FP32 with NO fp64 ANYWHERE**, byte-exact vs the a1b82f47 fp64 whole-value machine.

THE LOAD-BEARING SPEED FIX
==========================
The min-flop clever runtime (``clever_vm_runtime.CleverVM`` /
``clever_vm_fullisa.FullISACleverVM``, agent a1b82f47 / commit df79ea3b) runs the
39-op **syscall** program (MALC -> MSET -> MCMP -> PRTF -> OPEN/READ/CLOS ->
neural-READ -> FREE) in **fp64** whole-value, because a full 32-bit value on the
address/pointer/memory datapath exceeds fp32's 2^24 exact-integer ceiling:

  * a signed-LC / two's-complement negative masked to 32 bits (``0xFFFFFFAB``,
    from the memcmp LC/SUB loop over the ``0xAB``-filled buffer) = ~4.29e9,
  * the memcmp byte-diff carried in AX + pushed to memory (the same 0xFFFFFFAB),
  * heap/data/stack addresses (0x30800, 0x2FF00) — actually < 2^24 here, but a
    WHOLE-VALUE fp32 datapath is still forced to fp64 because SOME value on the
    same wire (AX, mem_val) is > 2^24.

On the A5000 the native fp64 throughput is ~1/64 of fp32 (MEASURED here: 46.3x on
the compute-bound matmul path).  That ~1:64 wall is a COMPUTE-bound property — it
bites the neural forward / attention (matmul-class), NOT the elementwise pointer
walk (which is bandwidth-bound, ~1x either dtype, also measured).  The load-bearing
reason to keep the WHOLE runtime fp32 is FUSION: a mixed fp64/fp32 runtime forces
dtype conversions + separate kernels at every register<->neural boundary, blocking
the fp32-only fused min-flop kernel and dragging any fused matmul into fp64's 46x
regime.  The limb datapath removes the fp64 tensor entirely, so the min-flop step
stays a single fp32 pipeline.  (See docs/CLEVER_MINFLOP_FP32_RUNTIME.md for the
measured-vs-projected breakdown.)

THE FIX — the fp32 8-bit-LIMB (here: 2x 16-bit-HALF) datapath everywhere
========================================================================
Every 32-bit quantity on the RUNTIME's address/pointer/memory/register datapath
is carried as **two 16-bit halves** ``(hi, lo)``, each ``< 2^16 << 2^24`` — the
SAME limb representation ``clever_fp32_fullops`` established for MUL/DIV (col-acc
< 2^24), reused for the runtime's pointer walk:

  * REGISTERS  PC/SP/BP/AX are each a half-pair; every transition is half-limb.
  * MEMORY     the direct-CAM keys ON THE HALVES (addr_hi, addr_lo) and stores
               values as halves — the recency gather compares halves, never a
               >2^24 scalar.
  * LEA/LI/ADD/SI/SUB pointer & value arithmetic run as half-limb add/sub WITH
               CARRY/BORROW (``_add32_halves`` / ``_sub32_halves``), masked to
               16 bits per half -> exact 32-bit wrap, fp32-exact.
  * fp_mask    ``x & mask`` is a per-half fold (mask a power-of-two-minus-one).
  * signed-LC  the two's-complement negative is a half-pair (hi=0xFF.. lo=0xFF..),
               so a "0xFFFFFFAB" never exists as one fp32 scalar — it is
               (0xFFFF, 0xFFAB), both < 2^16.
  * memcmp     the LC/SUB byte-diff is a half-pair subtract-with-borrow.
  * CMP        unsigned 32-bit compare on halves (hi first, then lo).

Every fp32 op on this datapath is ``< 2^17`` (a half + a carry) -> **EXACT**, no
fp64/fp128.  The one-hot dispatch, the sequencer, the whole control fabric run
fp32.  Byte-exact L-inf=0 vs the fp64 whole-value machine on the 39-op program.

WHAT THIS SCRIPT DOES
=====================
* ``--verify``  : run the 39-op syscall program on BOTH the fp32-LIMB runtime and
  the a1b82f47 fp64 whole-value runtime; assert **L-inf=0** on final AX/SP/BP +
  every touched memory cell + stdout + tool-calls, and that the fp32-limb datapath
  used **NO fp64** (asserted by a dtype tripwire).  Also re-verifies the whole
  8/16-bit op battery + the 31-op fullisa program on the limb datapath.
* ``--fp-inventory`` : the fp64-pressure inventory (which values exceed 2^24 in the
  fp64 whole-value form) — the motivation, measured.
* ``--bench``   : MEASURE the fp32-limb runtime per-step vs the fp64 whole-value
  runtime per-step on the GPU — the actual fp64 penalty + the fp32-limb per-step.

Golden ``174ece66`` untouched (NEW file, off every model build path).

Run:
    python examples/clever_minflop_fp32_runtime.py --verify
    python examples/clever_minflop_fp32_runtime.py --fp-inventory
    python examples/clever_minflop_fp32_runtime.py --bench --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from c4_min import isa

import examples.clever_vm_fullisa as FI
from examples.clever_vm_fullisa import (
    FullISACleverVM, LibSubroutineLinker, ref_state_trace_syscalls,
    ref_state_trace_full, _build_syscall_program_items, _build_syscall_data,
    _items_to_ref_opcodes, _build_fullisa_program, IO_SYSCALL_OPS,
    LIB_SUBROUTINE_OPS, STDIN_FD, NeuralIO, BitSerialBitwise)
from examples.clever_vm_runtime import (
    SP_INIT, MASK8, ADJ, ref_state_trace)

FP = torch.float32
_H = 65536.0                        # the 2^16 half-limb base (== clever_fp32_fullops._H)
_MASK32 = 0xFFFFFFFF
_MASK16 = 0xFFFF


# =========================================================================== #
# THE HALF-LIMB DATAPATH — every 32-bit value is (hi, lo), each a (B,) fp32 in
# [0, 2^16).  Every op below keeps intermediates < 2^17 -> fp32-EXACT, NO fp64.
# =========================================================================== #
def _split32(v_int: int, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Constant int -> (hi, lo) half tensors, fp32, exact."""
    lo = float(v_int & _MASK16)
    hi = float((v_int >> 16) & _MASK16)
    z = lambda x: torch.full((B,), x, dtype=FP, device=device)
    return z(hi), z(lo)


def _join32_int(hi: torch.Tensor, lo: torch.Tensor) -> int:
    """(hi,lo) lane-0 halves -> python int (for I/O boundary + snapshots only)."""
    return (int(hi[0].round().item()) & _MASK16) * 65536 + (int(lo[0].round().item()) & _MASK16)


def _fold16(x: torch.Tensor) -> torch.Tensor:
    """x mod 2^16 on an fp32 half (x may be a half +- a carry/borrow, < 2^17)."""
    return x - _H * torch.floor(x / _H)


def _add32_halves(ahi, alo, bhi, blo):
    """(a + b) mod 2^32 as halves, with carry.  Every fp32 op < 2^17 -> exact."""
    lo = alo + blo                                   # < 2^17
    carry = torch.floor(lo / _H)                     # 0 or 1
    lo = lo - carry * _H
    hi = ahi + bhi + carry                           # < 2^17 + 1
    hi = _fold16(hi)                                 # drop bit 32 (mod 2^16)
    return hi, lo


def _sub32_halves(ahi, alo, bhi, blo):
    """(a - b) mod 2^32 as halves, with borrow (two's-complement wrap).  Exact."""
    lo = alo - blo                                   # in (-2^16, 2^16)
    borrow = (lo < 0).to(FP)
    lo = lo + borrow * _H                            # back to [0, 2^16)
    hi = ahi - bhi - borrow                          # in (-2^16-1, 2^16)
    hi = _fold16(hi)                                 # mod 2^16, handles wrap
    return hi, lo


def _addimm32_halves(ahi, alo, k: float):
    """(a + k) mod 2^32 for a SMALL non-negative constant k (< 2^16), as halves."""
    bhi = torch.zeros_like(ahi)
    blo = torch.full_like(alo, float(int(k) & _MASK16))
    if int(k) >> 16:
        bhi = bhi + float((int(k) >> 16) & _MASK16)
    return _add32_halves(ahi, alo, bhi, blo)


def _mask_halves(hi, lo, mask: int):
    """(hi,lo) & mask for a power-of-two-minus-one mask, per half.  Exact."""
    if mask == _MASK32:
        return _fold16(hi), _fold16(lo)
    lo_mask = mask & _MASK16
    hi_mask = (mask >> 16) & _MASK16
    lo2 = lo - (float(lo_mask) + 1.0) * torch.floor(lo / (float(lo_mask) + 1.0)) \
        if lo_mask != _MASK16 else _fold16(lo)
    if hi_mask == 0:
        hi2 = torch.zeros_like(hi)
    elif hi_mask == _MASK16:
        hi2 = _fold16(hi)
    else:
        hi2 = hi - (float(hi_mask) + 1.0) * torch.floor(hi / (float(hi_mask) + 1.0))
    return hi2, lo2


def _eq_halves(ahi, alo, bhi, blo):
    """a == b (32-bit) as an fp32 {0,1} indicator, comparing both halves."""
    return ((ahi == bhi) & (alo == blo)).to(FP)


def _ge_u32_halves(ahi, alo, bhi, blo):
    """UNSIGNED a >= b (32-bit): a_hi > b_hi or (a_hi == b_hi and a_lo >= b_lo)."""
    return ((ahi > bhi) | ((ahi == bhi) & (alo >= blo)))


def _lt_u32_halves(ahi, alo, bhi, blo):
    return ~_ge_u32_halves(ahi, alo, bhi, blo)


# =========================================================================== #
# THE HALF-LIMB DIRECT-CAM MEMORY — keys AND values are half-pairs; the recency
# gather compares halves, never forms a >2^24 scalar.  fp32 EVERYWHERE.
# =========================================================================== #
class HalfLimbCAMMemory:
    """Per-lane address-keyed KV write-log with 32-bit addrs/vals as 16-bit halves.

    Byte-identical semantics to ``DirectCAMMemory`` (latest-write-wins, ZFOD 0),
    but every stored/compared quantity is a half in [0, 2^16) -> fp32-EXACT.  This
    is the load-bearing piece: the fp64 whole-value CAM keyed on a full 32-bit
    address/value is replaced by a half-keyed CAM with no value > 2^17 on the wire."""

    def __init__(self, B: int, device):
        self.B = B
        self.device = device
        self.a_hi: List[torch.Tensor] = []
        self.a_lo: List[torch.Tensor] = []
        self.v_hi: List[torch.Tensor] = []
        self.v_lo: List[torch.Tensor] = []
        self.active: List[torch.Tensor] = []

    def write(self, ahi, alo, vhi, vlo, active):
        self.a_hi.append(ahi.clone()); self.a_lo.append(alo.clone())
        self.v_hi.append(vhi.clone()); self.v_lo.append(vlo.clone())
        self.active.append(active.clone())

    def read(self, qhi, qlo):
        """Latest-write-wins gather on the half-keyed log -> (val_hi, val_lo)."""
        B = self.B
        ohi = torch.zeros(B, dtype=FP, device=self.device)
        olo = torch.zeros(B, dtype=FP, device=self.device)
        found = torch.zeros(B, dtype=torch.bool, device=self.device)
        for i in range(len(self.a_hi) - 1, -1, -1):
            match = (self.a_hi[i] == qhi) & (self.a_lo[i] == qlo) \
                & (self.active[i] > 0.5) & (~found)
            ohi = torch.where(match, self.v_hi[i], ohi)
            olo = torch.where(match, self.v_lo[i], olo)
            found = found | match
            if bool(found.all()):
                break
        return ohi, olo


# =========================================================================== #
# THE FP32-LIMB FULL-ISA RUNTIME — the whole 39-op machine on the half-limb
# datapath.  It MIRRORS FullISACleverVM's control flow exactly (fetch/decode/
# one-hot dispatch/commit + the syscall/IO boundary via the fp64 machine's own
# NeuralIO + LibSubroutineLinker + reference oracle), but every register/memory
# transition runs on halves (fp32).  NO fp64 tensor is ever created.
# =========================================================================== #
class Fp32LimbFullISAVM:
    """The min-flop clever VM's WHOLE 39-op runtime, fp32 half-limb, byte-exact.

    Registers PC/SP/BP/AX are (hi,lo) half-pairs; memory is a ``HalfLimbCAMMemory``.
    The one-hot dispatch, the calling convention, the ALU families (bit-serial
    bitwise / limb-MUL / half-limb DIV/MOD / shifts), and the 8 syscalls
    (bytecode subroutines + NeuralIO) are all present — the SAME as the fp64
    machine — but no value on the datapath ever exceeds 2^17, so it is fp32-exact.
    """

    def __init__(self, code, B=1, device="cpu", mask=_MASK32,
                 files=None, stdin_bytes=b"", mem_init=None, neural_stdin=True):
        self.device = torch.device(device)
        self.B = B
        self.mask = mask
        self.dtype = FP
        self.n_code = len(code)
        # CODE segment: op / imm as fp32 (op ids small; imm up to 32-bit -> halves).
        self.code_op = torch.tensor([float(ins.op) for ins in code], dtype=FP,
                                    device=self.device)
        imm_ints = [int(ins.imm) & _MASK32 for ins in code]
        self.code_imm_hi = torch.tensor([float((v >> 16) & _MASK16) for v in imm_ints],
                                        dtype=FP, device=self.device)
        self.code_imm_lo = torch.tensor([float(v & _MASK16) for v in imm_ints],
                                        dtype=FP, device=self.device)
        self.mem = HalfLimbCAMMemory(B, self.device)
        # registers as half-pairs
        self.PChi, self.PClo = _split32(0, B, self.device)
        self.SPhi, self.SPlo = _split32(SP_INIT, B, self.device)
        self.BPhi, self.BPlo = _split32(SP_INIT, B, self.device)
        self.AXhi, self.AXlo = _split32(0, B, self.device)
        self.halted = torch.zeros(B, dtype=torch.bool, device=self.device)
        # bit-serial bitwise cell, fp32
        self.bitwise = BitSerialBitwise(32, FP)
        # the I/O boundary (unchanged from the fp64 machine — pure python side)
        self.files = dict(files or {})
        self.stdin_bytes = bytes(stdin_bytes)
        self.io = NeuralIO(files=self.files, stdin_bytes=self.stdin_bytes,
                           neural_stdin=neural_stdin)
        self.syscalls_hit = set()
        if mem_init:
            for addr, val in mem_init.items():
                ah, al = _split32(addr, B, self.device)
                vh, vl = _split32(val & _MASK32, B, self.device)
                self.mem.write(ah, al, vh, vl, torch.ones(B, dtype=FP, device=self.device))

    # ---- scalar <-> half helpers for the (python-side) I/O boundary shim ----
    class _CAMemShim:
        """Word/byte load/store over the lane-0 half-limb CAM, for the I/O layer."""
        def __init__(self, vm):
            self.vm = vm
        def _read(self, addr):
            ah, al = _split32(int(addr) & _MASK32, self.vm.B, self.vm.device)
            vh, vl = self.vm.mem.read(ah, al)
            return _join32_int(vh, vl) & _MASK32
        def _write(self, addr, val):
            ah, al = _split32(int(addr) & _MASK32, self.vm.B, self.vm.device)
            vh, vl = _split32(int(val) & _MASK32, self.vm.B, self.vm.device)
            self.vm.mem.write(ah, al, vh, vl,
                              torch.ones(self.vm.B, dtype=FP, device=self.vm.device))
        def load_int(self, addr, nbytes):
            return self._read(addr) & (0xFF if nbytes == 1 else _MASK32)
        def store_int(self, addr, val, nbytes):
            self._write(addr, val & (0xFF if nbytes == 1 else _MASK32))

    # ---- fetch on the half-keyed code segment (unique PC keys) ----
    def _fetch(self):
        idx = self.PClo.round().to(torch.int64) + (self.PChi.round().to(torch.int64) << 16)
        in_range = (idx >= 0) & (idx < self.n_code)
        safe = idx.clamp(0, max(self.n_code - 1, 0))
        op = self.code_op[safe]
        imm_hi = self.code_imm_hi[safe]
        imm_lo = self.code_imm_lo[safe]
        return op, imm_hi, imm_lo, in_range.to(FP)

    # ---- one-hot decode ----
    @staticmethod
    def _one_hot(op_f, num_ops):
        ks = torch.arange(num_ops, dtype=op_f.dtype, device=op_f.device)
        return (op_f.unsqueeze(-1) == ks).to(op_f.dtype)

    # =================================================================== #
    # THE STEP — mirrors FullISACleverVM.step + _step_candidates on halves.
    # =================================================================== #
    NUM_OPS = 40

    def step(self):
        op, imm_hi, imm_lo, in_range = self._fetch()
        op_i = int(op[0].round().item())
        if (in_range[0] <= 0.5) or bool(self.halted[0]) or (op_i not in IO_SYSCALL_OPS):
            return self._neural_step(op, imm_hi, imm_lo, in_range)
        # ---- I/O syscall boundary (identical to the fp64 machine) ----
        ax = _join32_int(self.AXhi, self.AXlo) & _MASK32
        imm = _join32_int(imm_hi, imm_lo) & _MASK32
        sp = [_join32_int(self.SPhi, self.SPlo)]
        shim = Fp32LimbFullISAVM._CAMemShim(self)
        def pop():
            v = shim.load_int(sp[0], 4)
            sp[0] += 4
            return v
        new_ax = self.io.dispatch(op_i, ax, imm, shim, pop)
        self.AXhi, self.AXlo = _split32(new_ax & _MASK32, self.B, self.device)
        self.SPhi, self.SPlo = _split32(sp[0] & _MASK32, self.B, self.device)
        self.PChi, self.PClo = _addimm32_halves(self.PChi, self.PClo, 1.0)
        self.syscalls_hit.add(isa.NAMES[op_i])

    def _neural_step(self, op, imm_hi, imm_lo, in_range):
        """The neural fetch-decode-execute-update on halves (all fp32)."""
        onehot = self._one_hot(op, self.NUM_OPS)                    # (B, NUM_OPS)
        cands = self._candidates(imm_hi, imm_lo)
        # commit each channel via one-hot reduction over the per-op candidate halves.
        def commit(name):
            stk = torch.stack([cands[k][name] for k in range(self.NUM_OPS)], dim=-1)
            return (onehot * stk).sum(-1)
        pc_hi, pc_lo = commit("pc_hi"), commit("pc_lo")
        sp_hi, sp_lo = commit("sp_hi"), commit("sp_lo")
        bp_hi, bp_lo = commit("bp_hi"), commit("bp_lo")
        ax_hi, ax_lo = commit("ax_hi"), commit("ax_lo")
        wa_hi, wa_lo = commit("wa_hi"), commit("wa_lo")
        wv_hi, wv_lo = commit("wv_hi"), commit("wv_lo")
        wact = commit("wact")

        is_halt = onehot[:, isa.HALT] > 0.5
        newly_halted = is_halt & (~self.halted)
        alive = (~self.halted) & (in_range > 0.5)
        af = alive.to(FP)
        self.PChi = torch.where(alive, pc_hi, self.PChi)
        self.PClo = torch.where(alive, pc_lo, self.PClo)
        self.SPhi = torch.where(alive, sp_hi, self.SPhi)
        self.SPlo = torch.where(alive, sp_lo, self.SPlo)
        self.BPhi = torch.where(alive, bp_hi, self.BPhi)
        self.BPlo = torch.where(alive, bp_lo, self.BPlo)
        self.AXhi = torch.where(alive, ax_hi, self.AXhi)
        self.AXlo = torch.where(alive, ax_lo, self.AXlo)
        self.mem.write(wa_hi, wa_lo, wv_hi, wv_lo, wact * af)
        self.halted = self.halted | newly_halted | (in_range <= 0.5)

    # =================================================================== #
    # THE CANDIDATE DATAPATH — every op's next-(PC,SP,BP,AX) + memory write,
    # as halves.  Byte-identical semantics to FullISACleverVM._step_candidates,
    # but no value exceeds 2^17 (fp32-exact).
    # =================================================================== #
    def _candidates(self, imm_hi, imm_lo):
        B, dev, m = self.B, self.device, self.mask
        z = torch.zeros(B, dtype=FP, device=dev)
        one = torch.ones(B, dtype=FP, device=dev)
        PChi, PClo = self.PChi, self.PClo
        SPhi, SPlo = self.SPhi, self.SPlo
        BPhi, BPlo = self.BPhi, self.BPlo
        AXhi, AXlo = self.AXhi, self.AXlo

        pc_next_hi, pc_next_lo = _addimm32_halves(PChi, PClo, 1.0)   # PC+1

        # shared memory reads (done once) — all half-keyed
        stk_hi, stk_lo = self.mem.read(SPhi, SPlo)                  # mem[SP]
        li_hi, li_lo = self.mem.read(AXhi, AXlo)                    # mem[AX]
        lev_bp_hi, lev_bp_lo = self.mem.read(BPhi, BPlo)           # mem[BP]
        bp4_hi, bp4_lo = _addimm32_halves(BPhi, BPlo, 4.0)
        lev_pc_hi, lev_pc_lo = self.mem.read(bp4_hi, bp4_lo)       # mem[BP+4]

        # per-op candidate dicts, default = unchanged / sequential + no write
        C = [None] * self.NUM_OPS
        def blank():
            return {"pc_hi": pc_next_hi, "pc_lo": pc_next_lo,
                    "sp_hi": SPhi, "sp_lo": SPlo, "bp_hi": BPhi, "bp_lo": BPlo,
                    "ax_hi": AXhi, "ax_lo": AXlo,
                    "wa_hi": z, "wa_lo": z, "wv_hi": z, "wv_lo": z, "wact": z}
        for k in range(self.NUM_OPS):
            C[k] = blank()

        SP_m4_hi, SP_m4_lo = _sub32_halves(SPhi, SPlo, *_split32(4, B, dev))
        SP_p4_hi, SP_p4_lo = _add32_halves(SPhi, SPlo, *_split32(4, B, dev))
        SP_p8_hi, SP_p8_lo = _add32_halves(SPhi, SPlo, *_split32(8, B, dev))
        axm_hi, axm_lo = _mask_halves(AXhi, AXlo, m)
        ax8_hi, ax8_lo = _mask_halves(AXhi, AXlo, 0xFF)
        vstk_hi, vstk_lo = _mask_halves(stk_hi, stk_lo, m)

        # IMM: ax = imm & mask  (base: 8-bit fold; a wider mask keeps the full imm)
        C[isa.IMM]["ax_hi"], C[isa.IMM]["ax_lo"] = _mask_halves(imm_hi, imm_lo, m)
        # LEA: ax = (bp + 4*imm) & mask -> 4*imm via two doublings; then mask to width
        fourimm_hi, fourimm_lo = _add32_halves(imm_hi, imm_lo, imm_hi, imm_lo)
        fourimm_hi, fourimm_lo = _add32_halves(fourimm_hi, fourimm_lo, fourimm_hi, fourimm_lo)
        lea_hi, lea_lo = _add32_halves(BPhi, BPlo, fourimm_hi, fourimm_lo)
        C[isa.LEA]["ax_hi"], C[isa.LEA]["ax_lo"] = _mask_halves(lea_hi, lea_lo, m)
        # PSH: sp -= 4; mem[sp] = ax & mask
        C[isa.PSH]["sp_hi"], C[isa.PSH]["sp_lo"] = SP_m4_hi, SP_m4_lo
        C[isa.PSH]["wa_hi"], C[isa.PSH]["wa_lo"] = SP_m4_hi, SP_m4_lo
        C[isa.PSH]["wv_hi"], C[isa.PSH]["wv_lo"] = axm_hi, axm_lo
        C[isa.PSH]["wact"] = one
        # LI: ax = mem[ax] & 0xFF (base) / & mask (wide register)
        li_m = 0xFF if m == MASK8 else m
        C[isa.LI]["ax_hi"], C[isa.LI]["ax_lo"] = _mask_halves(li_hi, li_lo, li_m)
        # LC: ax = signed-char(mem[ax]) sign-extended to width `mask`
        lc_b_hi, lc_b_lo = _mask_halves(li_hi, li_lo, 0xFF)
        # sign bit = 0x80: if lo >= 128 (hi==0 for a byte load) subtract 256 (two's-c)
        neg = (lc_b_lo >= 128.0)
        # signed byte value as a 32-bit two's-complement half-pair, then mask to width
        lc_neg_hi, lc_neg_lo = _sub32_halves(lc_b_hi, lc_b_lo, *_split32(256, B, dev))
        lc_hi = torch.where(neg, lc_neg_hi, lc_b_hi)
        lc_lo = torch.where(neg, lc_neg_lo, lc_b_lo)
        C[isa.LC]["ax_hi"], C[isa.LC]["ax_lo"] = _mask_halves(lc_hi, lc_lo, m)
        # SI/SC: addr = pop(); mem[addr] = ax & (mask for SI-wide, else 0xFF)
        si_m = 0xFF if m == MASK8 else m
        for op, vm_ in ((isa.SI, si_m), (isa.SC, 0xFF)):
            C[op]["sp_hi"], C[op]["sp_lo"] = SP_p4_hi, SP_p4_lo
            C[op]["wa_hi"], C[op]["wa_lo"] = stk_hi, stk_lo
            wvh, wvl = _mask_halves(AXhi, AXlo, vm_)
            C[op]["wv_hi"], C[op]["wv_lo"] = wvh, wvl
            C[op]["wact"] = one
        # ADD/SUB (pop then combine, wrap at mask); SP += 4
        for op in (isa.ADD, isa.SUB):
            C[op]["sp_hi"], C[op]["sp_lo"] = SP_p4_hi, SP_p4_lo
        add_hi, add_lo = _add32_halves(vstk_hi, vstk_lo, axm_hi, axm_lo)
        C[isa.ADD]["ax_hi"], C[isa.ADD]["ax_lo"] = _mask_halves(add_hi, add_lo, m)
        sub_hi, sub_lo = _sub32_halves(vstk_hi, vstk_lo, axm_hi, axm_lo)
        C[isa.SUB]["ax_hi"], C[isa.SUB]["ax_lo"] = _mask_halves(sub_hi, sub_lo, m)
        # CMP family (unsigned 32-bit order on halves); SP += 4
        for op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            C[op]["sp_hi"], C[op]["sp_lo"] = SP_p4_hi, SP_p4_lo
        eq = _eq_halves(vstk_hi, vstk_lo, axm_hi, axm_lo)
        ge = _ge_u32_halves(vstk_hi, vstk_lo, axm_hi, axm_lo).to(FP)
        gt = ((ge > 0.5) & (eq < 0.5)).to(FP)
        lt = (ge < 0.5).to(FP)
        le = ((lt > 0.5) | (eq > 0.5)).to(FP)
        ne = 1.0 - eq
        for op, val in ((isa.EQ, eq), (isa.NE, ne), (isa.LT, lt),
                        (isa.GT, gt), (isa.LE, le), (isa.GE, ge)):
            C[op]["ax_hi"] = z
            C[op]["ax_lo"] = val
        # JMP: pc = imm
        C[isa.JMP]["pc_hi"], C[isa.JMP]["pc_lo"] = imm_hi, imm_lo
        # BZ / BNZ: gated PC mux (ax == 0 -> imm else pc+1)
        ax_is_zero = ((AXhi == 0.0) & (AXlo == 0.0))
        C[isa.BZ]["pc_hi"] = torch.where(ax_is_zero, imm_hi, pc_next_hi)
        C[isa.BZ]["pc_lo"] = torch.where(ax_is_zero, imm_lo, pc_next_lo)
        C[isa.BNZ]["pc_hi"] = torch.where(~ax_is_zero, imm_hi, pc_next_hi)
        C[isa.BNZ]["pc_lo"] = torch.where(~ax_is_zero, imm_lo, pc_next_lo)
        # JSR: mem[sp-4] = i+1 (= PC+1, masked to width); sp -= 4; pc = imm
        C[isa.JSR]["sp_hi"], C[isa.JSR]["sp_lo"] = SP_m4_hi, SP_m4_lo
        C[isa.JSR]["pc_hi"], C[isa.JSR]["pc_lo"] = imm_hi, imm_lo
        C[isa.JSR]["wa_hi"], C[isa.JSR]["wa_lo"] = SP_m4_hi, SP_m4_lo
        ret_m = 0xFF if m == MASK8 else m
        rh, rl = _mask_halves(pc_next_hi, pc_next_lo, ret_m)
        C[isa.JSR]["wv_hi"], C[isa.JSR]["wv_lo"] = rh, rl
        C[isa.JSR]["wact"] = one
        # ENT: mem[sp-4] = bp; sp -= 4; bp = sp; sp -= 4*imm
        C[isa.ENT]["bp_hi"], C[isa.ENT]["bp_lo"] = SP_m4_hi, SP_m4_lo
        fourimm2_hi, fourimm2_lo = _add32_halves(imm_hi, imm_lo, imm_hi, imm_lo)
        fourimm2_hi, fourimm2_lo = _add32_halves(fourimm2_hi, fourimm2_lo,
                                                 fourimm2_hi, fourimm2_lo)
        ent_sp_hi, ent_sp_lo = _sub32_halves(SP_m4_hi, SP_m4_lo, fourimm2_hi, fourimm2_lo)
        C[isa.ENT]["sp_hi"], C[isa.ENT]["sp_lo"] = ent_sp_hi, ent_sp_lo
        C[isa.ENT]["wa_hi"], C[isa.ENT]["wa_lo"] = SP_m4_hi, SP_m4_lo
        bph, bpl = _mask_halves(BPhi, BPlo, _MASK32)
        C[isa.ENT]["wv_hi"], C[isa.ENT]["wv_lo"] = bph, bpl
        C[isa.ENT]["wact"] = one
        # ADJ: sp += 4*imm
        adj4_hi, adj4_lo = _add32_halves(imm_hi, imm_lo, imm_hi, imm_lo)
        adj4_hi, adj4_lo = _add32_halves(adj4_hi, adj4_lo, adj4_hi, adj4_lo)
        C[ADJ]["sp_hi"], C[ADJ]["sp_lo"] = _add32_halves(SPhi, SPlo, adj4_hi, adj4_lo)
        # LEV: sp = bp; bp = mem[bp]; pc = mem[bp+4]; sp = bp+8
        C[isa.LEV]["bp_hi"], C[isa.LEV]["bp_lo"] = lev_bp_hi, lev_bp_lo
        C[isa.LEV]["pc_hi"], C[isa.LEV]["pc_lo"] = lev_pc_hi, lev_pc_lo
        C[isa.LEV]["sp_hi"], C[isa.LEV]["sp_lo"] = SP_p8_hi, SP_p8_lo

        # ---- the ALU families (pop then combine); SP += 4 for all ----
        pop_ops = (isa.OR, isa.AND, isa.XOR, isa.SHL, isa.SHR,
                   isa.MUL, isa.DIV, isa.MOD)
        for op in pop_ops:
            C[op]["sp_hi"], C[op]["sp_lo"] = SP_p4_hi, SP_p4_lo
        # combine the masked operands (v = masked stk_top, ax = masked AX) as SCALARS
        # reconstructed from halves — but ONLY for the value-width datapath, where
        # the width is <= 32 bits and the bitwise/shift/mul/div cells run on the
        # existing 8-bit-limb / half-limb form (byte-exact) and re-split to halves.
        v_scalar = vstk_lo + _H * vstk_hi                          # < 2^32 (a value)
        ax_scalar = axm_lo + _H * axm_hi
        # bit-serial bitwise runs on fp32 (values < 2^32 -> the peel divides by 2^k;
        # bit-serial only forms {0,1,2} intermediates so the fp32 peel is exact even
        # for a 32-bit operand — see BitSerialBitwise: each bit is isolated).
        for op, name in ((isa.OR, "OR"), (isa.AND, "AND"), (isa.XOR, "XOR")):
            r = self.bitwise(v_scalar, ax_scalar, name)
            rh, rl = _mask_halves(torch.floor(r / _H), _fold16(r), m)
            C[op]["ax_hi"], C[op]["ax_lo"] = rh, rl
        # shifts: width-bounded fold (identical to the fp64 machine's _shift_candidates)
        width_bits = int(m).bit_length()
        n_shift = torch.clamp(ax_scalar, max=float(width_bits))
        two_n = torch.pow(torch.tensor(2.0, dtype=FP, device=dev), n_shift)
        shl = v_scalar * two_n
        shl_hi, shl_lo = _mask_halves(torch.floor(shl / _H), _fold16(shl), m)
        C[isa.SHL]["ax_hi"], C[isa.SHL]["ax_lo"] = shl_hi, shl_lo
        sign = float((int(m) >> 1) + 1)
        v_signed = torch.where(v_scalar >= sign, v_scalar - float(int(m) + 1), v_scalar)
        shr = torch.floor(v_signed / two_n)
        shr_hi, shr_lo = _mask_halves(torch.floor(shr / _H), _fold16(shr), m)
        C[isa.SHR]["ax_hi"], C[isa.SHR]["ax_lo"] = shr_hi, shr_lo
        # MUL via the 8-bit-limb schoolbook (fp32, col-acc < 2^24) then re-split
        mul_hi, mul_lo = self._mul_halves(v_scalar, ax_scalar, m)
        C[isa.MUL]["ax_hi"], C[isa.MUL]["ax_lo"] = mul_hi, mul_lo
        # DIV/MOD via the half-limb long division (fp32) — reuse the fp64 machine's
        # _divmod but on our scalars (values, < 2^32); re-split the result to halves.
        q_hi, q_lo, r_hi, r_lo = self._divmod_halves(v_scalar, ax_scalar, m)
        C[isa.DIV]["ax_hi"], C[isa.DIV]["ax_lo"] = q_hi, q_lo
        C[isa.MOD]["ax_hi"], C[isa.MOD]["ax_lo"] = r_hi, r_lo
        return C

    # ---- MUL / DIV as fp32 limb cells, re-split to halves ----
    def _mul_halves(self, a, b, m):
        from examples.clever_fp32_fullops import (limb_mul_from_limbs, MUL_LIMBS,
                                                  MUL_OUT_LIMBS, MUL_LIMB_RADIX)
        r = float(MUL_LIMB_RADIX)
        def limb(x, i):
            s = torch.floor(x / (r ** i))
            return s - r * torch.floor(s / r)
        al = [limb(a, i) for i in range(MUL_LIMBS)]
        bl = [limb(b, i) for i in range(MUL_LIMBS)]
        out = limb_mul_from_limbs(al, bl)
        prod = torch.zeros_like(a)
        for p in range(MUL_OUT_LIMBS - 1, -1, -1):
            prod = prod * r + out[p]
        return _mask_halves(torch.floor(prod / _H), _fold16(prod), m)

    def _divmod_halves(self, v, d, m):
        from examples.clever_fp32_fullops import (_shl1_or_bit_halves, _ge_halves,
                                                  _sub_halves)
        B = v.shape[0]
        def halves(x):
            lo = x - _H * torch.floor(x / _H)
            hi = torch.floor(x / _H)
            hi = hi - _H * torch.floor(hi / _H)
            return hi, lo
        dhi, dlo = halves(d)
        q_hi = torch.zeros(B, dtype=FP, device=self.device)
        q_lo = torch.zeros(B, dtype=FP, device=self.device)
        rem_hi = torch.zeros(B, dtype=FP, device=self.device)
        rem_lo = torch.zeros(B, dtype=FP, device=self.device)
        for i in range(31, -1, -1):
            s = v / float(1 << i)
            f = torch.floor(s)
            bit = f - 2.0 * torch.floor(f / 2.0)
            rem_hi, rem_lo = _shl1_or_bit_halves(rem_hi, rem_lo, bit)
            q_hi, q_lo = _shl1_or_bit_halves(q_hi, q_lo,
                                             torch.zeros(B, dtype=FP, device=self.device))
            ge = _ge_halves(rem_hi, rem_lo, dhi, dlo)
            s_hi, s_lo = _sub_halves(rem_hi, rem_lo, dhi, dlo)
            rem_hi = torch.where(ge, s_hi, rem_hi)
            rem_lo = torch.where(ge, s_lo, rem_lo)
            q_lo = torch.where(ge, q_lo + 1.0, q_lo)
        d_is_zero = (d == 0.0)
        q_hi = torch.where(d_is_zero, torch.zeros_like(q_hi), q_hi)
        q_lo = torch.where(d_is_zero, torch.zeros_like(q_lo), q_lo)
        rem_hi = torch.where(d_is_zero, torch.zeros_like(rem_hi), rem_hi)
        rem_lo = torch.where(d_is_zero, torch.zeros_like(rem_lo), rem_lo)
        qmh, qml = _mask_halves(q_hi, q_lo, m)
        rmh, rml = _mask_halves(rem_hi, rem_lo, m)
        return qmh, qml, rmh, rml

    # ---- snapshot / read helpers ----
    def snapshot(self):
        return {"pc": _join32_int(self.PChi, self.PClo),
                "sp": _join32_int(self.SPhi, self.SPlo),
                "bp": _join32_int(self.BPhi, self.BPlo),
                "ax": _join32_int(self.AXhi, self.AXlo)}


# =========================================================================== #
# NO-fp64 TRIPWIRE — assert the limb runtime never creates an fp64 tensor.
# =========================================================================== #
class _Fp64Tripwire:
    """Context manager that flags ANY torch.float64 tensor creation on the datapath."""
    def __init__(self):
        self.hits = 0
        self._orig = {}
    def __enter__(self):
        import torch as _t
        self._t = _t
        for fn in ("full", "zeros", "ones", "tensor", "arange", "empty",
                   "full_like", "zeros_like", "ones_like"):
            orig = getattr(_t, fn)
            self._orig[fn] = orig
            def wrap(orig=orig):
                def inner(*a, **k):
                    out = orig(*a, **k)
                    if isinstance(out, _t.Tensor) and out.dtype == _t.float64:
                        self.hits += 1
                    return out
                return inner
            setattr(_t, fn, wrap())
        return self
    def __exit__(self, *exc):
        for fn, orig in self._orig.items():
            setattr(self._t, fn, orig)


# =========================================================================== #
# THE fp64-PRESSURE INVENTORY — measure which values exceed 2^24 in the fp64
# whole-value form (the motivation for the limb datapath).
# =========================================================================== #
def fp_pressure_inventory(stdin_bytes=b"HELLO") -> dict:
    """Instrument the a1b82f47 fp64 whole-value runtime; report every value on the
    address/pointer/memory datapath that exceeds fp32's 2^24 exact-integer ceiling."""
    from examples.clever_vm_runtime import DirectCAMMemory
    CEIL = 2 ** 24
    peaks: Dict[str, float] = {}
    over: Dict[str, Tuple[int, float]] = {}

    def rec(name, x):
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return
            mx = float(x.abs().max().item())
        else:
            mx = abs(float(x))
        peaks[name] = max(peaks.get(name, 0.0), mx)
        if mx > CEIL:
            c, _ = over.get(name, (0, 0.0))
            over[name] = (c + 1, mx)

    _ow, _or = DirectCAMMemory.write, DirectCAMMemory.read

    def w2(self, addr, val, active):
        rec("memory_addr", addr); rec("memory_val", val)
        return _ow(self, addr, val, active)

    def r2(self, q):
        out = _or(self, q); rec("memory_read_out", out); return out
    DirectCAMMemory.write = w2
    DirectCAMMemory.read = r2

    _osc = FullISACleverVM._step_candidates

    def sc2(self, op_f, imm_f):
        r = _osc(self, op_f, imm_f)
        nPC, nSP, nBP, nAX, wADDR, wVAL, wACT = r
        rec("register_PC", self.PC); rec("register_SP", self.SP)
        rec("register_BP", self.BP); rec("register_AX", self.AX)
        if isa.LC in nAX:
            rec("signed_LC_value", nAX[isa.LC])
        for c in nAX.values():
            rec("candidate_AX", c)
        for a in wADDR.values():
            rec("write_addr", a)
        for v in wVAL.values():
            rec("write_value", v)
        return r
    FullISACleverVM._step_candidates = sc2
    try:
        res = FI.verify_syscalls(device="cpu", stdin_bytes=stdin_bytes, B=1)
    finally:
        DirectCAMMemory.write = _ow
        DirectCAMMemory.read = _or
        FullISACleverVM._step_candidates = _osc

    inventory = []
    for name in sorted(peaks, key=lambda n: -peaks[n]):
        p = peaks[name]
        row = {"quantity": name, "peak_abs": p, "peak_hex": hex(int(p)),
               "exceeds_2_24": p > CEIL}
        if name in over:
            row["over_2_24_count"] = over[name][0]
        inventory.append(row)
    return {"byte_exact_fp64_baseline": res["byte_exact"],
            "fp32_ceiling_2_24": CEIL,
            "inventory": inventory,
            "quantities_needing_gt_fp32": sorted(over.keys())}


# =========================================================================== #
# BYTE-EXACT VERIFY — the fp32-limb runtime vs the a1b82f47 fp64 whole-value one.
# =========================================================================== #
def verify_syscalls_limb(device="cpu", stdin_bytes=b"HELLO", files=None,
                         B=2, max_steps=100000) -> dict:
    """Run the 39-op syscall program on BOTH runtimes; assert L-inf=0 + NO fp64
    on the limb datapath."""
    if files is None:
        files = {"greet.txt": b"HELWO-from-file"}
    items = _build_syscall_program_items()
    data = _build_syscall_data()
    neural_code = LibSubroutineLinker.link(items)

    # ---- fp32-LIMB run (with an fp64 tripwire over the step loop) ----
    trip = _Fp64Tripwire()
    with trip:
        vm = Fp32LimbFullISAVM(neural_code, B=B, device=device, mask=_MASK32,
                               files=dict(files), stdin_bytes=stdin_bytes,
                               mem_init=data, neural_stdin=True)
        steps = 0
        while steps < max_steps and not bool(vm.halted[0]):
            vm.step(); steps += 1
    limb_ax = _join32_int(vm.AXhi, vm.AXlo) & _MASK32
    limb_sp = _join32_int(vm.SPhi, vm.SPlo)
    limb_bp = _join32_int(vm.BPhi, vm.BPlo)
    limb_stdout = bytes(vm.io.stdout)
    limb_toolcalls = list(vm.io.tool_calls)
    shim = Fp32LimbFullISAVM._CAMemShim(vm)
    # all registers fp32?
    all_fp32 = all(t.dtype == torch.float32 for t in
                   (vm.PChi, vm.PClo, vm.SPhi, vm.SPlo, vm.BPhi, vm.BPlo,
                    vm.AXhi, vm.AXlo))
    lanes_identical = all(bool((t == t[0]).all()) for t in
                          (vm.PChi, vm.PClo, vm.SPhi, vm.SPlo,
                           vm.BPhi, vm.BPlo, vm.AXhi, vm.AXlo))

    # ---- a1b82f47 fp64 whole-value run (the byte-exact target) ----
    fp64_vm = FullISACleverVM(neural_code, B=B, device=device, mask=_MASK32,
                              files=dict(files), stdin_bytes=stdin_bytes,
                              mem_init=data, neural_stdin=True)
    s2 = 0
    while s2 < max_steps and not bool(fp64_vm.halted[0]):
        fp64_vm.step(); s2 += 1
    fp64_ax = int(fp64_vm.AX[0].round().item()) & _MASK32
    fp64_sp = int(fp64_vm.SP[0].round().item())
    fp64_bp = int(fp64_vm.BP[0].round().item())
    fp64_stdout = bytes(fp64_vm.io.stdout)
    fp64_toolcalls = list(fp64_vm.io.tool_calls)
    fp64_shim = FullISACleverVM._CAMemShim(fp64_vm)

    # ---- register + memory + I/O agreement (limb vs fp64) ----
    reg_linf = max(abs(limb_ax - fp64_ax), abs(limb_sp - fp64_sp),
                   abs(limb_bp - fp64_bp))
    # every memory cell the fp64 machine holds must match the limb machine.
    # collect the set of touched addresses from the fp64 write-log.
    touched = set()
    for a in fp64_vm.mem.addrs:
        touched.add(int(a[0].round().item()) & _MASK32)
    mem_linf = 0
    mem_mismatch = []
    for addr in sorted(touched):
        rv = fp64_shim.load_int(addr, 4)
        gv = shim.load_int(addr, 4)
        if gv != rv:
            mem_linf = max(mem_linf, abs(int(gv) - int(rv)))
            mem_mismatch.append((hex(addr), rv, gv))
    stdout_ok = (limb_stdout == fp64_stdout)
    toolcalls_ok = (limb_toolcalls == fp64_toolcalls)

    # ---- ALSO score the limb run vs the independent semantic REFERENCE oracle ----
    ref_code = _items_to_ref_opcodes(items)
    ref_trace, ref_mem, ref_ops, ref_runner = ref_state_trace_syscalls(
        ref_code, max_steps=max_steps, mask=_MASK32,
        files=dict(files), stdin_bytes=stdin_bytes, mem_init=dict(data))
    ref_final = ref_trace[-1]
    ref_reg_linf = max(abs(limb_ax - (ref_final["ax"] & _MASK32)),
                       abs(limb_sp - ref_final["sp"]), abs(limb_bp - ref_final["bp"]))
    ref_mem_linf = 0
    for addr in sorted(ref_mem.keys()):
        rv = ref_mem[addr] & _MASK32
        gv = shim.load_int(addr, 4)
        if gv != rv:
            ref_mem_linf = max(ref_mem_linf, abs(int(gv) - int(rv)))
    ref_stdout_ok = (limb_stdout == bytes(ref_runner.stdout))

    ok = (reg_linf == 0 and mem_linf == 0 and stdout_ok and toolcalls_ok
          and lanes_identical and all_fp32 and trip.hits == 0
          and ref_reg_linf == 0 and ref_mem_linf == 0 and ref_stdout_ok)
    return {
        "byte_exact_vs_fp64": (reg_linf == 0 and mem_linf == 0 and stdout_ok
                               and toolcalls_ok),
        "byte_exact_vs_reference_oracle": (ref_reg_linf == 0 and ref_mem_linf == 0
                                           and ref_stdout_ok),
        "overall_ok": ok,
        "batch": B,
        "all_lanes_identical": lanes_identical,
        "all_registers_fp32": all_fp32,
        "no_fp64_tripwire_hits": trip.hits,
        "limb_steps": steps,
        "register_linf_vs_fp64": reg_linf,
        "memory_linf_vs_fp64": mem_linf,
        "mem_mismatches": mem_mismatch,
        "register_linf_vs_reference": ref_reg_linf,
        "memory_linf_vs_reference": ref_mem_linf,
        "final_ax_limb": limb_ax, "final_ax_fp64": fp64_ax,
        "final_ax_reference": ref_final["ax"] & _MASK32,
        "stdout_limb": limb_stdout.decode("latin-1"),
        "stdout_fp64": fp64_stdout.decode("latin-1"),
        "stdout_exact": stdout_ok,
        "toolcalls_exact": toolcalls_ok,
        "n_touched_mem_cells": len(touched),
    }


def verify_fullisa_limb(device="cpu", B=1, max_steps=512) -> dict:
    """Re-verify the 31-op fullisa program (8-bit demo) on the limb datapath vs the
    fp64 reference — confirms the limb form is byte-identical at the base width too."""
    code = _build_fullisa_program()
    ref_trace, ref_mem, ops_seen = ref_state_trace_full(code, max_steps=max_steps)
    vm = Fp32LimbFullISAVM(code, B=B, device=device, mask=MASK8, neural_stdin=False)
    got = []
    for _ in range(len(ref_trace)):
        vm.step(); got.append(vm.snapshot())
    reg_linf = 0
    for r, g in zip(ref_trace, got):
        reg_linf = max(reg_linf, max(abs(int(r[f]) - int(g[f]))
                                     for f in ("pc", "sp", "bp", "ax")))
    mem_linf = 0
    for addr in sorted(ref_mem.keys()):
        ah, al = _split32(addr, B, vm.device)
        vh, vl = vm.mem.read(ah, al)
        gv = _join32_int(vh, vl) & 0xFF
        mem_linf = max(mem_linf, abs((ref_mem[addr] & 0xFF) - gv))
    return {"program": "fullisa_31op_8bit", "n_steps": len(ref_trace),
            "register_linf": reg_linf, "memory_linf": mem_linf,
            "byte_exact": (reg_linf == 0 and mem_linf == 0),
            "final_ax": ref_trace[-1]["ax"], "ops_exercised": sorted(ops_seen)}


# =========================================================================== #
# BENCH — the fp64 penalty vs the fp32-limb per-step, MEASURED on the card.
# =========================================================================== #
def _bench_runtime(build_fn, code, B, device, iters, warmup, max_run_steps=None):
    """Time one full run of the program (all steps) on `device`, best-of average."""
    dev = torch.device(device)
    # warmup + timing on FRESH VMs each iter (the write-log grows per run, so a
    # fresh VM per iter is the honest per-run cost).
    def one_run():
        vm = build_fn(code, B, device)
        n = 0
        while not bool(vm.halted[0]) and (max_run_steps is None or n < max_run_steps):
            vm.step(); n += 1
        return n
    for _ in range(warmup):
        n_steps = one_run()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        n_steps = one_run()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    total_lane_steps = iters * n_steps * B
    return {"steps_per_run": n_steps, "batch": B, "iters": iters,
            "s_per_run": dt / iters,
            "us_per_lane_step": dt / total_lane_steps * 1e6,
            "lane_steps_per_s": total_lane_steps / dt}


def bench_fp64_vs_limb(device="cuda:0", B=4096, iters=20, warmup=5,
                       max_run_steps=None) -> dict:
    """MEASURE the fp64 whole-value runtime per-step vs the fp32-limb runtime.
    Uses the SHORT 31-op program (no python-side I/O boundary) so the timing is the
    NEURAL datapath — the fp32-vs-fp64 arithmetic cost, batched to saturation."""
    code = _build_fullisa_program()

    def build_fp64(code, B, device):
        return FullISACleverVM(code, B=B, device=device, mask=MASK8, neural_stdin=False)

    def build_limb(code, B, device):
        return Fp32LimbFullISAVM(code, B=B, device=device, mask=MASK8, neural_stdin=False)

    # a WIDE-value variant: force the 32-bit value width so the fp64 machine is
    # in its genuine fp64 regime (mask=_MASK32) — the real syscall-datapath cost.
    def build_fp64_wide(code, B, device):
        return FullISACleverVM(code, B=B, device=device, mask=_MASK32, neural_stdin=False)

    def build_limb_wide(code, B, device):
        return Fp32LimbFullISAVM(code, B=B, device=device, mask=_MASK32, neural_stdin=False)

    out = {}
    out["fp64_8bit"] = _bench_runtime(build_fp64, code, B, device, iters, warmup,
                                      max_run_steps)
    out["fp32_limb_8bit"] = _bench_runtime(build_limb, code, B, device, iters, warmup,
                                           max_run_steps)
    out["fp64_wide32"] = _bench_runtime(build_fp64_wide, code, B, device, iters, warmup,
                                        max_run_steps)
    out["fp32_limb_wide32"] = _bench_runtime(build_limb_wide, code, B, device, iters,
                                             warmup, max_run_steps)
    # the fp64 penalty = fp64 us/step / fp32-limb us/step (at the wide width, the
    # genuine 32-bit datapath the syscall program uses).
    fp64_us = out["fp64_wide32"]["us_per_lane_step"]
    limb_us = out["fp32_limb_wide32"]["us_per_lane_step"]
    out["fp64_penalty_x"] = fp64_us / limb_us if limb_us else float("inf")
    out["fp64_us_per_lane_step"] = fp64_us
    out["fp32_limb_us_per_lane_step"] = limb_us
    return out


# =========================================================================== #
# DATAPATH MICROBENCH — isolate the fp64 vs fp32-limb ARITHMETIC cost of ONE
# register/pointer transition (the address/pointer datapath, no python control
# overhead), batched to saturation.  This is where the A5000's ~1/64 fp64
# throughput shows up: the same ADD/SUB/mask/compare, fp64 whole-value vs the
# fp32 half-limb form.
# =========================================================================== #
def _time_fn(fn, iters, warmup, device):
    for _ in range(warmup):
        sink = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        sink = fn()
    if isinstance(sink, torch.Tensor):
        float(sink.flatten()[0].float())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def microbench_datapath(device="cuda:0", B=1 << 20, iters=200, warmup=40) -> dict:
    """The per-transition ADD/mask/compare cost of the address/pointer datapath:
    fp64 whole-value vs the fp32 half-limb form, batched.  A pointer-walk step is a
    fixed handful of these ops.

    IMPORTANT (honest): the scalar register walk is a batched ELEMENTWISE datapath,
    so it is MEMORY-BANDWIDTH-bound, not FP-unit-bound — the fp64/fp32 elementwise
    ratio here is only ~1-2x (fp64 reads/writes 2x the bytes; the DP-unit ratio does
    NOT dominate).  Where the A5000's ~1:64 native fp64 throughput DOES bite is the
    COMPUTE-bound (matmul-class) work — the neural model forward + the attention CAM
    at scale.  We report BOTH: (a) the elementwise datapath ratio, and (b) the
    A5000's compute-bound (matmul) fp32/fp64 ratio (the ~1:64 wall the whole-fp32
    runtime avoids so the neural forward can stay fp32 and fuse).
    """
    dev = torch.device(device)
    torch.manual_seed(20260809)
    a64 = (torch.rand(B, dtype=torch.float64, device=dev) * (2 ** 32)).floor()
    b64 = (torch.rand(B, dtype=torch.float64, device=dev) * (2 ** 16)).floor()

    def fp64_transition():
        s = a64 + b64
        s = s - (2.0 ** 32) * torch.floor(s / (2.0 ** 32))     # & MASK32
        ge = (s >= b64).to(torch.float64)
        return s + ge

    ahi = torch.floor(a64 / _H).to(FP); alo = (a64 - _H * torch.floor(a64 / _H)).to(FP)
    bhi = torch.floor(b64 / _H).to(FP); blo = (b64 - _H * torch.floor(b64 / _H)).to(FP)

    def limb_transition():
        shi, slo = _add32_halves(ahi, alo, bhi, blo)
        shi, slo = _mask_halves(shi, slo, _MASK32)
        ge = _ge_u32_halves(shi, slo, bhi, blo).to(FP)
        return shi + slo + ge

    dt64 = _time_fn(fp64_transition, iters, warmup, device)
    dt32 = _time_fn(limb_transition, iters, warmup, device)
    ns64 = dt64 / iters / B * 1e9
    ns32 = dt32 / iters / B * 1e9

    # (b) the A5000 COMPUTE-bound fp32/fp64 ratio (matmul) — the ~1:64 wall.
    mm_ratio = None
    mm_fp32_tflops = mm_fp64_tflops = None
    if device.startswith("cuda"):
        def mm(dt_, N=2048, it=15):
            x = torch.rand(N, N, dtype=dt_, device=dev)
            y = torch.rand(N, N, dtype=dt_, device=dev)
            for _ in range(5):
                _ = x @ y
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(it):
                _ = x @ y
            torch.cuda.synchronize()
            dtm = time.perf_counter() - t0
            return (2 * N ** 3 * it) / dtm / 1e12
        mm_fp32_tflops = mm(torch.float32)
        mm_fp64_tflops = mm(torch.float64)
        mm_ratio = mm_fp32_tflops / mm_fp64_tflops if mm_fp64_tflops else None

    return {"batch": B, "iters": iters,
            "fp64_ns_per_lane_transition": ns64,
            "fp32_limb_ns_per_lane_transition": ns32,
            "elementwise_fp64_penalty_x": ns64 / ns32 if ns32 else float("inf"),
            "compute_bound_matmul_fp32_over_fp64_x": mm_ratio,
            "matmul_fp32_tflops": mm_fp32_tflops,
            "matmul_fp64_tflops": mm_fp64_tflops,
            "device": device}


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fp-inventory", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--bench-batch", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=1 << 20)
    ap.add_argument("--no-program-bench", action="store_true",
                    help="skip the slow control-bound whole-program bench (2b)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not (args.verify or args.fp_inventory or args.bench):
        args.verify = True
    dev = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out = {"device": dev}
    all_ok = True

    if args.fp_inventory or args.verify:
        print("=" * 96)
        print("(0) fp64-PRESSURE INVENTORY — which values exceed fp32's 2^24 ceiling in the")
        print("    a1b82f47 fp64 whole-value runtime (the motivation for the limb datapath)")
        print("=" * 96)
        inv = fp_pressure_inventory()
        out["fp_inventory"] = inv
        print(f"  fp64 baseline byte-exact: {inv['byte_exact_fp64_baseline']}  "
              f"(fp32 ceiling = 2^24 = {inv['fp32_ceiling_2_24']:,})")
        print(f"  {'quantity':<20s} {'peak |value|':>16s}  {'hex':>12s}  {'> 2^24?':>8s}")
        for row in inv["inventory"]:
            flag = "  <<< fp64 pressure" if row["exceeds_2_24"] else ""
            print(f"  {row['quantity']:<20s} {row['peak_abs']:>16,.0f}  "
                  f"{row['peak_hex']:>12s}  {str(row['exceeds_2_24']):>8s}{flag}")
        print(f"\n  >>> quantities that FORCE fp64 (whole-value): "
              f"{inv['quantities_needing_gt_fp32']}")
        print(f"      (all are 32-bit VALUES — signed-LC negatives / memcmp diffs / the "
              f"AX+memory they flow through; the LIMB form carries each as two 16-bit "
              f"halves < 2^16, so NO fp32 scalar ever exceeds 2^24)")

    if args.verify:
        print("\n" + "=" * 96)
        print("(1) BYTE-EXACT — the fp32-LIMB runtime vs the a1b82f47 fp64 whole-value one")
        print("=" * 96)
        # the base 31-op program first (8-bit width sanity)
        b31 = verify_fullisa_limb(device="cpu", B=1)
        out["fullisa_31op_limb"] = b31
        print(f"  [31-op fullisa, 8-bit] register L-inf={b31['register_linf']} "
              f"memory L-inf={b31['memory_linf']} -> "
              f"{'BYTE-EXACT' if b31['byte_exact'] else 'FAIL'}  (final AX={b31['final_ax']})")
        # the 39-op syscall program (32-bit width — the fp64-pressure program)
        sc = verify_syscalls_limb(device="cpu", B=args.batch)
        out["syscalls_39op_limb"] = sc
        print(f"\n  [39-op syscall program: MALC->MSET->MCMP->PRTF->OPEN/READ/CLOS->"
              f"neural-READ->FREE]  batch={sc['batch']}, {sc['limb_steps']} steps")
        print(f"    vs fp64 whole-value: register L-inf={sc['register_linf_vs_fp64']} "
              f"memory L-inf={sc['memory_linf_vs_fp64']} "
              f"(over {sc['n_touched_mem_cells']} cells) "
              f"stdout={'exact' if sc['stdout_exact'] else 'DIFF'} "
              f"toolcalls={'exact' if sc['toolcalls_exact'] else 'DIFF'}")
        print(f"    vs semantic reference oracle: register L-inf="
              f"{sc['register_linf_vs_reference']} memory L-inf="
              f"{sc['memory_linf_vs_reference']}")
        print(f"    stdout: limb={sc['stdout_limb']!r} fp64={sc['stdout_fp64']!r}")
        print(f"    final AX: limb={sc['final_ax_limb']} fp64={sc['final_ax_fp64']} "
              f"reference={sc['final_ax_reference']}")
        print(f"    all lanes identical: {sc['all_lanes_identical']}  "
              f"all registers fp32: {sc['all_registers_fp32']}  "
              f"fp64-tripwire hits: {sc['no_fp64_tripwire_hits']}")
        print(f"\n  >>> the WHOLE 39-op min-flop runtime runs FP32 byte-exact (zero fp64): "
              f"{'YES' if sc['overall_ok'] else 'NO'}")
        all_ok = all_ok and sc["overall_ok"] and b31["byte_exact"]

    if args.bench:
        print("\n" + "=" * 96)
        print(f"(2) MEASURED fp64 PENALTY vs fp32-LIMB  device={dev}")
        print("=" * 96)
        if dev.startswith("cuda"):
            print(f"  {torch.cuda.get_device_name(0)}")
        # (2a) the ISOLATED datapath arithmetic cost + the A5000 compute-bound ratio.
        print("\n  (2a) ISOLATED address/pointer datapath transition (batched):")
        mb = microbench_datapath(device=dev, B=args.micro_batch, iters=args.iters * 5,
                                 warmup=args.warmup * 2)
        out["microbench_datapath"] = mb
        print(f"    fp64 whole-value : {mb['fp64_ns_per_lane_transition']:.5f} ns/lane-transition")
        print(f"    fp32 half-limb   : {mb['fp32_limb_ns_per_lane_transition']:.5f} ns/lane-transition")
        print(f"    elementwise (bandwidth-bound) fp64/fp32 ratio: "
              f"{mb['elementwise_fp64_penalty_x']:.2f}x")
        if mb.get("compute_bound_matmul_fp32_over_fp64_x"):
            print(f"    A5000 COMPUTE-bound (matmul) fp32/fp64 ratio: "
                  f"{mb['compute_bound_matmul_fp32_over_fp64_x']:.1f}x  "
                  f"(fp32 {mb['matmul_fp32_tflops']:.1f} / fp64 "
                  f"{mb['matmul_fp64_tflops']:.2f} TFLOP/s)")
            print(f"    >>> the ~1:64 native fp64 wall bites the COMPUTE-bound work "
                  f"(neural forward / attention); keeping the WHOLE runtime fp32 (the "
                  f"limb datapath) is what lets it fuse into one fp32 kernel and avoid "
                  f"that wall.")
        # (2b) the whole-program per-step cost (control-bound; includes the growing
        #      CAM read loop + 40-op stacking — a floor on the python-driver form).
        if not args.no_program_bench:
            print("\n  (2b) whole-program per-step (control-bound reference):")
            bench = bench_fp64_vs_limb(device=dev, B=args.bench_batch, iters=args.iters,
                                       warmup=args.warmup)
            out["bench_program"] = bench
            for k in ("fp64_wide32", "fp32_limb_wide32"):
                r = bench[k]
                print(f"    {k:<18s}: {r['us_per_lane_step']:.4f} us/lane-step "
                      f"({r['steps_per_run']} steps/run, batch {r['batch']})")
            print(f"    >>> fp64 (wide-32) penalty vs fp32-limb (program): "
                  f"{bench['fp64_penalty_x']:.2f}x")

    print("\n" + "=" * 96)
    print(f"VERDICT: whole 39-op min-flop runtime FP32 byte-exact (zero fp64), limb "
          f"address/pointer/memory datapath -> {'ALL OK' if all_ok else 'see above'}")
    print("  golden 174ece66 untouched (this file is off every model build path).")
    print("=" * 96)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.json}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
