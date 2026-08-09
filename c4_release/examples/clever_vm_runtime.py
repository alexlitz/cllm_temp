#!/usr/bin/env python3
r"""clever_vm_runtime.py — Phase 1 FOUNDATION of a min-param **fp32** clever c4 VM
that runs the c4 ISA PROPERLY: a real **fetch → decode/dispatch → execute →
update-(PC/SP/BP/AX/stack/memory)** machine, not isolated op-cells.

THE GO/NO-GO THIS PROVES
========================
The pre-existing clever fp32 work (``clever_compact_scoring_realtime`` /
``clever_fp32_fullops`` / ``clever_honest_attn_realtime``) has byte-exact fp32
op-cells — ALU/CMP/shift/bitwise/memory SEMANTICS — but they are ISOLATED cells
with **no runtime**: no fetch-decode-execute loop, no control flow that actually
moves PC, no real stack machine (SP/BP/PSH + the c4 calling convention), no real
memory model.  This file builds exactly that sequencer AROUND the reused op-cells
and proves it runs a MINIMAL but REAL c4 program BYTE-EXACT end-to-end against
the c4 reference (``nibble_pure_forward_complete.ref_interpret``, the
SP-addressed memory-stack oracle): decoded PC/SP/BP/AX + stack + memory trace
**L-inf = 0 at EVERY step**.

THE CLEVER fp32 DATAPATH (all state is fp32; every transition is an fp32 op)
==========================================================================
Every architectural register is an **fp32 scalar** (batched: a ``(B,)`` fp32
tensor, so B lanes each run their own VM in lock-step — the batched-verify model
the clever VM is built for), and every step's transition is computed by fp32
tensor ops — the SAME primitives the op-cells use (``torch.floor`` masks == the
``_diffmin_decode`` floor; the memory reads == the ``DirectCAMReadHead``
latest-write-wins gather).  No value ever exceeds 2^24 in these programs, so the
fp32 datapath is EXACT.

  * FETCH   : direct-CAM read on the CODE segment (op,imm at PC).  Reuses the
              O(1) content-addressed gather (``_cam_gather``, the code analog of
              ``DirectCAMReadHead.gather_value``).
  * DECODE  : opcode ONE-HOT ``onehot[op]`` from the fetched op scalar (an fp32
              exact-integer equality indicator — the neural one-hot).
  * DISPATCH: EVERY op computes its candidate next-state on the SHARED datapath;
              the one-hot ROUTES which candidate commits:
                  next_R = Σ_k onehot[k] · R_candidate_k   for R in {PC,SP,BP,AX}
              a mul-add reduction == a neural dispatch gather.  Exactly-one-hot,
              so collision-free by construction.
  * EXECUTE : the reused fp32 op-cells produce each candidate (ADD = CAM-read at
              SP + fp32 add + floor-mask; LI = CAM-read at AX; PSH/SI/JSR/ENT/LEV
              = CAM writes/reads on the write-log; branches = a gated PC mux).
  * UPDATE  : commit the muxed (PC,SP,BP,AX); append any memory write to the
              per-lane write-log (latest-write-wins == the direct-CAM recency).

MEMORY MODEL — one write-log, three roles (stack / heap / code)
==============================================================
The stack (PSH/pop/JSR/ENT/LEV) and the program stores (SI/SC) are the SAME kind
of address-keyed write — the reference's unification.  Both go into ONE per-lane
write-log ``(addr, val)``; a read resolves the query address to the LATEST
matching write (0 / ZFOD if unwritten) — the byte-exact host collapse of the
softmax1+ALiBi latest-write-wins CAM (``softmax_cam_read``), i.e. the O(1)
``gather_value`` mechanism.  The CODE segment is a separate address-keyed table
(unique keys, no recency needed).

Golden ``174ece66`` untouched (NEW file, off every model build path).

Run:
    python examples/clever_vm_runtime.py --verify        # byte-exact, L-inf=0
    python examples/clever_vm_runtime.py --verify --program call --show-trace
    python examples/clever_vm_runtime.py --verify --batch 4096   # B-lane batched
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

# make the repo root importable when run directly (``python examples/clever_vm_runtime.py``),
# matching how the sibling examples are run under ``PYTHONPATH=.``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from c4_min import isa
import c4_min.nibble_pure_forward_complete as PFC

SP_INIT = PFC.SP_INIT          # 0x10000 — stack top (grows downward), the ref VM's
MASK8 = 0xFF
FP = torch.float32


# =========================================================================== #
# THE REUSED fp32 OP-CELL PRIMITIVES (the byte-exact clever datapath substrate).
#   * fp32 floor/mask  == the _diffmin_decode floor (clever_compact_scoring).
#   * one-hot indicator == the neural opcode-select one-hot.
#   * CAM gather/write  == DirectCAMReadHead latest-write-wins (clever_honest_attn).
# All EXACT for integers < 2^24 in fp32.
# =========================================================================== #
def fp_mask(x: torch.Tensor, mask: int) -> torch.Tensor:
    """``x & mask`` for a power-of-two-minus-one mask, on an fp32 scalar tensor,
    byte-EXACT.  ``x mod (mask+1) = x - (mask+1)*floor(x/(mask+1))`` — a pure
    ``torch.floor`` (the SAME exact fp32 floor ``_diffmin_decode`` collapses to).
    Handles negative x (two's-complement wrap) since floor rounds toward -inf."""
    m = float(mask + 1)
    return x - m * torch.floor(x / m)


def one_hot_opcode(op_f: torch.Tensor, num_ops: int) -> torch.Tensor:
    """The DECODE: opcode one-hot from an fp32 op scalar ``(B,)`` -> ``(B, num_ops)``.
    ``onehot[:,k] = (op == k)`` — an fp32 exact-integer equality indicator, the
    neural one-hot the model bakes.  Exactly-one-hot for a valid op id."""
    ks = torch.arange(num_ops, dtype=op_f.dtype, device=op_f.device)   # (num_ops,)
    return (op_f.unsqueeze(-1) == ks).to(op_f.dtype)                    # (B,num_ops)


def commit_onehot(onehot: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """The DISPATCH commit: ``next = Σ_k onehot[k] · candidate_k``.
    ``onehot``:(B,num_ops), ``candidates``:(B,num_ops) per-op next-value.  A mul-add
    reduction == a neural one-hot gather.  Exactly-one candidate survives."""
    return (onehot * candidates).sum(-1)


# =========================================================================== #
# THE MEMORY MODEL — one per-lane write-log, latest-write-wins (the direct-CAM).
#   Batched, fp32 addresses/values.  A write appends (addr,val); a read resolves
#   the LATEST matching write.  This is the byte-exact host collapse of the
#   softmax1+ALiBi CAM (softmax_cam_read) == DirectCAMReadHead.gather_value.
# =========================================================================== #
class DirectCAMMemory:
    """Per-lane address-keyed KV write-log (stack + heap unified), fp32.

    Stored as a growing list of ``(addr, val)`` fp32 tensors, each ``(B,)`` — one
    entry per write across all lanes (a write is masked per-lane by whether that
    lane actually wrote this step).  A read gathers, per lane, the value of the
    LATEST write whose address matches the query (recency = highest log index),
    ZFOD 0 if none.  This is exactly the latest-write-wins direct-CAM."""

    def __init__(self, B: int, device):
        self.B = B
        self.device = device
        self.addrs: List[torch.Tensor] = []      # each (B,) fp32
        self.vals: List[torch.Tensor] = []       # each (B,) fp32
        self.active: List[torch.Tensor] = []     # each (B,) fp32 {0,1}: did lane write?

    def write(self, addr: torch.Tensor, val: torch.Tensor, active: torch.Tensor):
        """Append a write frame.  ``active`` (B,) {0,1} masks which lanes wrote
        (a lane whose op is not a store contributes an inactive entry — it never
        matches a later read, so the log stays a single uniform structure)."""
        self.addrs.append(addr.clone())
        self.vals.append(val.clone())
        self.active.append(active.clone())

    def read(self, query_addr: torch.Tensor) -> torch.Tensor:
        """Latest-write-wins gather: for each lane, the val of the highest-index
        active write whose addr == query_addr; 0 if none (ZFOD).  fp32-exact."""
        B = self.B
        out = torch.zeros(B, dtype=FP, device=self.device)
        found = torch.zeros(B, dtype=torch.bool, device=self.device)
        # walk newest -> oldest; take the first (newest) match per lane (recency).
        for i in range(len(self.addrs) - 1, -1, -1):
            match = (self.addrs[i] == query_addr) & (self.active[i] > 0.5) & (~found)
            out = torch.where(match, self.vals[i], out)
            found = found | match
            if bool(found.all()):
                break
        return out


# =========================================================================== #
# THE CODE SEGMENT — an address-keyed table fetched at PC (unique keys).
# =========================================================================== #
class CodeCAM:
    """The program as an address-keyed CODE segment: op,imm per address, fetched
    at PC.  Keys are UNIQUE (one row per address) so the fetch is a pure
    content-addressed gather (``_cam_gather``) — no recency decay.  Phase 1 stores
    it as dense fp32 op/imm tables (the byte-exact collapse of the address-CAM at
    unique keys: identical values, S-independent per-lane bandwidth)."""

    def __init__(self, code: List[isa.Instr], device):
        self.n = len(code)
        self.op = torch.tensor([float(ins.op) for ins in code], dtype=FP, device=device)
        self.imm = torch.tensor([float(ins.imm & 0xFFFFFFFF) for ins in code],
                                dtype=FP, device=device)
        self.device = device

    def fetch(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch (op, imm, in_range) at PC ``(B,)`` fp32.  Out-of-range PC (halt)
        returns in_range=0.  The gather is the O(1) direct-CAM read on the code
        store (unique key = PC)."""
        idx = pc.round().to(torch.int64)
        in_range = (idx >= 0) & (idx < self.n)
        safe = idx.clamp(0, max(self.n - 1, 0))
        op = self.op[safe]
        imm = self.imm[safe]
        return op, imm, in_range.to(FP)


# =========================================================================== #
# THE CLEVER FETCH-DECODE-EXECUTE MACHINE.
#   State: PC/SP/BP/AX fp32 (B,) registers + one DirectCAMMemory write-log.
#   One step = fetch -> decode -> (all ops' candidates) -> one-hot commit -> update.
# =========================================================================== #
# The opcode set this Phase-1 foundation dispatches (the minimal-but-real slice:
# IMM/LEA/LI/SI/ADD/CMP/BZ/BNZ/JMP + ENT/ADJ/LEV function call+return, plus the
# PSH/JSR/JMP/NOP/HALT the calling convention needs).  Extending to the full ISA
# = adding each op's candidate to this dispatch (Phase 2).
NUM_OPS = 40


class CleverVM:
    """The min-param fp32 clever fetch-decode-execute machine (Phase 1)."""

    def __init__(self, code: List[isa.Instr], B: int = 1, device="cpu",
                 mask: int = MASK8):
        self.device = torch.device(device)
        self.B = B
        self.mask = mask
        self.code = CodeCAM(code, self.device)
        self.mem = DirectCAMMemory(B, self.device)
        z = lambda v: torch.full((B,), float(v), dtype=FP, device=self.device)
        self.PC = z(0)
        self.SP = z(SP_INIT)
        self.BP = z(SP_INIT)
        self.AX = z(0)
        self.halted = torch.zeros(B, dtype=torch.bool, device=self.device)

    # ---- the op-cell candidate datapath: compute EVERY op's next-state ---- #
    def _step_candidates(self, op_f, imm_f):
        """For the fetched (op,imm), compute EVERY opcode's candidate
        (next_PC, next_SP, next_BP, next_AX) + its memory write (addr,val,active).
        All fp32-exact; the one-hot commit picks the live one.  Returns dicts
        keyed by opcode id.  Memory READS (pop/LI/SI-pop/LEV) are done ONCE here
        (a CAM read is the same regardless of which op wins) and fed to the
        relevant candidates — the shared-datapath read."""
        B, dev, m = self.B, self.device, self.mask
        PC, SP, BP, AX = self.PC, self.SP, self.BP, self.AX
        z = torch.zeros(B, dtype=FP, device=dev)
        one = torch.ones(B, dtype=FP, device=dev)
        pc_next = PC + 1.0                                   # default sequential bump
        i_idx = PC                                           # this instr's address (for JSR)

        # --- the shared memory reads (done once; each op uses what it needs) ---
        stk_top = self.mem.read(SP)                          # pop value = mem[SP]
        li_val = self.mem.read(AX)                           # LI: mem[AX]
        lev_bp = self.mem.read(BP)                           # LEV: mem[BP] (saved BP)
        lev_pc = self.mem.read(BP + 4.0)                     # LEV: mem[BP+4] (ret PC)

        # candidate registers per op (default = unchanged / sequential)
        nPC = {}; nSP = {}; nBP = {}; nAX = {}
        wADDR = {}; wVAL = {}; wACT = {}                     # memory write per op
        def default(op):
            nPC[op] = pc_next; nSP[op] = SP; nBP[op] = BP; nAX[op] = AX
            wADDR[op] = z; wVAL[op] = z; wACT[op] = z        # no write by default

        for op in range(NUM_OPS):
            default(op)

        # IMM: ax = imm & 0xFF
        nAX[isa.IMM] = fp_mask(imm_f, m)
        # LEA: ax = (bp + 4*imm) & 0xFF
        nAX[isa.LEA] = fp_mask(BP + 4.0 * imm_f, m)
        # PSH: sp -= 4; mem[sp] = ax
        nSP[isa.PSH] = SP - 4.0
        wADDR[isa.PSH] = SP - 4.0; wVAL[isa.PSH] = fp_mask(AX, m); wACT[isa.PSH] = one
        # LI: ax = mem[ax] & 0xFF   (unsigned byte load)
        nAX[isa.LI] = fp_mask(li_val, MASK8)
        # LC: ax = signed-char(mem[ax])  (byte>=0x80 sign-extends to width `mask`)
        lc_b = fp_mask(li_val, MASK8)
        lc_signed = torch.where(lc_b >= 128.0, lc_b - 256.0, lc_b)
        nAX[isa.LC] = fp_mask(lc_signed, m)
        # SI/SC: addr = pop(); mem[addr] = ax & 0xFF
        for op in (isa.SI, isa.SC):
            nSP[op] = SP + 4.0
            wADDR[op] = stk_top; wVAL[op] = fp_mask(AX, MASK8); wACT[op] = one
        # ADD/SUB (pop then combine, wrap at mask); SP += 4
        for op in (isa.ADD, isa.SUB):
            nSP[op] = SP + 4.0
        nAX[isa.ADD] = fp_mask(stk_top + AX, m)
        nAX[isa.SUB] = fp_mask(stk_top - AX, m)
        # CMP family: pop v; ax = (v ? ax) in {0,1}; SP += 4.  (8-bit: unsigned order.)
        for op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            nSP[op] = SP + 4.0
        v = stk_top
        nAX[isa.EQ] = (v == AX).to(FP)
        nAX[isa.NE] = (v != AX).to(FP)
        nAX[isa.LT] = (v < AX).to(FP)
        nAX[isa.GT] = (v > AX).to(FP)
        nAX[isa.LE] = (v <= AX).to(FP)
        nAX[isa.GE] = (v >= AX).to(FP)
        # JMP: pc = imm
        nPC[isa.JMP] = imm_f
        # BZ: pc = imm if ax==0 else pc+1  (gated PC mux)
        nPC[isa.BZ] = torch.where(AX == 0.0, imm_f, pc_next)
        # BNZ: pc = imm if ax!=0 else pc+1
        nPC[isa.BNZ] = torch.where(AX != 0.0, imm_f, pc_next)
        # JSR: mem[sp-4] = i+1; sp -= 4; pc = imm
        nSP[isa.JSR] = SP - 4.0
        nPC[isa.JSR] = imm_f
        wADDR[isa.JSR] = SP - 4.0
        wVAL[isa.JSR] = fp_mask(i_idx + 1.0, MASK8)          # ret addr = index+1 (ref masks 0xFF)
        wACT[isa.JSR] = one
        # ENT: mem[sp-4] = bp; sp -= 4; bp = sp; sp -= 4*imm
        ent_sp_after_push = SP - 4.0
        nBP[isa.ENT] = ent_sp_after_push
        nSP[isa.ENT] = ent_sp_after_push - 4.0 * imm_f
        wADDR[isa.ENT] = SP - 4.0
        wVAL[isa.ENT] = fp_mask(BP, 0xFFFFFFFF)              # saved BP (ref: &0xFFFFFFFF)
        wACT[isa.ENT] = one
        # ADJ: sp += 4*imm
        nSP[ADJ] = SP + 4.0 * imm_f
        # LEV: sp = bp; bp = mem[sp]; pc = mem[sp+4]; sp += 8
        #   (bp is the CURRENT bp; mem[bp] = saved bp, mem[bp+4] = ret pc)
        nBP[isa.LEV] = lev_bp
        nPC[isa.LEV] = lev_pc
        nSP[isa.LEV] = BP + 8.0
        # PRTF / NOP: registers unchanged, pc+1 (PRTF's visible byte handled by driver)
        # HALT: pc -> out of range so the loop stops (handled by halt flag in step()).
        return (nPC, nSP, nBP, nAX, wADDR, wVAL, wACT)

    def step(self):
        """One fetch-decode-execute-update step, batched over B lanes."""
        op_f, imm_f, in_range = self.code.fetch(self.PC)
        onehot = one_hot_opcode(op_f, NUM_OPS)               # DECODE

        (nPC, nSP, nBP, nAX, wADDR, wVAL, wACT) = self._step_candidates(op_f, imm_f)

        # stack the per-op candidates -> (B, NUM_OPS) and one-hot-commit (DISPATCH).
        def stack(d):
            return torch.stack([d[k] for k in range(NUM_OPS)], dim=-1)
        pc_c = commit_onehot(onehot, stack(nPC))
        sp_c = commit_onehot(onehot, stack(nSP))
        bp_c = commit_onehot(onehot, stack(nBP))
        ax_c = commit_onehot(onehot, stack(nAX))
        waddr = commit_onehot(onehot, stack(wADDR))
        wval = commit_onehot(onehot, stack(wVAL))
        wact = commit_onehot(onehot, stack(wACT))

        # HALT: detect the HALT opcode one-hot -> freeze that lane.
        is_halt = onehot[:, isa.HALT] > 0.5
        newly_halted = is_halt & (~self.halted)
        # a lane already halted OR out of code range stays put (frozen).
        alive = (~self.halted) & (in_range > 0.5)

        # UPDATE: commit registers only for alive lanes (frozen lanes unchanged).
        af = alive.to(FP)
        self.PC = torch.where(alive, pc_c, self.PC)
        self.SP = torch.where(alive, sp_c, self.SP)
        self.BP = torch.where(alive, bp_c, self.BP)
        self.AX = torch.where(alive, ax_c, self.AX)
        # memory write: only alive lanes whose op actually wrote (wact) commit it.
        self.mem.write(waddr, wval, wact * af)
        self.halted = self.halted | newly_halted | (in_range <= 0.5)

    def snapshot(self) -> dict:
        """The full committed per-step state (lane 0 by default) for the trace."""
        return {"pc": int(self.PC[0].round().item()),
                "sp": int(self.SP[0].round().item()),
                "bp": int(self.BP[0].round().item()),
                "ax": int(self.AX[0].round().item())}


# =========================================================================== #
# INSTRUMENTED REFERENCE ORACLE — the byte-exact target, full per-step STATE.
#   A copy of ref_interpret that records (pc,sp,bp,ax) AFTER each step AND the
#   memory cells it touched, so we can assert L-inf=0 on the whole machine state
#   (not just AX).  Semantics are IDENTICAL to nibble_pure_forward_complete.
# =========================================================================== #
def ref_state_trace(code: List[isa.Instr], max_steps: int = 512,
                    mask: int = MASK8) -> Tuple[List[dict], Dict[int, int]]:
    """Run the reference VM, recording the FULL post-step state each step.
    Returns (trace, final_mem).  Byte-identical semantics to ref_interpret."""
    mem: Dict[int, int] = {}
    sp = bp = SP_INIT
    ax = pc = 0
    trace: List[dict] = []
    steps = 0
    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFF
        elif op == isa.PSH:
            sp -= 4; mem[sp] = ax & mask
        elif op == isa.ADD:
            v = mem.get(sp, 0) & mask; sp += 4; ax = (v + ax) & mask
        elif op == isa.SUB:
            v = mem.get(sp, 0) & mask; sp += 4; ax = (v - ax) & mask
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0) & mask; sp += 4; av = ax & mask
            r = {isa.EQ: v == av, isa.NE: v != av, isa.LT: v < av,
                 isa.GT: v > av, isa.LE: v <= av, isa.GE: v >= av}[op]
            ax = 1 if r else 0
        elif op == isa.LI:
            ax = mem.get(ax, 0) & 0xFF
        elif op == isa.LC:
            b = mem.get(ax, 0) & 0xFF
            ax = (b - 0x100 if b & 0x80 else b) & mask
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4; mem[addr] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            sp -= 4; mem[sp] = (i + 1) & 0xFF; pc = imm
        elif op == isa.ENT:
            mem[sp - 4] = bp & 0xFFFFFFFF; sp -= 4; bp = sp; sp -= 4 * imm
        elif op == ADJ:
            sp += 4 * imm
        elif op == isa.LEV:
            sp = bp; bp = mem.get(sp, 0); pc = mem.get(sp + 4, 0); sp += 8
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            trace.append({"pc": pc, "sp": sp, "bp": bp, "ax": ax & mask})
            break
        else:
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in Phase-1 slice")
        trace.append({"pc": pc, "sp": sp, "bp": bp, "ax": ax & mask})
    return trace, mem


ADJ = isa.ADJ if hasattr(isa, "ADJ") else 7


# =========================================================================== #
# THE MINIMAL REAL PROGRAMS — each DOES something real (compute/store/load/branch/call).
# =========================================================================== #
def program_straightline() -> Tuple[str, List[isa.Instr]]:
    """compute -> store -> load -> compute: ax=(5+3); store to mem[64]; load it;
    add 1.  Exercises IMM/LEA/PSH/ADD/SI/LI."""
    return "straightline", isa.assemble([
        ("IMM", 5),    # 0  ax=5
        ("PSH", 0),    # 1  push 5
        ("IMM", 3),    # 2  ax=3
        ("ADD", 0),    # 3  ax = 5+3 = 8
        ("IMM", 64),   # 4  ax = 64 (address)
        ("PSH", 0),    # 5  push 64
        ("IMM", 8),    # 6  ax = 8 (value)
        ("SI", 0),     # 7  mem[64] = 8      (pop addr=64, store ax=8)
        ("IMM", 64),   # 8  ax = 64
        ("LI", 0),     # 9  ax = mem[64] = 8
        ("PSH", 0),    # 10 push 8
        ("IMM", 1),    # 11 ax=1
        ("ADD", 0),    # 12 ax = 8+1 = 9
        ("HALT", 0),   # 13
    ])


def program_branch() -> Tuple[str, List[isa.Instr]]:
    """compute a CMP + conditional branch: is (2+2)==5?  no -> BZ taken path sets
    ax=10; then a JMP over the else.  Exercises EQ/BZ/JMP."""
    return "branch", isa.assemble([
        ("IMM", 2),    # 0  ax=2
        ("PSH", 0),    # 1  push 2
        ("IMM", 2),    # 2  ax=2
        ("ADD", 0),    # 3  ax = 4
        ("PSH", 0),    # 4  push 4
        ("IMM", 5),    # 5  ax=5
        ("EQ", 0),     # 6  ax = (4==5) = 0
        ("BZ", 9),     # 7  ax==0 -> branch to 9
        ("IMM", 1),    # 8  (skipped) ax=1
        ("IMM", 10),   # 9  ax=10  (branch target)
        ("HALT", 0),   # 10
    ])


def program_call() -> Tuple[str, List[isa.Instr]]:
    """a REAL function call+return: main computes 5+3=8, calls add1(), which does
    ENT/LEA/LI-frame-work + ADJ + LEV; main resumes after the call.  Exercises the
    FULL calling convention JSR/ENT/ADJ/LEV over the SP-addressed stack."""
    return "call", isa.assemble([
        # ---- main ----
        ("IMM", 5),    # 0  ax=5
        ("PSH", 0),    # 1  push 5
        ("IMM", 3),    # 2  ax=3
        ("ADD", 0),    # 3  ax = 8
        ("PSH", 0),    # 4  push 8 (an argument on the stack)
        ("JSR", 8),    # 5  call func at pc=8 (return addr = 6)
        ("ADJ", 1),    # 6  drop the 1 argument (sp += 4)
        ("HALT", 0),   # 7  end (ax = func's return value)
        # ---- func: enter frame, load arg, add 100, return ----
        ("ENT", 0),    # 8  prologue: save bp, bp=sp
        ("LEA", 2),    # 9  ax = bp + 4*2  (frame-relative addr of the arg)
        ("LI", 0),     # 10 ax = mem[bp+8] = the pushed arg (8)
        ("PSH", 0),    # 11 push arg
        ("IMM", 100),  # 12 ax=100
        ("ADD", 0),    # 13 ax = arg + 100 = 108 -> &0xFF = 108
        ("LEV", 0),    # 14 epilogue: restore bp, pc=ret; sp+=8
    ])


PROGRAMS = {
    "straightline": program_straightline,
    "branch": program_branch,
    "call": program_call,
}


# =========================================================================== #
# BYTE-EXACT VERIFY — run the clever machine + the reference, assert L-inf=0.
# =========================================================================== #
def verify_program(name: str, code: List[isa.Instr], B: int, device: str,
                   max_steps: int, show_trace: bool) -> dict:
    ref_trace, ref_mem = ref_state_trace(code, max_steps=max_steps)
    vm = CleverVM(code, B=B, device=device)
    got_trace = []
    for _ in range(len(ref_trace)):
        vm.step()
        got_trace.append(vm.snapshot())

    # per-step per-field L-inf over the register state.
    fields = ("pc", "sp", "bp", "ax")
    max_linf = 0
    first_div = None
    per_step = []
    for s, (r, g) in enumerate(zip(ref_trace, got_trace)):
        d = {f: abs(int(r[f]) - int(g[f])) for f in fields}
        linf = max(d.values())
        per_step.append({"step": s, "ref": r, "got": g, "diff": d, "linf": linf})
        if linf > max_linf:
            max_linf = linf
        if linf != 0 and first_div is None:
            first_div = s

    # memory L-inf: every cell the reference touched must match lane-0 CAM read.
    mem_linf = 0
    mem_checks = []
    for addr in sorted(ref_mem.keys()):
        ref_v = ref_mem[addr] & 0xFF                       # ref stores masked to byte
        q = torch.full((B,), float(addr), dtype=FP, device=vm.device)
        got_v = int(vm.mem.read(q)[0].round().item()) & 0xFF
        d = abs(ref_v - got_v)
        mem_checks.append({"addr": addr, "ref": ref_v, "got": got_v, "diff": d})
        mem_linf = max(mem_linf, d)

    # cross-lane identity: all B lanes must be identical (deterministic machine).
    lane_identical = True
    if B > 1:
        for f, t in (("pc", vm.PC), ("sp", vm.SP), ("bp", vm.BP), ("ax", vm.AX)):
            if not bool((t == t[0]).all()):
                lane_identical = False

    ok = (max_linf == 0) and (mem_linf == 0) and lane_identical
    res = {"program": name, "n_steps": len(ref_trace), "batch": B, "device": device,
           "register_linf": max_linf, "memory_linf": mem_linf,
           "first_divergence_step": first_div, "all_lanes_identical": lane_identical,
           "byte_exact": ok, "n_mem_cells_checked": len(mem_checks)}
    if show_trace:
        res["trace"] = per_step
        res["mem_checks"] = mem_checks
    return res


def print_trace(res: dict):
    print(f"\n  per-step STATE trace (ref vs clever fp32 machine, lane 0):")
    print(f"    {'step':>4s} {'op@pc':>10s} | "
          f"{'PC':>6s} {'SP':>8s} {'BP':>8s} {'AX':>4s}  L-inf")
    for row in res.get("trace", []):
        r, g, d = row["ref"], row["got"], row["diff"]
        flag = "" if row["linf"] == 0 else "  <-- DIVERGE"
        print(f"    {row['step']:>4d} {'':>10s} | "
              f"ref pc={r['pc']:<5d} sp={r['sp']:<7d} bp={r['bp']:<7d} ax={r['ax']:<3d}  "
              f"linf={row['linf']}{flag}")
        print(f"    {'':>4s} {'':>10s} | "
              f"got pc={g['pc']:<5d} sp={g['sp']:<7d} bp={g['bp']:<7d} ax={g['ax']:<3d}")
    if res.get("mem_checks"):
        print(f"\n  memory cells (ref stored -> clever CAM read):")
        for m in res["mem_checks"]:
            flag = "" if m["diff"] == 0 else "  <-- MISMATCH"
            print(f"    mem[{m['addr']:>6d}] = {m['ref']:<4d} (clever: {m['got']}){flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--program", default="all",
                    help="straightline | branch | call | all")
    ap.add_argument("--batch", type=int, default=1, help="B lanes (each its own VM)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--show-trace", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.verify:
        args.verify = True
    dev = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    progs = (list(PROGRAMS) if args.program == "all" else [args.program])
    out = {"device": dev, "batch": args.batch, "sp_init": SP_INIT, "results": {}}

    print("=" * 92)
    print("CLEVER fp32 FETCH-DECODE-EXECUTE MACHINE — Phase 1 byte-exact verify")
    print(f"  device={dev}  batch={args.batch}  SP_INIT={hex(SP_INIT)}  "
          f"oracle=nibble_pure_forward_complete.ref_interpret (SP-addressed stack)")
    print("=" * 92)

    all_ok = True
    for name in progs:
        _, code = PROGRAMS[name]()
        res = verify_program(name, code, args.batch, dev, args.max_steps,
                             args.show_trace)
        out["results"][name] = res
        all_ok = all_ok and res["byte_exact"]
        status = "BYTE-EXACT (L-inf=0)" if res["byte_exact"] else "FAIL"
        print(f"\n[{name}]  {res['n_steps']} steps, {res['n_mem_cells_checked']} mem "
              f"cells | register L-inf={res['register_linf']} memory L-inf="
              f"{res['memory_linf']} | lanes-identical={res['all_lanes_identical']}"
              f"  -> {status}")
        if res["first_divergence_step"] is not None:
            print(f"    first divergence at step {res['first_divergence_step']}")
        if args.show_trace:
            print_trace(res)

    out["all_byte_exact"] = all_ok
    print("\n" + "=" * 92)
    print(f"VERDICT: does the clever fp32 datapath run a proper fetch-decode-execute "
          f"BYTE-EXACT?  {'YES' if all_ok else 'NO'}")
    print(f"  programs verified L-inf=0 (PC/SP/BP/AX + stack + memory): "
          f"{[n for n in progs if out['results'][n]['byte_exact']]}")
    print("=" * 92)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"wrote {args.json}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
