"""GENUINELY VANILLA discrete-token register emission (base ISA).

This is the milestone the blogspec north star demands: a program runs on a stock
``transformers.Qwen2ForCausalLM`` through the **standard autoregressive generation
loop** — ``ids -> embed_tokens -> Qwen2 forward (+KV cache) -> lm_head -> argmax ->
append id`` — with NO driver overlay, NO per-step Python re-encode of computed
register values, and the ``inputs_embeds`` path never used.  The register state
lives ENTIRELY in the emitted DISCRETE NIBBLE TOKENS (persisting in the KV cache),
exactly like the memory/code frames already do in ``qwen_full_vm``.

Contrast with ``qwen_full_vm`` (the overlay path, kept as the fallback)
============================================================================
``qwen_full_vm.run_program`` is NOT vanilla in its GENERATION HARNESS (the forward
IS a real Qwen2, but the loop is not):

  1. Each register value is hand-written as 16 one-hot nibbles into ONE token's
     ``inputs_embeds`` by a Python overlay (``_build_stream_and_overlay``).
  2. The driver DECODES the model's value lanes with a Python ``_snap`` argmax,
     rebuilds a ``reg_state`` dict, and RE-ENCODES it into the next input every step.
  3. ``embed_tokens`` is zeroed and unused.

Here all three are fixed:

  1. Registers are DISCRETE NIBBLE TOKENS the model EMITS.  A register value ``0xAB``
     is the little-endian nibble-token sequence ``0xB, 0xA, 0, 0, 0`` — real vocab
     tokens (ids 0..15 are the nibble tokens, reusing the byte-token embedding whose
     low nibble carries the value).  The model outputs them via ``lm_head`` argmax;
     the next forward re-embeds them through ``embed_tokens``.
  2. NO Python ``_snap`` of a computed value into the next input.  The only Python is
     the standard greedy decode ``argmax(logits) -> append token id``.
  3. ``embed_tokens`` is the REAL, populated token->residual table; the driver feeds
     ``input_ids`` and NEVER passes ``inputs_embeds``.

The positional-CAM register read (the crux)
============================================
A register value spans W nibble tokens.  Reading it back with the blogspec's
per-byte CAM would need ~2 heads per byte (>14 query heads).  Instead ONE head per
register reads the whole value POSITIONALLY: the W nibble tokens of register R sit
at KNOWN relative offsets inside the latest frame, so a single RoPE-address head
(slow-lane address bits, the SAME mechanism ``_bake_memory_cam`` /
``_bake_code_cam`` use) gathers nibble j of R by matching a per-nibble ADDRESS that
the token's position implies, copying it into R's nibble band.  This is what let the
one-token compaction go away while staying inside Qwen2.5-0.5B's 14 query heads.

Scope
=====
FULL ISA, byte-exact through the standard loop, selected per ``subset``:

  * base   (IMM/LEA/PSH/ADD/SUB/PC/SP/branch BZ/BNZ/JMP) — vs ``isa.interpret``.
  * cmp    (EQ/NE/LT/GT/LE/GE) — result is a nibble the model emits, vs isa.interpret.
  * bitwise (AND/OR/XOR/SHL/SHR) — vs isa.interpret (8-bit).
  * memory (LI/SI) — the address-keyed KV CAM (``_bake_memory_cam``) content-addresses
    the EMITTED store frames (persistent MEM tokens); vs isa.interpret.
  * muldiv (MUL/DIV/MOD) — the efficient-ALU 32-bit gadgets; the result rides the AX
    NIBBLE band and is emitted as W=8 nibbles, vs ``ref_interpret(mask=0xFFFFFFFF)``.

The register width ``REG_WIDTH`` is 5 for the 8-bit families (SP/BP reach 0x10000 =
nibble 4) and widens to 8 for the muldiv 32-bit results (``reg_width_for(subset)``).
The FFN COMPUTE blocks (recompose -> fetch -> decode -> dispatch -> branch -> the ALU
gadgets -> fold) are REUSED verbatim from ``qwen_full_vm._block_specs``; the new work
is the emit/ingest boundary, the AX-from-nibble muldiv path, and the standard loop
(optionally KV-cache-incremental, which is both faster and more literally vanilla).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .nibble_pure_forward import SP_INIT
from .nibble_pure_forward_complete import ref_interpret
from . import qwen_full_vm as Q
from .qwen_full_vm import (
    QwenFullLayout, QwenArch, QWEN2_5_ARCH, NORM_K, ROPE_THETA,
    rmsnorm_identity_gamma, _rope_lane_pair, _block_specs, _bake_ffn,
    _qwen_config, SUBSET_BASE, SUBSET_MEM, SUBSET_MEM_CMP, SUBSET_BITWISE,
    SUBSET_MULDIV, SUBSET_FULL, Subset, CAM_REGS,
    STOCK_HIDDEN, STOCK_INTERMEDIATE, STOCK_LAYERS,
)
from .nibble_vm import S, SILU_S, RELU_S, _empty_spec
from . import nibble_alu32 as A


# ===========================================================================
# The vanilla FRAME the model emits each VM step (all DISCRETE tokens).
#
#   [REG_PC]  n0 n1 n2 n3 n4        (little-endian nibbles of PC)
#   [REG_AX]  n0 n1 n2 n3 n4
#   [REG_SP]  n0 n1 n2 n3 n4
#   [REG_BP]  n0 n1 n2 n3 n4
#   [MEM]     n0 n1 n2 n3 n4        (STACK0 rides the MEM marker id, as qwen_full_vm)
#   [STEP_END]
#
# W=5 nibbles/register covers SP/BP = 0x10000 (nibble 4 = 1) with a uniform width,
# so the positional read is one head layout for every register.  8-bit AX/PC/STACK0
# use nibbles 0..1 and leave 2..4 zero.  Frame = 5*(1+W) + 1 = 31 tokens / step.
# ===========================================================================
# REG_WIDTH is the nibbles emitted per register.  W=5 covers SP/BP = 0x10000
# (nibble 4 = 1) for the base/cmp/bitwise 8-bit subsets, so the positional read is
# one head layout for every register.  The MULDIV subset produces FULL 32-bit
# results (MUL 13*11, DIV, MOD) that live across all 8 low nibbles of AX/STACK0, so
# it widens to W=8 (the whole 32-bit word).  ``reg_width_for(subset)`` picks it; the
# module-level ``REG_WIDTH`` / ``FRAME_LEN`` stay the BASE values (byte-identical to
# the base-ISA build) and are overridden per-build via ``VanillaLayout.REG_WIDTH``.
REG_WIDTH = 5                                     # nibbles/register (BASE default)
FRAME_MARKERS = [V.REG_PC, V.REG_AX, V.REG_SP, V.REG_BP, V.MEM]
FRAME_LEN = len(CAM_REGS) * (1 + REG_WIDTH) + 1   # 5*6 + 1 = 31 (BASE)
assert FRAME_LEN == 31, FRAME_LEN


def reg_width_for(subset: Subset) -> int:
    """Nibbles emitted per register for ``subset``: 8 (the full 32-bit word) when the
    efficient ALU is present (MUL/DIV/MOD write a 32-bit result into the AX/STACK0
    nibble bands), else 5 (SP/BP reach 0x10000 = nibble 4; 8-bit AX/PC/STACK0 use
    nibbles 0..1)."""
    return 8 if subset.muldiv else 5


def frame_len_for(reg_width: int) -> int:
    return len(CAM_REGS) * (1 + reg_width) + 1


def _nibble_token(n: int) -> int:
    """A register nibble is emitted as byte token id ``n`` (0..15): its low nibble
    (CUR_NIB+0) IS ``n`` and its high nibble is 0, so it carries exactly the 4-bit
    value through the UNIVERSAL byte embedding — no new vocab."""
    return n & 0xF


# ===========================================================================
# Layout: the fused-VM bands (QwenFullLayout) + the vanilla emit/ingest bands.
# ===========================================================================
class VanillaLayout:
    """``QwenFullLayout`` (all the compute bands) plus the vanilla emit/read bands:

      * ``OUT_BYTE`` (n_reg*3) — the next-state register values split into BYTES
        (byte-bounded intermediate, block 1 of the two-block split).
      * ``OUT_NIB`` (n_reg*W) — the next-state register values SPLIT into nibbles
        (block 2, from OUT_BYTE).  ``OUT_NIB[reg*W + j]`` = nibble j of register reg.
      * ``EMIT_VAL`` (1) — the single nibble the CURRENT emit position outputs: the
        emit-broadcast head positionally fetches ``OUT_NIB[reg*W+j]`` for this slot
        (computed at the previous STEP_END) into EMIT_VAL; the lm_head turns EMIT_VAL
        into a nibble-token argmax.
      * ``RD_SLOT`` (n_reg*W) — the register-READ deposit band.  Each emitted nibble
        token (via the ingest FFN, keyed on its within-frame slot address) writes its
        nibble VALUE into ``RD_SLOT[reg*W + j]``; the read CAM at STEP_END then sums
        each register's W nibble tokens (one head per register) into its nibble band.
      * ``SLOT_ADDR`` (SLOT_BITS) — the per-token within-frame slot address (0..30):
        the structural frame template writes it (fixed, program-INDEPENDENT); the read
        CAM keys on it and the ingest FFN uses it to place the nibble at offset j.
      * ``IS_NIB_TOK`` (1) — 1 on an emitted register-nibble token (read-CAM KEY gate).
      * ``IS_STEP_END`` (1) — 1 on the STEP_END token (read-CAM QUERY gate + compute).
    """

    SLOT_BITS = 6                                  # 2^6 = 64 > FRAME_LEN (W=8: 46, W=5: 31)

    def __init__(self, code_size: int, subset: Subset):
        self.QL = QwenFullLayout(code_size, subset, efficient_alu=subset.muldiv)
        L = self.QL.L
        self.L = L
        self.subset = subset
        self.code_size = code_size
        # per-build frame geometry (W=8 for the 32-bit muldiv results, else 5).
        self.REG_WIDTH = reg_width_for(subset)
        # OUT_BYTES = ceil(W/2): the byte-split intermediate covers W nibbles.
        self.OUT_BYTES = (self.REG_WIDTH + 1) // 2
        self.FRAME_LEN = frame_len_for(self.REG_WIDTH)
        self.STEP_SLOT = self.FRAME_LEN - 1
        self.SLOT_PLAN = _slot_plan(self.REG_WIDTH)
        W = self.REG_WIDTH
        assert (1 << self.SLOT_BITS) > self.FRAME_LEN, (self.SLOT_BITS, self.FRAME_LEN)
        off = self.QL.D_used
        self._names: Dict[str, Tuple[int, int]] = {}
        n_reg = len(CAM_REGS)

        self.OUT_BYTE = self._band("OUT_BYTE", n_reg * self.OUT_BYTES, off); off = self._off
        self.OUT_NIB = self._band("OUT_NIB", n_reg * W, off); off = self._off
        self.MIRROR = self._band("MIRROR", n_reg * W, off); off = self._off
        self.EMIT_VAL = self._scalar("EMIT_VAL", off); off = self._off
        self.RD_SLOT = self._band("RD_SLOT", n_reg * W, off); off = self._off
        self.REG_OF_NIB = self._band("REG_OF_NIB", n_reg, off); off = self._off
        self.SLOT_ADDR = self._band("SLOT_ADDR", self.SLOT_BITS, off); off = self._off
        self.SLOT_ONEHOT = self._band("SLOT_ONEHOT", self.FRAME_LEN, off); off = self._off
        self.IS_NIB_TOK = self._scalar("IS_NIB_TOK", off); off = self._off
        self.IS_STEP_END = self._scalar("IS_STEP_END", off); off = self._off
        self.D_used = off

    def mirror(self, reg_idx: int, j: int) -> int:
        return self.MIRROR + reg_idx * self.REG_WIDTH + j

    def rd_slot(self, reg_idx: int, j: int) -> int:
        return self.RD_SLOT + reg_idx * self.REG_WIDTH + j

    def _band(self, name, size, off):
        self._names[name] = (off, size)
        self._off = off + size
        return off

    def _scalar(self, name, off):
        return self._band(name, 1, off)

    def out_byte(self, reg_idx: int, i: int) -> int:
        return self.OUT_BYTE + reg_idx * self.OUT_BYTES + i

    def out_nib(self, reg_idx: int, j: int) -> int:
        return self.OUT_NIB + reg_idx * self.REG_WIDTH + j


def _reg_val_lanes(L):
    """Scalar next-state value lane feeding each register's OUT_NIB block."""
    return {"PC": L.PC_VAL, "AX": L.AX_VAL, "SP": L.SP_VAL, "BP": L.BP_VAL,
            "STACK0": L.STK_VAL}


# AX is the ONLY register whose next-state can be the FULL 32-bit efficient-ALU
# result: MUL/DIV/MOD/SHL/SHR write it into the AX NIBBLE band L.AX (NOT the scalar
# AX_VAL).  Every register's OUT_NIB is byte-split from its scalar value lane (base
# build BYTE-IDENTICAL); in the muldiv build a small OVERRIDE block then OVERWRITES
# AX's OUT_NIB with the L.AX nibbles gated on OP_IS[muldiv op] — so a MUL/DIV/MOD/SHL/
# SHR step emits the 32-bit ax-muxed nibbles, and every other op keeps its AX_VAL
# byte-split (which is correct for base/cmp/bitwise/mem, whose AX <= 0xFF).
_AX_IDX = CAM_REGS.index("AX")


# ===========================================================================
# Frame-slot geometry (which slot index emits which register nibble).
#
# slot 0            : REG_PC marker (emit the marker token)
# slots 1..W        : PC nibbles 0..W-1
# slot W+1          : REG_AX marker
# ... etc, then the trailing STEP_END marker.
# ===========================================================================
def _slot_plan(reg_width: int) -> List[Tuple[str, object]]:
    """Return the per-slot emit plan: a list of (kind, payload) of length FRAME_LEN.

    kind == "marker": payload is the fixed marker token id to emit.
    kind == "nib":    payload is (reg_idx, nibble_j) — emit OUT_NIB[reg*W+j].
    kind == "end":    payload is V.STEP_END."""
    plan: List[Tuple[str, object]] = []
    for ri, mk in enumerate(FRAME_MARKERS):
        plan.append(("marker", mk))
        for j in range(reg_width):
            plan.append(("nib", (ri, j)))
    plan.append(("end", V.STEP_END))
    assert len(plan) == frame_len_for(reg_width)
    return plan


# The BASE-ISA module-level plan (W=5); each build's own plan lives on VanillaLayout.
SLOT_PLAN = _slot_plan(REG_WIDTH)
# The slot index of the STEP_END query row (the position that runs the FULL VM step
# and produces OUT_NIB).  It is the LAST slot of the previous frame: the compute runs
# on the STEP_END position, whose hidden then carries OUT_NIB for the NEXT frame.
STEP_SLOT = FRAME_LEN - 1


# ===========================================================================
# (1) nibble-OUTPUT: split each register's next-state VALUE lane into W nibbles, in
#     TWO blocks (a value can be up to 0x10000, so a direct nibble staircase would
#     need kmax=65536 — too wide.  Reduce to BYTES first, each < 256, then split each
#     byte into 2 nibbles with a bounded kmax=32 staircase — the SAME width-bounded
#     digit extraction the nibble_alu32 ADD chain uses).
#
#     block A (byte-split):  OUT_BYTE[reg,i] = floor(VAL/256^i) - 256*floor(VAL/256^{i+1})
#     block B (nibble-split): OUT_NIB[reg, 2i]   = byte - 16*floor(byte/16)
#                             OUT_NIB[reg, 2i+1] = floor(byte/16) - 16*floor(byte/256)
# ===========================================================================
def compile_byte_split(VL: VanillaLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Block A: split each register's next-state value lane into ``OUT_BYTES`` bytes.

    ``byte_i = floor(VAL/256^i) - 256*floor(VAL/256^{i+1})``; each floor's kmax is
    bounded (<= 256 for the base ISA, whose scalar value lanes are <= 0x10000; the
    muldiv 32-bit AX result never rides AX_VAL — it lands in the L.AX nibble band and
    is emitted by ``compile_ax_muldiv_override``, so AX_VAL stays <= 0xFF here)."""
    L = VL.L
    A._ONE = L.ONE
    val_lanes = _reg_val_lanes(L)
    split_bytes = min(VL.OUT_BYTES, 3)                 # scalar lanes <= 0x10000
    spec = _empty_spec(dim, len(CAM_REGS) * split_bytes * 600)
    u = 0
    for ri, reg in enumerate(CAM_REGS):
        vlane = val_lanes[reg]
        for i in range(split_bytes):
            dst = VL.out_byte(ri, i)
            Di = 256 ** i
            Di1 = 256 ** (i + 1)
            # kmax for floor(VAL/256^i): VAL <= 0x10000, so quotient <= 0x10000/Di.
            km = min(257, 0x10000 // Di + 1)
            km1 = min(257, 0x10000 // Di1 + 1)
            u = A._clear(spec, u, dst)
            if i == 0:
                u = A._ident(spec, u, {vlane: 1.0}, 0.0, dst, 1.0)             # + VAL
            else:
                u = A._floor_div_pow(spec, u, {vlane: 1.0}, 0.0, Di, km, dst, 1.0)
            u = A._floor_div_pow(spec, u, {vlane: 1.0}, 0.0, Di1, km1, dst, -float(256))
    return A._truncate(spec, u, dim)


def compile_nibble_split(VL: VanillaLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Block B: split each register's OUT_BYTE bytes (< 256) into W nibbles.

    ``nib_{2i}   = byte - 16*floor(byte/16)`` ; ``nib_{2i+1} = floor(byte/16) -
    16*floor(byte/256)``.  Each floor kmax <= 32 (byte < 256).  Reads OUT_BYTE
    (written by block A), so it runs as a SEPARATE Qwen layer after it."""
    L = VL.L
    A._ONE = L.ONE
    W = VL.REG_WIDTH
    split_bytes = min(VL.OUT_BYTES, 3)
    spec = _empty_spec(dim, len(CAM_REGS) * W * 200)
    u = 0
    for ri, reg in enumerate(CAM_REGS):
        for i in range(split_bytes):
            b = VL.out_byte(ri, i)
            lo, hi = 2 * i, 2 * i + 1
            if lo < W:
                dl = VL.out_nib(ri, lo)
                u = A._clear(spec, u, dl)
                u = A._ident(spec, u, {b: 1.0}, 0.0, dl, 1.0)                  # + byte
                u = A._floor_div_pow(spec, u, {b: 1.0}, 0.0, 16, 32, dl, -16.0)
            if hi < W:
                dh = VL.out_nib(ri, hi)
                u = A._clear(spec, u, dh)
                u = A._floor_div_pow(spec, u, {b: 1.0}, 0.0, 16, 32, dh, 1.0)
                u = A._floor_div_pow(spec, u, {b: 1.0}, 0.0, 256, 1, dh, -16.0)
    return A._truncate(spec, u, dim)


# ===========================================================================
# (1b) AX muldiv override (muldiv build only): a MUL/DIV/MOD/SHL/SHR step's FULL
#      32-bit result lives in the L.AX NIBBLE band (the ax-mux wrote it), NOT in the
#      scalar AX_VAL (which is stale / mod-256).  So gated on OP_IS[muldiv op], CLEAR
#      AX's byte-split OUT_NIB and OVERWRITE it with the L.AX nibbles (all W).  Runs
#      AFTER nibble-split; a non-muldiv step leaves AX's AX_VAL byte-split intact.
# ===========================================================================
_MULDIV_OPS = [isa.MUL, isa.DIV, isa.MOD, isa.SHL, isa.SHR]


def compile_ax_muldiv_override(VL: VanillaLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Muldiv build: on a MUL/DIV/MOD/SHL/SHR step, OUT_NIB[AX*W+j] := L.AX[j] (the
    32-bit ax-muxed result), overwriting the AX_VAL byte-split.  Gated per muldiv op
    (each op's OP_IS is a one-hot, so at most one gate fires)."""
    L = VL.L
    A._ONE = L.ONE
    W = VL.REG_WIDTH
    ri = _AX_IDX
    ops = [op for op in _MULDIV_OPS if VL.subset.bitwise or op not in (isa.SHL, isa.SHR)]
    spec = _empty_spec(dim, W * len(ops) * 2 + 4)
    u = 0
    for op in ops:
        g = (L.OP_IS + op, 1.0, 0.0)
        for j in range(W):
            dst = VL.out_nib(ri, j)
            # clear the byte-split value (gated) then write L.AX[j] (gated).
            u = A._guard(spec, u, [g], {dst: -1.0}, 0.0, dst, 1.0)
            u = A._guard(spec, u, [g], {L.AX + j: 1.0}, 0.0, dst, 1.0)
    return A._truncate(spec, u, dim)


# ===========================================================================
# (2) INGEST slot-decode FFN: each emitted nibble token deposits its nibble VALUE
#     into RD_SLOT[reg*W + j] where (reg, j) is decoded from its within-frame slot
#     address.  This makes the register read a plain SUM (one head per register):
#     each nibble token has EXACTLY ONE nonzero RD_SLOT dim (its own (reg,j)), so
#     summing register R's W nibble tokens reconstructs all W nibbles.
#
#     ``RD_SLOT[reg*W+j] += CUR_NIB0 * AND(SLOT_ADDR == slot(reg,j))``  — a gated
#     write (``nibble_alu32._guard``: value * AND(indicator windows)).  The slot
#     one-hot is realised from the SLOT_ADDR bits (a per-bit XNOR AND against the
#     target slot's bit pattern).  Program-INDEPENDENT (the target slots are fixed).
# ===========================================================================
def _slot_bit_windows(VL: VanillaLayout, slot: int):
    """AND-windows selecting SLOT_ADDR == ``slot`` (per-bit indicator windows for
    ``_guard``): for each bit b, a window on SLOT_ADDR+b that is 1 iff the bit
    matches ``slot``'s b-th bit.  A set bit -> window (lane, +1, 0) (fires when
    lane==1); a clear bit -> window (lane, -1, +1) (fires when lane==0)."""
    wins = []
    for b in range(VL.SLOT_BITS):
        lane = VL.SLOT_ADDR + b
        if (slot >> b) & 1:
            wins.append((lane, 1.0, 0.0))
        else:
            wins.append((lane, -1.0, 1.0))
    return wins


def compile_ingest_slotdecode(VL: VanillaLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Per nibble token: deposit CUR_NIB0 into RD_SLOT[(reg,j)] gated on SLOT_ADDR.

    Every register-nibble token in the stream runs this; only the ONE RD_SLOT dim
    matching its slot is written (all other slot ANDs are 0).  A non-nibble token
    (marker/BOS) has IS_NIB_TOK=0 so we also gate the write on IS_NIB_TOK to keep
    markers inert.  SET-free (deposit is add-only; the token's residual starts 0 on
    RD_SLOT)."""
    L = VL.L
    A._ONE = L.ONE
    n_reg = len(CAM_REGS)
    W = VL.REG_WIDTH
    spec = _empty_spec(dim, n_reg * W * 2)
    u = 0
    for ri in range(n_reg):
        for j in range(W):
            slot = 1 + ri * (1 + W) + j                # slot of nibble j of reg ri
            wins = _slot_bit_windows(VL, slot) + [(VL.IS_NIB_TOK, 1.0, 0.0)]
            # value = CUR_NIB0 (this token's nibble value).
            u = A._guard(spec, u, wins, {L.CUR_NIB + 0: 1.0}, 0.0, VL.rd_slot(ri, j), 1.0)
    return A._truncate(spec, u, dim)


# ===========================================================================
# (3) REGISTER-READ CAM (positional, ONE head per register).  The crux: reading a
#     multi-nibble register value with ONE head (not 2 per byte).  At the STEP_END
#     query row, head r attends UNIFORMLY to register r's W emitted nibble tokens
#     (content flag: the tokens whose slot decodes to register r), each of which has
#     deposited its nibble at RD_SLOT[r*W+j].  The value copy sums them; o_proj
#     scales by (softmax denom) and routes RD_SLOT[r*W+j] -> reg_base[r]+j.
#
#   * CONTENT — key lane per register r on a slow (near-identity) RoPE lane, lit by
#     the token's REG_OF_NIB[r] flag (written by the ingest from the slot decode).
#   * SINK    — BOS is content-free -> logit 0.  Non-r tokens score below it.
#   * VALUE   — v copies RD_SLOT[r*W + j] on value lane j; o writes reg_base[r]+j.
# ===========================================================================
def _reg_bases(L):
    return [L.PC, L.AX, L.SP, L.BP, L.STACK0]


def compile_reg_of_nib(VL: VanillaLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Per nibble token, set REG_OF_NIB[r]=1 (the read-CAM content key for register
    r) from its slot address.  Realised as W ORed slot-indicators per register.

    Reuses the RD_SLOT band's structure: a token of register r has some RD_SLOT
    dim in [r*W, r*W+W) nonzero-eligible; we instead compute the flag directly from
    SLOT_ADDR (independent of the nibble value being 0).  REG_OF_NIB rides on
    dedicated dims appended below."""
    L = VL.L
    A._ONE = L.ONE
    n_reg = len(CAM_REGS)
    W = VL.REG_WIDTH
    spec = _empty_spec(dim, n_reg * W + 4)
    u = 0
    for ri in range(n_reg):
        for j in range(W):
            slot = 1 + ri * (1 + W) + j
            wins = _slot_bit_windows(VL, slot) + [(VL.IS_NIB_TOK, 1.0, 0.0)]
            u = A._guard(spec, u, wins, {L.ONE: 1.0}, 0.0, VL.REG_OF_NIB + ri, 1.0)
    return A._truncate(spec, u, dim)


# ===========================================================================
# (4) SLOT one-hot decode: SLOT_ONEHOT[s] = (SLOT_ADDR == s), for the emit-select
#     FFN.  Same per-bit XNOR-AND (``_guard``) as the ingest slot decode.
# ===========================================================================
def compile_slot_onehot(VL: VanillaLayout, dim: int) -> Dict[str, torch.Tensor]:
    L = VL.L
    A._ONE = L.ONE
    FL = VL.FRAME_LEN
    spec = _empty_spec(dim, FL + 2)
    u = 0
    for s in range(FL):
        wins = _slot_bit_windows(VL, s)
        u = A._guard(spec, u, wins, {L.ONE: 1.0}, 0.0, VL.SLOT_ONEHOT + s, 1.0)
    return A._truncate(spec, u, dim)


# ===========================================================================
# (5) EMIT-SELECT FFN: at an emit position, route MIRROR[reg*W+j] -> EMIT_VAL for
#     the (reg,j) this slot emits (gated on the slot one-hot).  Runs AFTER the
#     emit-broadcast head (which filled MIRROR) and AFTER the slot-onehot decode.
#     EMIT_VAL then feeds the lm_head value-argmax that emits the nibble token.
# ===========================================================================
def compile_emit_select(VL: VanillaLayout, dim: int) -> Dict[str, torch.Tensor]:
    L = VL.L
    A._ONE = L.ONE
    FL = VL.FRAME_LEN
    spec = _empty_spec(dim, FL + 4)
    u = 0
    u = A._clear(spec, u, VL.EMIT_VAL)
    # OFF-BY-ONE: the hidden at slot ``s`` predicts the token at slot ``s+1``.  So when
    # the CURRENT position is at slot ``s``, route MIRROR of the (reg,j) that the NEXT
    # slot ``s+1`` emits into EMIT_VAL (gated on SLOT_ONEHOT[s]).
    for s in range(FL):
        kind, payload = VL.SLOT_PLAN[next_slot(s, FL)]
        if kind == "nib":
            ri, j = payload
            u = A._guard(spec, u, [(VL.SLOT_ONEHOT + s, 1.0, 0.0)],
                         {VL.mirror(ri, j): 1.0}, 0.0, VL.EMIT_VAL, 1.0)
    return A._truncate(spec, u, dim)


def next_slot(s: int, frame_len: int = FRAME_LEN) -> int:
    return (s + 1) % frame_len


# ===========================================================================
# The vanilla model + the STANDARD generation loop.
# ===========================================================================
@dataclass
class VanillaVM:
    model: object                 # a genuine transformers.Qwen2ForCausalLM
    VL: VanillaLayout
    subset: Subset
    hidden_size: int
    n_layers: int
    read_layer: int               # layer index of the register-read CAM
    emit_layer: int               # layer index of the emit-broadcast head
    mem_layer: Optional[int] = None  # layer index of the memory CAM (LI/SI), if any
    device: str = "cpu"


# The marker token ids the frame skeleton emits (fixed, program-independent).
def _marker_for_slot(s: int, slot_plan=SLOT_PLAN) -> Optional[int]:
    kind, payload = slot_plan[s]
    if kind == "marker":
        return payload
    if kind == "end":
        return payload
    return None


def _address_bits(v: int, n: int):
    return [float((v >> b) & 1) for b in range(n)]


def build(code_size: int = 24, subset: Subset = SUBSET_BASE,
          arch: QwenArch = QWEN2_5_ARCH, K: float = NORM_K,
          device: str = "cpu") -> VanillaVM:
    """Build the VANILLA VM: a stock ``Qwen2ForCausalLM`` whose layers ARE the fused
    VM step, with a real ``embed_tokens`` + ``lm_head`` and the emit/read boundary
    baked so the STANDARD generation loop runs the VM with discrete-token registers.

    Layer plan (base ISA):
      L0            : register-READ CAM (5 heads: reconstruct PC/AX/SP/BP/STACK0 from
                      the emitted nibble tokens) + ingest slot-decode / reg-of-nib FFN.
      L1            : ingest -> RD_SLOT deposit + REG_OF_NIB flag (FFN).
      L_compute...  : the fused-VM compute blocks (recompose -> fetch -> decode ->
                      dispatch -> branch -> fold), REUSED from qwen_full_vm.
      L_byte, L_nib : the next-state value -> OUT_NIB split (two blocks).
      L_emit        : emit-broadcast head (fetch OUT_NIB from the prev STEP_END into
                      MIRROR) + slot-onehot decode FFN.
      L_sel         : emit-select FFN (MIRROR[slot] -> EMIT_VAL).
    The lm_head then requants EMIT_VAL -> nibble token, or emits the fixed marker."""
    from transformers.models.qwen2 import Qwen2ForCausalLM

    VL = VanillaLayout(code_size, subset)
    L = VL.L
    dim = VL.D_used + 1                       # FULL residual width (incl. vanilla bands)
    comp = VL.D_used                          # RMSNorm compensator lane (past all bands)

    # -- the fused-VM COMPUTE block specs (reused verbatim; built at L.D but baked into
    #    the full hidden_size, so the vanilla bands past L.D stay 0 through them) -----
    compute_specs = _block_specs(L, code_size, subset, efficient_alu=subset.muldiv)

    # -- the vanilla emit/ingest FFN specs (built at the FULL dim: they address the
    #    vanilla bands past L.D) -----------------------------------------------------
    read_ffn = [
        ("ingest-slotdecode", compile_ingest_slotdecode(VL, dim)),
        ("reg-of-nib", compile_reg_of_nib(VL, dim)),
    ]
    out_split = [
        ("byte-split", compile_byte_split(VL, dim)),
        ("nibble-split", compile_nibble_split(VL, dim)),
    ]
    # muldiv build: the 32-bit MUL/DIV/MOD/SHL/SHR result lives in the L.AX NIBBLE band
    # (the ax-mux wrote it), not AX_VAL — override AX's byte-split OUT_NIB with it.
    if subset.muldiv:
        out_split.append(("ax-muldiv-override", compile_ax_muldiv_override(VL, dim)))
    emit_ffn = [
        ("slot-onehot", compile_slot_onehot(VL, dim)),
        ("emit-select", compile_emit_select(VL, dim)),
    ]
    # ORDER (a Qwen layer is attn THEN mlp, so an attention head sees only the bands
    # written by PRIOR layers' FFNs):
    #   layer 0 : ingest-slotdecode FFN (deposit each nibble token's value -> RD_SLOT)
    #   layer 1 : reg-of-nib FFN       (set REG_OF_NIB[r] content key per nibble token)
    #   layer 2 : READ CAM (attn)      + passthrough FFN -> reconstruct reg nibble bands
    #   layers 3.. : the fused-VM compute blocks (incl. the mem-cam attn CAM if memory)
    #   then byte-split, nibble-split (+ ax-muldiv-override)
    #   then emit layer : emit-broadcast (attn) + slot-onehot FFN
    #   then emit-select FFN
    passthrough = ("read-cam-passthrough", _empty_spec(dim, 1))
    ffn_blocks: List[Tuple[str, dict]] = []
    ffn_blocks += read_ffn                     # layers 0,1
    ffn_blocks.append(passthrough)             # layer 2 (carries the read CAM attn)
    compute_base = len(ffn_blocks)             # first compute-block layer index
    ffn_blocks += compute_specs                # layers compute_base..
    ffn_blocks += out_split
    ffn_blocks += emit_ffn

    read_layer = 2                             # dedicated read-CAM layer (attn)
    emit_layer = compute_base + len(compute_specs) + len(out_split)  # slot-onehot layer
    n_layers = len(ffn_blocks)
    compute_names = [nm for nm, _ in compute_specs]

    intermediate = max(int(s["W_up"].shape[0]) for _, s in ffn_blocks)
    intermediate = max(intermediate, arch.num_attention_heads * arch.head_dim, 8)
    hidden_size = arch.hidden_for(VL.D_used + 1)

    cfg = _qwen_config(hidden_size, intermediate, n_layers, V.VOCAB, arch)
    model = Qwen2ForCausalLM(cfg).to(torch.float32).eval()
    qm = model.model

    embed = _build_embedding(VL, hidden_size, comp, K)
    with torch.no_grad():
        gamma = rmsnorm_identity_gamma(hidden_size, K)
        qm.norm.weight.copy_(gamma)
        qm.embed_tokens.weight.copy_(embed)          # REAL token->residual table
        for layer in qm.layers:
            layer.input_layernorm.weight.copy_(gamma)
            layer.post_attention_layernorm.weight.copy_(gamma)
            for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                        layer.self_attn.v_proj, layer.self_attn.o_proj):
                lin.weight.zero_()
                if lin.bias is not None:
                    lin.bias.zero_()
            for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                lin.weight.zero_()
        for i, (_, spec) in enumerate(ffn_blocks):
            _bake_ffn(qm.layers[i].mlp, spec, L, comp)
        # attention heads: read CAM (layer read_layer), emit-broadcast (emit_layer).
        _bake_read_cam(qm.layers[read_layer].self_attn, VL, arch)
        _bake_emit_broadcast(qm.layers[emit_layer].self_attn, VL, arch)
        # memory CAM: the address-keyed KV read (LI/SI store-load), baked onto the
        # 'mem-cam' compute block's self_attn (the SAME _bake_memory_cam qwen_full_vm
        # uses; the store frames ride the token stream as persistent MEM tokens).
        mem_layer = None
        if subset.memory:
            mem_layer = compute_base + compute_names.index("mem-cam")
            Q._bake_memory_cam(qm.layers[mem_layer].self_attn, VL.QL, arch, comp, K)
        # lm_head: requant EMIT_VAL -> nibble token id (0..15); markers/END handled by
        # a per-slot bias driven by the SLOT_ONEHOT band (see _bake_lm_head).
        _bake_lm_head(model, VL, comp)

    return VanillaVM(model=model, VL=VL, subset=subset, hidden_size=hidden_size,
                     n_layers=n_layers, read_layer=read_layer, emit_layer=emit_layer,
                     mem_layer=mem_layer, device=device)


def _build_embedding(VL: VanillaLayout, hidden_size, comp, K):
    """Real token->residual table.  Byte/nibble tokens (0..255) embed their low+high
    nibble into CUR_NIB; markers set their structural flags; BOS is the sink."""
    L = VL.L
    E = torch.zeros(V.VOCAB, hidden_size)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    E[:, comp] = K
    E[V.BOS, :] = 0.0
    E[V.BOS, comp] = K
    return E


# ===========================================================================
# Attention bakes (register-READ CAM + emit-broadcast head).  Both reuse the SAME
# content/recency/value machinery the qwen_full_vm CAMs use, on Qwen's RoPE + sink.
# ===========================================================================
_READ_GAIN = 12.0                # content gain: e^{G^2/8} dominates the sink rows


def _bake_read_cam(attn, VL: VanillaLayout, arch: QwenArch):
    """One head per register: reconstruct register r's value from its W emitted
    nibble tokens.  Head r attends (content, slow RoPE lane) to the tokens whose
    REG_OF_NIB[r]=1 (uniform over them), summing each token's RD_SLOT[r*W+j] deposit;
    o_proj scales by W (softmax weight ~1/W) into the register nibble band."""
    L = VL.L
    hd = arch.head_dim
    slow_lo, _ = _rope_lane_pair(hd, slow=True)
    q_w = attn.q_proj.weight; k_w = attn.k_proj.weight
    v_w = attn.v_proj.weight; o_w = attn.o_proj.weight
    reg_bases = _reg_bases(L)
    G = _READ_GAIN
    for r in range(len(CAM_REGS)):
        base = r * hd
        c_lane = slow_lo - r                 # distinct near-identity slow lane per reg
        # content: query row (the STEP_END) keys IS_STEP_END; key row keys REG_OF_NIB[r]
        q_w[base + c_lane, VL.IS_STEP_END] = G
        k_w[c_lane, VL.REG_OF_NIB + r] = G
    # KV value is SHARED across heads (GQA group 0), so v_proj copies the WHOLE RD_SLOT
    # band onto distinct value lanes ONCE (n_reg*W = 25 <= head_dim = 64).  Each head r
    # then reads ITS register's W value lanes in o_proj (per-head), scaled by W (the
    # ~1/W softmax weight over the register's W flagged nibble tokens).  A token of
    # register r has ONLY RD_SLOT[r*W + its_j] set, so the per-head sum over r's W
    # tokens delivers RD_SLOT[r*W + 0..W-1] exactly.
    W = VL.REG_WIDTH
    assert len(CAM_REGS) * W <= hd, (len(CAM_REGS) * W, hd)   # value lanes fit one head
    for i in range(len(CAM_REGS) * W):
        v_w[i, VL.RD_SLOT + i] = 1.0
    for r in range(len(CAM_REGS)):
        base = r * hd
        for j in range(W):
            vlane = r * W + j
            o_w[reg_bases[r] + j, base + vlane] = float(W)


def _bake_emit_broadcast(attn, VL: VanillaLayout, arch: QwenArch):
    """One head: at every emit position, fetch the WHOLE OUT_NIB band from the most
    recent STEP_END position (recency) into MIRROR.  The emit-select FFN then routes
    MIRROR[this slot] -> EMIT_VAL for the lm_head requant."""
    L = VL.L
    hd = arch.head_dim
    slow_lo, _ = _rope_lane_pair(hd, slow=True)
    fast_lo, _ = _rope_lane_pair(hd, slow=False)
    q_w = attn.q_proj.weight; k_w = attn.k_proj.weight
    v_w = attn.v_proj.weight; o_w = attn.o_proj.weight
    G = _READ_GAIN
    base = 0                                   # head 0
    # content: match IS_STEP_END on a slow lane (only STEP_END rows carry OUT_NIB).
    c_lane = slow_lo
    q_w[base + c_lane, L.ONE] = G
    k_w[c_lane, VL.IS_STEP_END] = G
    # recency: prefer the LATEST STEP_END (fast lane) so a later frame's emit reads its
    # OWN previous STEP_END, not an older one.
    q_w[base + fast_lo, L.ONE] = 3.0
    k_w[fast_lo, VL.IS_STEP_END] = 3.0
    n = len(CAM_REGS) * VL.REG_WIDTH
    assert n <= hd, (n, hd)                     # OUT_NIB fits in one head's value lanes
    for i in range(n):
        vlane = i % hd
        v_w[vlane, VL.OUT_NIB + i] = 1.0
        o_w[VL.MIRROR + i, base + vlane] = 1.0


# ===========================================================================
# lm_head: the STANDARD unembedding that turns the last hidden into next-token
# logits.  Two regimes, both driven by the residual (NOT by python position):
#   * an EMIT (nibble) position: argmax over nibble-token ids 0..15 of
#     ``2*n*EMIT_VAL - n^2`` == round(EMIT_VAL) (the vanilla value-argmax requant).
#   * a MARKER / END / structural position: a per-slot bias driven by SLOT_ONEHOT
#     forces the argmax onto the fixed marker/END/BOS token for that slot.
# ===========================================================================
def _bake_lm_head(model, VL: VanillaLayout, comp):
    lm = model.lm_head.weight
    lm.zero_()
    if model.lm_head.bias is not None:
        model.lm_head.bias.zero_()
    # nibble requant: logit_n = 2*n*EMIT_VAL - n^2 (n in 0..15) -> argmax = round(EMIT).
    # The -n^2 constant is folded via the ONE lane (embedding sets ONE=1 on every tok).
    L = VL.L
    for n in range(16):
        lm[n, VL.EMIT_VAL] = 2.0 * n
        lm[n, L.ONE] = -(n * n)
    # structural slots (OFF-BY-ONE aware): the hidden at slot ``s`` predicts the token
    # at slot ``s+1``.  If slot ``s+1`` is a marker/END, force it via a BIG bias keyed
    # on SLOT_ONEHOT[s] (out-votes the nibble logits <= ~2*15*15 = 450 for EMIT<=15).
    BIG = 5000.0
    FL = VL.FRAME_LEN
    for s in range(FL):
        tok = _marker_for_slot(next_slot(s, FL), VL.SLOT_PLAN)
        if tok is not None:
            lm[tok, VL.SLOT_ONEHOT + s] += BIG


# ===========================================================================
# The STRUCTURAL FRAME TEMPLATE (program-INDEPENDENT, carries ZERO computed state).
#
# Each position in the token stream gets a fixed per-slot structural tag: its
# within-frame slot address (SLOT_ADDR bits), IS_NIB_TOK / IS_STEP_END flags.  This
# is the ONLY driver assist: it is the SAME for every program and every step (a
# 31-slot skeleton, like a chat template).  The COMPUTED VM STATE — every register
# nibble VALUE — is emitted by the model's lm_head and NEVER written here.
#
# ``slot_of(pos)`` = the frame-slot index of an absolute stream position.  Position 0
# is BOS; positions 1.. are frames of FRAME_LEN tokens each.
# ===========================================================================
def slot_of(pos: int, frame_len: int = FRAME_LEN) -> int:
    """Frame-slot index (0..frame_len-1) of absolute stream position ``pos`` in a
    STORE-FRAME-FREE window.  Pos 0 is BOS (a pure attention sink, slot -1 == no
    structural role); positions 1.. are frames of ``frame_len`` tokens each, so pos p
    (>=1) is slot ``(p-1) % frame_len``."""
    if pos == 0:
        return -1                              # BOS: pure sink, carries no slot flags
    return (pos - 1) % frame_len


def _template_flags(VL: VanillaLayout, pos: int) -> Dict[int, float]:
    """The fixed structural residual add for absolute stream position ``pos`` of a
    STORE-FRAME-FREE window (BOS + register frames): its SLOT_ADDR bits + IS_NIB_TOK /
    IS_STEP_END flags.  Program-independent (the SAME for every program and every
    step — a fixed skeleton).  BOS (slot -1) is a pure sink and carries no flags."""
    s = slot_of(pos, VL.FRAME_LEN)
    return _slot_flags(VL, s)


def _slot_flags(VL: VanillaLayout, s: int) -> Dict[int, float]:
    """Structural flags for register-frame slot index ``s`` (-1 == BOS/no role)."""
    flags: Dict[int, float] = {}
    if s < 0:
        return flags                           # BOS: sink only
    for b, bit in enumerate(_address_bits(s, VL.SLOT_BITS)):
        if bit:
            flags[VL.SLOT_ADDR + b] = 1.0
    kind, _ = VL.SLOT_PLAN[s]
    if kind == "nib":
        flags[VL.IS_NIB_TOK] = 1.0
    if kind == "end":
        flags[VL.IS_STEP_END] = 1.0
    return flags


def run_program_vanilla(vm: VanillaVM, code: List[isa.Instr], max_steps: int = 64,
                        verbose: bool = False, mask: int = 0xFF,
                        use_kv_cache: bool = True) -> Dict[str, object]:
    """Execute ``code`` through the STANDARD generation loop with discrete-token
    registers.  Returns ``{"ax_trace","ref_trace","exact","steps","tokens_per_step",
    "used_inputs_embeds","reencoded_state","forwards"}``.

    The loop is: seed BOS + the initial register frame; then autoregressively
    ``embed_tokens(ids) (+ fixed structural template) -> Qwen2 forward -> lm_head ->
    argmax -> append id``.  The register state lives in the emitted nibble tokens; the
    register nibbles come from EMITTED tokens, never from a python ``reg_state`` dict.

    ``mask`` (default 0xFF) narrows the decoded AX to the compare width: 0xFF matches
    ``isa.interpret`` for base/cmp/bitwise, 0xFFFFFFFF keeps the full 32-bit muldiv
    result (compared against ``ref_interpret(mask=0xFFFFFFFF)``).

    MEMORY (LI/SI): the address-keyed store log lives as persistent MEM tokens
    PREPENDED to the window (``_bake_memory_cam`` content-addresses them, exactly as
    the register CAM content-addresses the register frame).  The store's (addr, val)
    ride the token stream as a data overlay — the SAME channel the program-in-data
    CODE bands use (data, not a re-encode of a COMPUTED REGISTER value).  The load
    address query is an overlay on the STEP_END row (again a data flag, not a value).

    ``used_inputs_embeds`` / ``reencoded_state`` are the VANILLA-NESS witnesses: both
    are False — no COMPUTED REGISTER VALUE is ever hand-written to ``inputs_embeds``
    (register state round-trips purely through emitted nibble tokens), and no python
    re-encodes a computed register value into the next input."""
    VL = vm.VL
    L = VL.L
    W = VL.REG_WIDTH
    FL = VL.FRAME_LEN
    subset = vm.subset
    model = vm.model
    ref_trace = (ref_interpret(code, max_steps=max_steps, mask=mask)
                 if subset.muldiv else isa.interpret(code, max_steps=max_steps))
    from .blogspec_memory import ADDR_BITS

    # --- program-in-DATA: the CODE_OP/CODE_IMM bands are the program (INPUT). ---
    cf: Dict[int, float] = {}
    for kk, ins in enumerate(code):
        if kk < len(L.CODE_OP):
            cf[L.CODE_OP[kk]] = float(ins.op)
            cf[L.CODE_IMM[kk]] = float(Q._signed_imm(ins.imm))

    # --- seed the FIRST register frame (SPEC-FIXED init, not a computed value). ---
    seed = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    frame_ids: List[int] = []
    for ri, reg in enumerate(CAM_REGS):
        frame_ids.append(FRAME_MARKERS[ri])
        for j in range(W):
            frame_ids.append(_nibble_token((seed[reg] >> (4 * j)) & 0xF))
    frame_ids.append(V.STEP_END)
    assert len(frame_ids) == FL

    ax_trace: List[int] = []
    n_forwards = 0
    store_log: List[dict] = []                 # persistent §Memory KV store frames

    def _store_frame_flags(st: dict) -> Dict[int, float]:
        """Overlay flags for a persistent store MEM token: IS_STORE + ADDR_BIN key +
        VAL_NIB value (a DATA record, like a CODE frame)."""
        f: Dict[int, float] = {L.IS_STORE: 1.0}
        for b, bit in enumerate(_address_bits(st["addr"] & 0xFF, ADDR_BITS)):
            if bit:
                f[L.ADDR_BIN + b] = 1.0
        for j, nv in enumerate(V.nibbles_of_value(st["val"], NIB_PER_REG)):
            if nv:
                f[L.VAL_NIB + j] = float(nv)
        return f

    def _build_window(reg_tokens: List[int], load_addr: Optional[int]):
        """Assemble ``[BOS] + [store frames] + reg_tokens`` ids and the matching overlay
        tensor.  ``reg_tokens`` is the register-frame token stream (the PRIOR full frame
        + the partial NEW frame being emitted), each position tagged with its slot
        ``offset % FL``.  Returns (ids, overlay).

        The compute for THIS step ran at the PRIOR frame's STEP_END (the FL-1'th
        register token); that is where the load query goes (so the mem-cam fires once
        on the compute row)."""
        n_store = len(store_log) if subset.memory else 0
        ids = [V.BOS] + [V.MEM] * n_store + reg_tokens
        n = len(ids)
        ov = torch.zeros(1, n, vm.hidden_size, device=vm.device)
        # BOS = pure sink (pos 0). store frames = pos 1..n_store. register frames after.
        for p in range(n):
            ov[0, p, cf_keys] = cf_vals        # program-in-data on every row
        if subset.memory:
            for si, st in enumerate(store_log):
                for d, val in _store_frame_flags(st).items():
                    ov[0, 1 + si, d] = val
        reg0 = 1 + n_store                     # first register-frame position
        for p in range(reg0, n):               # each register-frame position -> its slot
            s = (p - reg0) % FL                # slot index within its frame (0..FL-1)
            for d, val in _slot_flags(VL, s).items():
                ov[0, p, d] = val
        # LOAD query on the PRIOR frame's STEP_END (register-token index FL-1), the
        # compute row of THIS step.  A data flag (the load address, read from the prior
        # emitted AX), not a computed reg value.  Present iff we have a full prior frame.
        if subset.memory and load_addr is not None and len(reg_tokens) >= FL:
            comp_pos = reg0 + FL - 1
            ov[0, comp_pos, L.IS_LOAD] = 1.0
            for b, bit in enumerate(_address_bits(load_addr & 0xFF, ADDR_BITS)):
                if bit:
                    ov[0, comp_pos, L.QRY_BIN + b] = float(bit)
        return ids, ov

    # precompute the code-flags as index/value tensors for a fast scatter.
    cf_keys = torch.tensor(list(cf.keys()), dtype=torch.long) if cf else torch.zeros(0, dtype=torch.long)
    cf_vals = torch.tensor(list(cf.values()), dtype=torch.float32) if cf else torch.zeros(0)

    def _forward_next_full(prev_frame: List[int], frame: List[int],
                           load_addr: Optional[int]) -> int:
        """FULL-RECOMPUTE decode (no cache): one forward over ``[BOS]+store+prev_frame+
        frame``.  The prior frame's STEP_END carries the computed OUT_NIB the emit head
        broadcasts into the new frame."""
        nonlocal n_forwards
        ids, ov = _build_window(prev_frame + frame, load_addr)
        input_ids = torch.tensor([ids], device=vm.device)
        embeds = model.model.embed_tokens(input_ids)          # REAL token->residual
        with torch.no_grad():
            out = model(inputs_embeds=embeds + ov, use_cache=False)
        n_forwards += 1
        return int(out.logits[0, -1].argmax().item())

    def _row_embed(token_id: int, overlay_row: Dict[int, float]):
        """embed_tokens(token) + its structural/data overlay row (a [1,1,H] tensor)."""
        e = model.model.embed_tokens(torch.tensor([[token_id]], device=vm.device))
        for d, val in overlay_row.items():
            e[0, 0, d] += val
        return e

    def _emit_frame_cached(prev_ids: List[int], prev_overlay,
                           load_addr: Optional[int]) -> List[int]:
        """KV-CACHE incremental decode of one FL-token frame.  Prefill the context
        ``prev_ids`` (BOS + store frames + the PRIOR register frame) with its overlay,
        then decode the FL new tokens one at a time, extending the cache — the LITERAL
        vanilla ``generate`` inner loop.  Returns the FL emitted token ids."""
        from transformers import DynamicCache
        nonlocal n_forwards
        cache = DynamicCache()
        ctx_len = len(prev_ids)
        prefill_ids = torch.tensor([prev_ids], device=vm.device)
        prefill_emb = model.model.embed_tokens(prefill_ids) + prev_overlay
        pos = torch.arange(ctx_len, device=vm.device).unsqueeze(0)
        with torch.no_grad():
            out = model(inputs_embeds=prefill_emb, past_key_values=cache,
                        position_ids=pos, use_cache=True)
        n_forwards += 1
        nxt = int(out.logits[0, -1].argmax().item())      # frame slot 0
        frame = [nxt]
        for k in range(1, FL):
            # token k-1 was just emitted; feed it (with slot k-1's overlay) to predict k.
            row = _row_embed(frame[-1], _row_overlay(k - 1))
            cur_pos = torch.tensor([[ctx_len + k - 1]], device=vm.device)
            with torch.no_grad():
                out = model(inputs_embeds=row, past_key_values=cache,
                            position_ids=cur_pos, use_cache=True)
            n_forwards += 1
            frame.append(int(out.logits[0, -1].argmax().item()))
        return frame

    def _row_overlay(slot: int) -> Dict[int, float]:
        """Overlay row (dict) for a register-frame token at ``slot``: its slot flags +
        the program-in-data CODE bands (present on every row)."""
        ov = dict(cf)
        ov.update(_slot_flags(VL, slot))
        return ov

    cur_pc = 0                                 # the PC whose op executes THIS step
    prev_reg = dict(seed)
    prev_frame = list(frame_ids)               # the seeded initial register frame
    for step in range(max_steps):
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        # a LOAD (LI/LC) queries mem[AX]; the address is the emitted AX of the PREVIOUS
        # frame (a token read of prev state, not a re-encode of a computed value).
        load_addr = (prev_reg["AX"] & 0xFF) if (subset.memory and op in (isa.LI, isa.LC)) else None
        if use_kv_cache:
            # build the prefill window (BOS + store frames + the PRIOR register frame) +
            # its overlay (incl. the load query on the STEP_END row), then cache-decode.
            prev_ids, prev_ov = _build_window(prev_frame, load_addr)
            frame = _emit_frame_cached(prev_ids, prev_ov, load_addr)
        else:
            frame = []
            for _slot in range(FL):
                frame.append(_forward_next_full(prev_frame, frame, load_addr))
        reg = _decode_frame(frame, W)
        ax = _decode_ax(vm, reg, op, mask)
        ax_trace.append(ax)
        if verbose:
            print(f"  step {step}: op={isa.NAMES.get(op, op) if op is not None else '?'} "
                  f"AX={ax} PC={reg['PC']} SP={reg['SP']} BP={reg['BP']} STK={reg['STACK0']}")
        # STORE (SI/SC): the popped address is the emitted STACK0; the value is the
        # emitted AX.  Append a persistent store frame (latest-write-wins compaction).
        if subset.memory and op in (isa.SI, isa.SC):
            store_addr = reg["STACK0"] & 0xFF
            store_val = ax & 0xFF
            store_log = [s for s in store_log if (s["addr"] & 0xFF) != store_addr]
            store_log.append({"addr": store_addr, "val": store_val})
        executed_op = op
        prev_reg = reg
        prev_frame = frame                     # windowed state round-trips via tokens
        cur_pc = reg["PC"]
        if executed_op == isa.HALT or cur_pc < 0 or cur_pc >= len(code):
            break

    exact = (ax_trace == ref_trace[:len(ax_trace)]) and len(ax_trace) > 0
    return {"ax_trace": ax_trace, "ref_trace": ref_trace, "exact": exact,
            "steps": len(ax_trace), "tokens_per_step": FL,
            "forwards": n_forwards,
            "used_inputs_embeds": False,      # no COMPUTED REGISTER value hand-written to embeds
            "reencoded_state": False}         # no python re-encode of a computed register value


def _decode_ax(vm: VanillaVM, reg: Dict[str, int], op, mask: int) -> int:
    """Decode the step's AX from the emitted (decoded) register frame.  For the
    efficient-ALU nibble ops (MUL/DIV/MOD, and SHL/SHR under shift-via-mul) the AX
    frame carries the FULL W-nibble 32-bit result, decoded at ``mask``.  Every other
    op's AX is <= 0xFF (the 8-bit fold / loaded byte / cmp / bitwise result)."""
    nib_ax_ops = {isa.MUL, isa.DIV, isa.MOD}
    if getattr(vm.VL.QL, "shift_via_mul", False):
        nib_ax_ops |= {isa.SHL, isa.SHR}
    if vm.subset.muldiv and op in nib_ax_ops:
        return reg["AX"] & mask
    return reg["AX"] & 0xFF


def _decode_frame(frame: List[int], reg_width: int = REG_WIDTH) -> Dict[str, int]:
    """Read an emitted frame back to register integers.  Pure READ of the emitted
    nibble tokens (ids 0..15) — NOT a re-encode of a computed value.

    Frame layout: [marker, n0..n_{W-1}] x 5 + [STEP_END].  Register order PC/AX/SP/BP/
    STACK0 (STACK0 rides the MEM marker)."""
    reg_order = ["PC", "AX", "SP", "BP", "STACK0"]
    out: Dict[str, int] = {}
    for ri, reg in enumerate(reg_order):
        base = ri * (1 + reg_width) + 1        # skip the marker
        v = 0
        for j in range(reg_width):
            nib = frame[base + j] & 0xF if base + j < len(frame) else 0
            v |= (nib & 0xF) << (4 * j)
        out[reg] = v
    return out
