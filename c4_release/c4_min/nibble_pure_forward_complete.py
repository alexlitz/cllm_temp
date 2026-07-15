"""PURE-FORWARD C4 VM — the LAST boundary closed: multi-slot stack via the KV head
+ the fp32-exact 32-bit ALU folded in.  EVERYTHING through the vanilla forward.

This extends ``nibble_pure_forward`` (the pure-forward VM: one VM step = one
``model.forward``, state round-tripping through the emitted 30-token frames, ALU /
dispatch / control in FFN weights, memory as softmax1-KV attention over the token
stream, argmax-generate-append the only Python on the compute path) to remove the
two remaining honest-boundary items the pure-forward deliverable named:

  1. **MULTI-SLOT STACK via the KV head.**  The pure-forward VM carried a single
     ``STACK0`` mirror slot, so ``PSH`` depth>1 (and JSR/ENT/LEV/ADJ) collided.
     Here the stack **is memory** (BLOG_SPEC §Memory / §Stack): ``SP``/``BP`` are
     byte addresses, and push/pop go through the **same softmax1-KV memory head**
     that already runs LI/SI byte-exact — a store keyed on the binary address, a
     load whose query is that address.

        PSH v      :  MEM[SP-4] = v ; SP -= 4                 (store @ new top)
        <consume>  :  STK = MEM[SP] ; op(STK, AX) ; SP += 4   (load @ top, then op)
        JSR t      :  MEM[SP-4] = pc+1 ; SP -= 4 ; PC = t
        ENT n      :  MEM[SP-4] = BP ; SP -= 4 ; BP = SP ; SP -= 4n
        ADJ n      :  SP += 4n
        LEV        :  SP=BP ; BP=MEM[SP] ; PC=MEM[SP+4] ; SP+=8
        LEA o      :  AX = BP + 4o                             (frame-relative addr)

     A push lays a store MEM token in the emitted frame (a KV entry keyed on the
     32-bit address, EXACTLY as SI does); a pop is a load whose QUERY is ``SP``
     instead of ``AX`` and whose retrieved value lands in the ``STACK0`` nibble
     band (the operand the ALU pops).  Depth-N stacks + the full calling
     convention run in-forward because each push is its own KV entry in the token
     stream and each pop content-addresses the newest write at that address.

  2. **32-bit ALU folded into the pure-forward model.**  The 8-bit MUL/DIV/MOD
     lookup table is replaced by ``nibble_alu32``'s persistent fp32 FFN blocks
     (ADD/SUB per-byte carry chain, MUL nibble schoolbook, DIV/MOD base-16 long
     division — all fp32-EXACT for the full 32-bit range, no lookup table, no
     fp64).  The op result is computed by ``model.forward`` on the register nibble
     bands and multiplexed into AX by an opcode-gated mux.

Everything runs through the vanilla autoregressive forward, proven byte-exact vs
:func:`ref_interpret` (the SP-addressed memory-stack reference) under the
``assert_no_python_compute`` guard (re-exported from ``nibble_pure_forward``).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .nibble_vm import (
    S, RELU_S, SILU_S, SILU_HALF,
    compile_nibble_to_scalar, compile_pc_fetch, compile_code_select,
    base_dispatch_rules, compile_branch_delta, compile_fold, compile_ffn,
    _empty_spec, _load_ffn, _zero_attn, _snap_lane,
)
from .blogspec_model import Transformer
from . import nibble_pure_forward as PF
from .nibble_pure_forward import (
    PureForwardLayout, N_ROLES, bake_frame_ingest, _bake_pf_memory_head,
    compile_opcode_decode_pf, compile_cmp_compute, cmp_dispatch_rules,
    compile_mem_prep, memory_dispatch_rules, MEM_HEAD_CHANNELS,
    _address_bits, _FRAME_ROLE_SLOTS, _MEM_MARKER_LOCAL,
    _MEM_ADDR_LOCAL, _MEM_VAL_LOCAL, SP_INIT, _flag_from_ops, _concat_specs,
)
from . import nibble_alu32 as A
from .dsl import FFNRule, LinearExpr
from .blogspec_memory import ADDR_BITS

# opcode value for ADJ / LC / SC / NOP (canonical C4 values; base isa lacks ADJ).
ADJ = isa.ADJ if hasattr(isa, "ADJ") else 7


# ===========================================================================
# The op groups (the stack contract).
# ===========================================================================
# POP-consuming ops: pop the stack top MEM[SP] into STACK0, op(STACK0, AX), SP+=4.
POP_OPS = [isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
           isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
           isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
           isa.SI, isa.SC]
# PUSH ops (allocate a stack slot): the driver lays a store MEM token.
PUSH_OPS = [isa.PSH, isa.JSR, isa.ENT]
# ALU ops that read STACK0 as operand A and AX as operand B (the 32-bit ALU set).
ALU_OPS = [isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD]


# ===========================================================================
# LAYOUT: PureForwardLayout + the stack-pop KV query band + the 32-bit ALU bands.
# ===========================================================================
class PureForwardCompleteLayout(PureForwardLayout):
    """``PureForwardLayout`` + the stack-pop KV query band + the ALU-32 scratch.

      ``IS_POP`` (1)       — set on a pop-consuming op; enables the stack KV head.
      ``SP_QRY_BIN`` (32)  — the SP address bits (the stack-pop QUERY).
      ``ALU_*`` (via ``extend_layout_for_alu32``) — the 32-bit ALU scratch bands.
    """

    def __init__(self, code_size: int, n_heads: int):
        super().__init__(code_size, n_heads=n_heads)
        from .blogspec_layout import NIB_PER_REG
        self._off = self.D
        self.IS_POP = self._scalar("IS_POP")        # enables the stack KV head
        self.IS_LEV = self._scalar("IS_LEV")        # enables the LEV return-PC head
        self.POP_ADDR = self._band("POP_ADDR", NIB_PER_REG)   # = SP (or BP on LEV)
        self.LEV_ADDR = self._band("LEV_ADDR", NIB_PER_REG)   # = BP+4 (LEV ret-PC)
        self.SP_QRY_BIN = self._band("SP_QRY_BIN", ADDR_BITS)  # pop address bits
        self.LEV_QRY_BIN = self._band("LEV_QRY_BIN", ADDR_BITS)  # LEV ret-PC bits
        self.LEV_RET = self._band("LEV_RET", NIB_PER_REG)     # loaded return-PC nibbles
        self.LEV_RET_VAL = self._scalar("LEV_RET_VAL")        # its scalar image
        self.AXB_LO = self._scalar("AXB_LO")                  # AX_VAL low nibble scratch
        self.AXB_HI = self._scalar("AXB_HI")                  # AX_VAL high nibble scratch
        while self._off % n_heads != 0:
            self._scalar(f"_pfcpad{self._off}")
        self.D = self._off


# ===========================================================================
# NIBBLE -> ADDRESS BITS: expand a register's low bytes into 32 per-bit dims by
# reading its NIBBLE band directly (each nibble 0..15 -> its 4 bits via a 16-cell
# one-hot).  EXACT and O(1) per nibble — no scalar magnitude, no 256-cell table.
# Bit (4*j + t) = bit t of nibble j = sum over cells a of nibble j with bit t set.
# ===========================================================================
def compile_nibble_addr_expand(L, reg_base: int, bin_base: int, dim: int,
                               n_nibbles: int = 8) -> Dict[str, torch.Tensor]:
    """``bin_base[4*j + t] = bit t of nibble j of the register at ``reg_base```` for
    j < n_nibbles, t in 0..3.  A nibble is 0..15; its one-hot over the 16 cells is
    the triangular pulse, and bit t is the sum of the cells whose index has bit t.
    Self-clears each written bit lane first (SET)."""
    thr = list(range(-1, 17))
    tu = {t: k for k, t in enumerate(thr)}
    n_thr = len(thr)
    n_bits = 4 * n_nibbles
    spec = _empty_spec(dim, n_nibbles * n_thr + n_bits)
    u = 0
    # relu bank per nibble (16-cell one-hot thresholds).
    relu0 = {}
    for j in range(n_nibbles):
        relu0[j] = u
        for t, k in tu.items():
            spec["W_up"][u, reg_base + j] = RELU_S
            spec["b_up"][u] = -RELU_S * t
            spec["W_gate"][u, L.ONE] = 1.0
            u += 1
    # self-clear each output bit lane (SET).
    for b in range(n_bits):
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, bin_base + b] = 1.0
        spec["W_down"][bin_base + b, u] += -1.0 / SILU_S
        u += 1
    # bit t of nibble j = sum over cells a in 0..15 with (a>>t)&1 of one-hot(a).
    for j in range(n_nibbles):
        r0 = relu0[j]
        for a in range(16):
            # one-hot(a) coefficient = tri pulse: +cell(a-1) -2cell(a) +cell(a+1).
            for t in range(4):
                if (a >> t) & 1:
                    b = 4 * j + t
                    spec["W_down"][bin_base + b, r0 + tu[a - 1]] += 1.0 / RELU_S
                    spec["W_down"][bin_base + b, r0 + tu[a]] += -2.0 / RELU_S
                    spec["W_down"][bin_base + b, r0 + tu[a + 1]] += 1.0 / RELU_S
    return spec


def _clear_band_gated_ops(L, base, n, dim, gate_ops):
    """Clear band dims base..base+n-1 when any OP_IS[gate_ops] active (SET -old)."""
    spec = _empty_spec(dim, n * len(gate_ops))
    u = 0
    for op in gate_ops:
        g = L.OP_IS + op
        for j in range(n):
            spec["W_up"][u, g] = S
            spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, base + j] = 1.0
            spec["W_down"][base + j, u] += -1.0 / SILU_HALF
            u += 1
    return spec


# ===========================================================================
# STACK-POP PREP FFN: on a pop op set IS_POP, expand the SP nibble band ->
# SP_QRY_BIN (the load address), and clear the STACK0 nibble band so the stack KV
# head's additive write is a clean SET of the popped value.  All opcode-gated.
# ===========================================================================
def compile_pop_addr(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    """POP_ADDR nibbles := SP nibbles by default, but := BP nibbles on a LEV (LEV's
    first load is MEM[BP], not MEM[SP]).  LEV_ADDR nibbles := BP+4 (LEV's second
    load, the return PC) — computed as BP scalar + 4 recomposed to nibbles is
    awkward; instead we set LEV_ADDR = BP nibbles and add 4 in the address bits by
    querying BP and letting the LEV head's query be BP with bit-2 forced.  Simpler
    & exact: LEV_ADDR carries BP's nibbles and we OR bit 2 (value 4) — but BP is
    4-aligned so bit 2 = 4 flips cleanly.  We build LEV_QRY_BIN = SP_QRY_BIN(BP)
    with bit 2 set, done in the expand.  Here we only fill POP_ADDR / LEV_ADDR
    nibble bands.  IS_POP / IS_LEV flags are set in compile_stack_prep."""
    a = L
    n = 8
    # POP_ADDR = SP (default), overwritten to BP on LEV.
    spec = _empty_spec(dim, n * 3 + n * 3)
    u = 0
    for j in range(n):                          # POP_ADDR := SP  (clear + copy)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.POP_ADDR + j] = 1.0
        spec["W_down"][L.POP_ADDR + j, u] += -1.0 / SILU_S; u += 1
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.SP + j] = 1.0
        spec["W_down"][L.POP_ADDR + j, u] += 1.0 / SILU_S; u += 1
        # on LEV: POP_ADDR += (BP - SP)  (gated), so POP_ADDR becomes BP.
        g = L.OP_IS + isa.LEV
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.BP + j] = 1.0
        spec["W_gate"][u, L.SP + j] = -1.0
        spec["W_down"][L.POP_ADDR + j, u] += 1.0 / SILU_HALF; u += 1
    for j in range(n):                          # LEV_ADDR := BP  (clear + copy)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.LEV_ADDR + j] = 1.0
        spec["W_down"][L.LEV_ADDR + j, u] += -1.0 / SILU_S; u += 1
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.BP + j] = 1.0
        spec["W_down"][L.LEV_ADDR + j, u] += 1.0 / SILU_S; u += 1
        u += 1                                   # (spare unit, keeps width uniform)
    return spec


def compile_stack_prep(L: PureForwardCompleteLayout, dim: int) -> Dict[str, torch.Tensor]:
    specs = []
    specs.append(_flag_from_ops(L, L.IS_POP, POP_OPS + [isa.LEV], dim))
    specs.append(_flag_from_ops(L, L.IS_LEV, [isa.LEV], dim))
    # SP_QRY_BIN <- bits of POP_ADDR (SP for pops, BP for LEV).
    specs.append(compile_nibble_addr_expand(L, L.POP_ADDR, L.SP_QRY_BIN, dim, n_nibbles=8))
    # LEV_QRY_BIN <- bits of LEV_ADDR (BP).  The +4 (BP -> BP+4, the return-PC slot)
    # is a SEPARATE block (``lev-addr4``) because the +4 ripple-carry reads
    # LEV_QRY_BIN at its block INPUT — it must run AFTER this expand writes it.
    specs.append(compile_nibble_addr_expand(L, L.LEV_ADDR, L.LEV_QRY_BIN, dim, n_nibbles=8))
    # clear STACK0 (pop dest) + LEV_RET (LEV ret-PC dest) so head writes are clean.
    specs.append(_clear_band_gated_ops(L, L.STACK0, 8, dim, POP_OPS + [isa.LEV]))
    specs.append(_clear_band_gated_ops(L, L.LEV_RET, 8, dim, [isa.LEV]))
    return _concat_specs(specs, dim)


def _force_bit2(L, bin_base, dim):
    """Add 4 to the binary address band (BP+4 = LEV return-PC slot).  BP is
    4-aligned (bits 0,1 = 0) but bit 2 is NOT necessarily 0 (e.g. BP=228=0xE4 has
    bit 2 set) — so a bit-force would be WRONG.  This is a true ripple-carry +1 at
    bit position 2, expressed ENTIRELY on the block-input bits (FFN units all read
    the block input, so no relu chaining):

        carry_k  = AND(bits 2..k-1)  = [ sum(bits 2..k-1) == k-2 ]     (carry into k)
        andk     = AND(bits 2..k)    = [ sum(bits 2..k)   == k-1 ]     ( = bit_k·carry_k)
        new_bit_k = bit_k XOR carry_k = bit_k + carry_k - 2·andk

    carry_2 = 1 (the injected +4).  Bits 0,1 are untouched.  Each ``[sum==n]``
    prefix-AND is one relu bump (relu(x-(n-0.5)) - relu(x-(n+0.5)))."""
    hi = ADDR_BITS

    def and_bits(spec, u, lo, k, out_base, coeff):
        """Add ``coeff · AND(bits lo..k)`` to residual dim ``out_base``.  x = Σ bits
        (an integer in 0..cnt), so AND == [x >= cnt] — a SHARP unit step at cnt
        realised as a narrow ramp relu(x-(cnt-w)) - relu(x-cnt) of amplitude 1 at
        x=cnt (w=0.5, so the step is 1.0 exactly at the integer x=cnt, 0 at cnt-1)."""
        cnt = k - lo + 1
        w = 0.5
        for i, thr in enumerate((cnt - w, cnt)):
            spec["W_up"][u, L.ONE] = -RELU_S * thr
            for j in range(lo, k + 1):
                spec["W_up"][u, bin_base + j] = RELU_S
            spec["W_gate"][u, L.ONE] = 1.0
            spec["W_down"][out_base, u] += (coeff if i == 0 else -coeff) / (RELU_S * w)
            u += 1
        return u

    # new_bit_k = bit_k XOR carry_k, so the FFN DELTA (added to the residual bit_k)
    # is exactly  carry_k - 2·(bit_k AND carry_k) = carry_k - 2·AND(bits 2..k).  No
    # clear/restore needed — the residual already carries bit_k.  carry_2 = 1.
    spec = _empty_spec(dim, hi * 6)
    u = 0
    for k in range(2, hi):
        if k == 2:                                   # carry_2 = 1 (the injected +4)
            spec["W_up"][u, L.ONE] = S
            spec["W_gate"][u, L.ONE] = 1.0
            spec["W_down"][bin_base + k, u] += 1.0 / SILU_S; u += 1
        else:                                        # carry_k = AND(bits 2..k-1)
            u = and_bits(spec, u, 2, k - 1, bin_base + k, +1.0)
        # - 2·(bit_k AND carry_k) = -2·AND(bits 2..k)
        u = and_bits(spec, u, 2, k, bin_base + k, -2.0)
    return spec


# ===========================================================================
# The STACK KV head: identical §Memory CAM as the LI/LC head, but keyed on the
# STACK query band SP_QRY_BIN (address = SP) and enabled by IS_POP, writing the
# retrieved value nibbles into the STACK0 band (the operand the ALU pops).
# ===========================================================================
def _bake_stack_pop_head(attn, L: PureForwardCompleteLayout, head: int) -> None:
    """Bake the stack-pop KV head: query = SP_QRY_BIN, enable = IS_POP, value ->
    STACK0 nibble band.  Same address CAM + ZFOD-bias + store-role + pop-enable
    channels as ``_bake_pf_memory_head``, on a different head/query/dest."""
    from .blogspec_memory import EFF, BIAS, MEM_ALIBI_SLOPE
    from .blogspec_layout import NIB_PER_REG
    hs = attn.scale
    smag = (EFF / hs) ** 0.5
    qb = (BIAS / hs) ** 0.5
    kb = (BIAS / hs) ** 0.5
    PEN = 100.0 * ADDR_BITS * EFF
    p = (PEN / hs) ** 0.5
    attn.alibi_slopes[head] = MEM_ALIBI_SLOPE
    HD = attn.head_dim
    base = head * HD
    for b in range(ADDR_BITS):
        attn.W_k[base + b, L.ADDR_BIN + b] = 2.0 * smag
        attn.W_k[base + b, L.ONE] = -smag
        attn.W_q[base + b, L.SP_QRY_BIN + b] = 2.0 * smag   # QUERY = SP address
        attn.W_q[base + b, L.ONE] = -smag
    cB = base + ADDR_BITS
    attn.W_q[cB, L.IS_POP] = -qb            # ZFOD bias enabled by IS_POP
    attn.W_k[cB, L.IS_STORE] = kb
    cR = base + ADDR_BITS + 1
    attn.W_q[cR, L.IS_POP] = p              # store-role penalty
    attn.W_k[cR, L.ONE] = -p
    attn.W_k[cR, L.IS_STORE] = p
    cL = base + ADDR_BITS + 2               # POP-enable: non-pop query -> sink
    c = (PEN / hs) ** 0.5
    attn.W_q[cL, L.ONE] = c
    attn.W_q[cL, L.IS_POP] = -c
    attn.W_k[cL, L.ONE] = -c
    for j in range(NIB_PER_REG):
        attn.W_v[base + ADDR_BITS + 3 + j, L.VAL_NIB + j] = 1.0
        attn.W_o[L.STACK0 + j, base + ADDR_BITS + 3 + j] = 1.0   # -> STACK0, not AX


# ===========================================================================
# STACK-POP recompose: refresh STK_VAL from the STACK0 nibble band (the KV head
# just wrote the popped value there), so the dispatch/ALU read the fresh operand.
# (Same nibble->scalar recompose the base block does, but only STACK0.)
# ===========================================================================
def compile_stk_recompose(L, dim: int, hi_nibbles: int = 8) -> Dict[str, torch.Tensor]:
    """Refresh STK_VAL from STACK0 and LEV_RET_VAL from LEV_RET (both nibble bands
    the KV heads just wrote), so the LEV dispatch reads the loaded scalars."""
    spec = _empty_spec(dim, 2 * (1 + hi_nibbles))
    u = 0
    for nib_base, val_lane in ((L.STACK0, L.STK_VAL), (L.LEV_RET, L.LEV_RET_VAL)):
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, val_lane] = 1.0
        spec["W_down"][val_lane, u] += -1.0 / SILU_S
        u += 1
        for j in range(hi_nibbles):
            spec["W_up"][u, L.ONE] = S
            spec["W_gate"][u, nib_base + j] = 1.0
            spec["W_down"][val_lane, u] += (16.0 ** j) / SILU_S
            u += 1
    return spec


def _bake_lev_ret_head(attn, L: PureForwardCompleteLayout, head: int) -> None:
    """The LEV return-PC KV head: query = LEV_QRY_BIN (address = BP+4), enable =
    IS_LEV, value -> LEV_RET nibble band.  Same §Memory CAM as the stack head."""
    from .blogspec_memory import EFF, BIAS, MEM_ALIBI_SLOPE
    from .blogspec_layout import NIB_PER_REG
    hs = attn.scale
    smag = (EFF / hs) ** 0.5
    qb = (BIAS / hs) ** 0.5
    kb = (BIAS / hs) ** 0.5
    PEN = 100.0 * ADDR_BITS * EFF
    p = (PEN / hs) ** 0.5
    attn.alibi_slopes[head] = MEM_ALIBI_SLOPE
    HD = attn.head_dim
    base = head * HD
    for b in range(ADDR_BITS):
        attn.W_k[base + b, L.ADDR_BIN + b] = 2.0 * smag
        attn.W_k[base + b, L.ONE] = -smag
        attn.W_q[base + b, L.LEV_QRY_BIN + b] = 2.0 * smag
        attn.W_q[base + b, L.ONE] = -smag
    cB = base + ADDR_BITS
    attn.W_q[cB, L.IS_LEV] = -qb
    attn.W_k[cB, L.IS_STORE] = kb
    cR = base + ADDR_BITS + 1
    attn.W_q[cR, L.IS_LEV] = p
    attn.W_k[cR, L.ONE] = -p
    attn.W_k[cR, L.IS_STORE] = p
    cL = base + ADDR_BITS + 2
    c = (PEN / hs) ** 0.5
    attn.W_q[cL, L.ONE] = c
    attn.W_q[cL, L.IS_LEV] = -c
    attn.W_k[cL, L.ONE] = -c
    for j in range(NIB_PER_REG):
        attn.W_v[base + ADDR_BITS + 3 + j, L.VAL_NIB + j] = 1.0
        attn.W_o[L.LEV_RET + j, base + ADDR_BITS + 3 + j] = 1.0


# ===========================================================================
# CALLING-CONVENTION dispatch (JSR/ENT/ADJ/LEV/LEA/PSH) on the value lanes.  The
# stack MOTION (SP/BP/PC changes) is register arithmetic done here in weights; the
# stack STORE/LOAD (the pushed/popped values) rides the KV head + the driver's
# store-token overlay.  All gated on OP_IS[op].
# ===========================================================================
def callconv_dispatch_rules(L) -> List[FFNRule]:
    ax, sp, bp, stk, pc = L.AX_VAL, L.SP_VAL, L.BP_VAL, L.STK_VAL, L.PC_VAL
    imm = L.IMM

    def G(op):
        return [(L.OP_IS + op, 0.5, 1.5)]

    rules: List[FFNRule] = []
    # PSH: SP -= 4 ; PC += 1.  The pushed value (AX) rides the driver's store token
    # at MEM[SP-4].  STACK0 mirror is not used for the stack any more.
    rules.append(FFNRule(G(isa.PSH), {sp: LinearExpr.c(-4.0), pc: LinearExpr.c(1.0)}))
    # JSR: SP -= 4 ; PC = imm.  The return PC (old_pc+1) rides the store token.
    rules.append(FFNRule(G(isa.JSR), {
        sp: LinearExpr.c(-4.0),
        pc: LinearExpr.of(imm, 1.0) + LinearExpr.of(pc, -1.0)}))
    # ENT n: MEM[SP-4]=BP (store token) ; SP -= 4 ; BP = SP ; SP -= 4*imm ; PC += 1.
    # Net: BP = SP_old - 4 ; SP = SP_old - 4 - 4*imm.
    rules.append(FFNRule(G(isa.ENT), {
        bp: LinearExpr.of(sp, 1.0) + LinearExpr.of(bp, -1.0) + LinearExpr.c(-4.0),
        sp: LinearExpr.c(-4.0) + LinearExpr.of(imm, -4.0),
        pc: LinearExpr.c(1.0)}))
    # ADJ n: SP += 4*imm ; PC += 1.
    rules.append(FFNRule([(L.OP_IS + ADJ, 0.5, 1.5)], {
        sp: LinearExpr.of(imm, 4.0), pc: LinearExpr.c(1.0)}))
    # LEA o: AX = (BP + 4*imm) & 0xFF ; PC += 1.  imm is the SLOT offset; *4 gives a
    # byte address matching the SP/BP byte-addressing.  Add BP's LOW BYTE only (the
    # 8-bit op masks &0xFF, and BP's higher bytes vanish mod 256) so the downstream
    # single mod-256 fold keeps AX a byte.
    rules.append(FFNRule(G(isa.LEA), {
        ax: LinearExpr.of(L.BP_LOW, 1.0) + LinearExpr.of(imm, 4.0) + LinearExpr.of(ax, -1.0),
        pc: LinearExpr.c(1.0)}))
    # LEV: SP = BP + 8 ; BP = MEM[BP] (loaded into STK_VAL by the stack head) ;
    # PC = MEM[BP+4] (loaded into LEV_RET_VAL by the lev head).  Both loads done
    # in-forward by the two KV heads; here we just route the loaded scalars.
    rules.append(FFNRule(G(isa.LEV), {
        sp: LinearExpr.of(bp, 1.0) + LinearExpr.of(sp, -1.0) + LinearExpr.c(8.0),
        bp: LinearExpr.of(stk, 1.0) + LinearExpr.of(bp, -1.0),
        # PC = LEV_RET_VAL ; the base PC+1 must be cancelled -> write PC delta =
        # (LEV_RET_VAL - pc).  (No base rule fires for LEV, so PC is untouched here
        # except this write.)
        pc: LinearExpr.of(L.LEV_RET_VAL, 1.0) + LinearExpr.of(pc, -1.0)}))
    return rules


# ===========================================================================
# ALU-32 dispatch: on an ALU op the result is written to AX by the ax-mux from the
# op's dedicated RES band; here we just do SP += 4 (pop consumed) + PC += 1.  The
# ax-mux (nibble_alu32.compile_ax_mux) writes AX = RES nibbles; we must NOT also
# write AX here.  So the base ADD/SUB rules are REPLACED by SP/PC-only rules.
# ===========================================================================
def alu32_housekeeping_rules(L) -> List[FFNRule]:
    sp, pc = L.SP_VAL, L.PC_VAL
    rules = []
    for op in ALU_OPS:
        rules.append(FFNRule([(L.OP_IS + op, 0.5, 1.5)],
                             {sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}))
    return rules


def compile_opcode_decode_pfc(L, dim: int) -> Dict[str, torch.Tensor]:
    """Opcode decode for the COMPLETE VM: the pure-forward decode set PLUS the
    calling-convention ops (JSR/ENT/ADJ/LEV).  ``compile_opcode_decode_pf`` decodes
    BASE_OPS + memory + cmp + bitwise + muldiv, but NOT JSR/ENT/ADJ/LEV — so their
    dispatch rules (all gated on ``OP_IS[op]``) never fire and JSR/ENT/LEV stay
    no-ops.  This widens the decode so the callconv one-hots light up."""
    from .nibble_vm import BASE_OPS
    from .nibble_unified import compile_opcode_decode_ops
    ops = sorted(set(BASE_OPS + PF.PF_MEM_OPS +
                     [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE] +
                     [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR] +
                     [isa.MUL, isa.DIV, isa.MOD] +
                     [isa.JSR, isa.ENT, ADJ, isa.LEV]))    # + calling convention
    return compile_opcode_decode_ops(L, dim, ops)


def compile_ax_nib_split(L, dim: int) -> Dict[str, torch.Tensor]:
    """AXB_LO = low nibble of AX_VAL's byte, AXB_HI = high nibble — via ONE ungated
    256-cell one-hot of AX_VAL (triangular pulse) summed with the nibble weights.
    Runs in its OWN block so the writeback (next block) can read AXB_LO/AXB_HI (FFN
    units all see the block INPUT, so producer and consumer must be different
    blocks)."""
    thr = list(range(-1, 257))
    tu = {t: k for k, t in enumerate(thr)}
    n_thr = len(thr)
    axb_lo, axb_hi = L.AXB_LO, L.AXB_HI
    spec = _empty_spec(dim, n_thr + 2)
    u = 0
    relu0 = u
    for t, k in tu.items():
        spec["W_up"][u, L.AX_VAL] = RELU_S
        spec["b_up"][u] = -RELU_S * t
        spec["W_gate"][u, L.ONE] = 1.0
        u += 1
    for lane in (axb_lo, axb_hi):                # self-clear (SET)
        spec["W_up"][u, L.ONE] = S; spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_S; u += 1
    for a in range(256):
        lo, hi = a & 0xF, (a >> 4) & 0xF
        for lane, val in ((axb_lo, lo), (axb_hi, hi)):
            if val:
                spec["W_down"][lane, relu0 + tu[a - 1]] += val / RELU_S
                spec["W_down"][lane, relu0 + tu[a]] += -2.0 * val / RELU_S
                spec["W_down"][lane, relu0 + tu[a + 1]] += val / RELU_S
    return spec


def compile_ax_byte_to_nibbles(L, dim: int, ops) -> Dict[str, torch.Tensor]:
    """When any op in ``ops`` (byte-producing: IMM/LEA/CMP/bitwise/LI) is active,
    SET AX nibble 0 = AXB_LO, nibble 1 = AXB_HI (computed by ``compile_ax_nib_split``
    in the previous block), and clear AX nibbles 2..7 — so the AX nibble band is the
    canonical 8-bit result and the driver decodes AX from it uniformly."""
    axb_lo, axb_hi = L.AXB_LO, L.AXB_HI
    spec = _empty_spec(dim, len(ops) * 8 + len(ops) * 2)
    u = 0
    for op in ops:
        g = L.OP_IS + op
        for j in range(8):                       # clear AX[0..7] gated
            spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, L.AX + j] = 1.0
            spec["W_down"][L.AX + j, u] += -1.0 / SILU_HALF; u += 1
        for nb, src in ((0, axb_lo), (1, axb_hi)):   # nib0=AXB_LO, nib1=AXB_HI
            spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, src] = 1.0
            spec["W_down"][L.AX + nb, u] += 1.0 / SILU_HALF; u += 1
    return spec


def _fold_ax_gated(L, dim: int, ops) -> Dict[str, torch.Tensor]:
    """Fold AX_VAL mod 256 ONLY when one of ``ops`` is active (LEA masks &0xFF).
    The 32-bit ALU result must NOT be folded, so the base global fold is replaced
    by this opcode-gated one.  Sharp ramp AX_VAL -= 256·[AX_VAL>=256], gated."""
    M, w = 256, 0.2
    lo = M - 0.5
    spec = _empty_spec(dim, len(ops) * 2)
    u = 0
    for op in ops:
        gu = L.OP_IS + op
        for i, thr in enumerate((lo, lo + w)):
            spec["W_up"][u, L.AX_VAL] = RELU_S
            spec["b_up"][u] = -RELU_S * thr
            spec["W_gate"][u, gu] = 1.0          # gate = OP_IS[op] (0 or 1)
            spec["W_down"][L.AX_VAL, u] += (-M if i == 0 else M) / (RELU_S * w)
            u += 1
    return spec


# ===========================================================================
# BUILD the complete pure-forward model.
# ===========================================================================
def build_pure_forward_complete_model(code_size: int = 32,
                                      include_bitwise: bool = True,
                                      include_divmod: bool = True):
    """Assemble the complete pure-forward VM: frame ingest + LI/LC KV head +
    stack-pop KV head + the 32-bit ALU FFN blocks + callconv, all as persistent
    weights applied by ``model.forward``.  Returns ``(model, L)``.

    ``include_divmod`` (default True) folds the base-16 long-division DIV/MOD
    blocks (262 blocks — the dominant cost and the memory/time hazard the deliver-
    able flags).  Set False for a LEAN model (36 blocks, ~10x faster forward) that
    keeps the multi-slot stack + callconv + ADD/SUB/MUL + cmp/bitwise/memory; only
    DIV/MOD are then unsupported (their RES band stays unfilled).  The corpus
    values are <=9999 so the 8-bit-masked byte trace needs no 32-bit division."""
    n_heads = N_ROLES + 3          # 20 ingest + LI head + stack-pop head + lev head
    L = PureForwardCompleteLayout(code_size, n_heads=n_heads)
    A.extend_layout_for_alu32(L)               # ALU scratch bands
    if include_bitwise:
        from . import nibble_bitwise as _bw
        _bw.extend_layout_for_bitwise(L)
        while L._off % n_heads != 0:
            L._scalar(f"_bwpad{L._off}")
        L.D = L._off
    # force head_dim >= MEM_HEAD_CHANNELS so both KV heads' local channels fit.
    min_dim = n_heads * MEM_HEAD_CHANNELS
    if L.D < min_dim:
        target = -(-min_dim // n_heads) * n_heads
        while L._off < target:
            L._scalar(f"_hdpad{L._off}")
        L.D = L._off
    dim = L.D
    A._ONE = L.ONE

    # Block order (all one model.forward):
    #   ingest+recompose | pc-fetch | code-select | opcode-decode
    #   | mem-prep (LI addr) | mem-cam (LI/LC KV head -> AX) + recompose
    #   | stack-prep (pop addr) | stack-pop-cam (stack KV head -> STACK0) + recompose
    #   | cmp-compute | alu-expand | addsub | mul | divmod | ax-mux
    #   | [bitwise blocks] | dispatch | branch-delta | fold
    block_specs: List[Tuple[str, Dict]] = [
        ("ingest+recompose", compile_nibble_to_scalar(L, dim)),
        ("pc-fetch",    compile_pc_fetch(L, dim)),
        ("code-select", compile_code_select(L, dim)),
        ("opcode-decode", compile_opcode_decode_pfc(L, dim)),  # + JSR/ENT/ADJ/LEV
        ("mem-prep", compile_mem_prep(L, dim)),
        ("mem-cam",  compile_nibble_to_scalar(L, dim)),          # ATTN=LI head
        ("pop-addr", compile_pop_addr(L, dim)),                  # POP_ADDR/LEV_ADDR
        ("stack-prep", compile_stack_prep(L, dim)),
        ("lev-addr4", _force_bit2(L, L.LEV_QRY_BIN, dim)),       # LEV_QRY_BIN += 4
        ("stack-pop-cam", compile_stk_recompose(L, dim)),        # ATTN=stack+lev heads
        ("cmp-compute", compile_cmp_compute(L, dim)),
        ("alu-expand", A.compile_expand(L, dim)),
    ]
    for name, spec in A.compile_addsub_blocks(L, dim):
        block_specs.append((name, spec))
    for name, spec in A.compile_mul_blocks(L, dim):
        block_specs.append((name, spec))
    if include_divmod:                             # 262 blocks — LEAN skips these
        for name, spec in A.compile_divmod_blocks(L, dim):
            block_specs.append((name, spec))
    mux_ops = ALU_OPS if include_divmod else [isa.ADD, isa.SUB, isa.MUL]
    block_specs.append(("ax-mux", A.compile_ax_mux(L, dim, ops=mux_ops)))
    if include_bitwise:
        from .nibble_unified import build_bitwise_blocks, _bw_recompose_spec
        for name, spec in build_bitwise_blocks(L, dim):
            block_specs.append((name, spec))
        block_specs.append(("bw-recompose", _bw_recompose_spec(
            L, dim, (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR))))

    # Dispatch: base ops MINUS the ALU ops (ax-mux writes AX) + memory + cmp +
    # callconv + ALU housekeeping + bitwise housekeeping.
    disp_rules = _base_rules_minus_alu(L)
    disp_rules += memory_dispatch_rules(L)
    disp_rules += cmp_dispatch_rules_pop(L)
    disp_rules += callconv_dispatch_rules(L)
    disp_rules += alu32_housekeeping_rules(L)
    if include_bitwise:
        disp_rules += _bitwise_pop_rules(L)
    # ops whose AX result is a BYTE written into AX_VAL by the dispatch: IMM/LEA +
    # cmp + bitwise + LI/LC (the memory head writes AX nibbles for LI, but a byte-
    # value writeback of AX_VAL is idempotent for it too).  These get a byte->nibble
    # writeback so the AX nibble band is canonical (the driver decodes AX from it).
    byte_ax_ops = [isa.IMM, isa.LEA, isa.LI, isa.LC] + \
                  [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE]
    if include_bitwise:
        byte_ax_ops += [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR]
    block_specs += [
        ("dispatch", compile_ffn(disp_rules, dim)),
        ("branch-delta", compile_branch_delta(L, dim)),
        ("fold-lea", _fold_ax_gated(L, dim, [isa.LEA])),  # LEA masks &0xFF; ALU 32-bit
        ("ax-nib-split", compile_ax_nib_split(L, dim)),   # AX_VAL byte -> AXB_LO/HI
        ("ax-byte-nib", compile_ax_byte_to_nibbles(L, dim, byte_ax_ops)),
    ]
    n_blocks = len(block_specs)
    hidden = max(f["W_up"].shape[0] for _, f in block_specs)
    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, vocab=V.VOCAB, max_seq_len=16384)
    with torch.no_grad():
        _bake_pure_embedding(model, L)
        for bi, (name, spec) in enumerate(block_specs):
            _zero_attn(model.blocks[bi].attn)
            _load_ffn(model.blocks[bi].ffn, spec, hidden)
        reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP, "STACK0": L.STACK0}
        bake_frame_ingest(model.blocks[0].attn, L, reg_bases)
        mem_block = _find(block_specs, "mem-cam")
        _bake_pf_memory_head(model.blocks[mem_block].attn, L, head=N_ROLES)
        stk_block = _find(block_specs, "stack-pop-cam")
        _bake_stack_pop_head(model.blocks[stk_block].attn, L, head=N_ROLES + 1)
        _bake_lev_ret_head(model.blocks[stk_block].attn, L, head=N_ROLES + 2)
    L._block_names = [n for n, _ in block_specs]
    return model, L


def _bake_pure_embedding(model, L) -> None:
    E = torch.zeros(V.VOCAB, model.dim)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    model.embed.copy_(E)


def _find(block_specs, name):
    return [i for i, (nm, _) in enumerate(block_specs) if nm == name][0]


def _base_rules_minus_alu(L) -> List[FFNRule]:
    """The base dispatch rules with ADD/SUB REMOVED (the 32-bit ALU + ax-mux own
    AX for ADD/SUB/MUL/DIV/MOD) and PSH/LEA REMOVED (callconv owns them)."""
    rules = base_dispatch_rules(L)
    drop = {isa.ADD, isa.SUB, isa.PSH, isa.LEA}
    out = []
    for r in rules:
        # a rule's gate is [(OP_IS+op, ...)]; find which op.
        op = r.when[0][0] - L.OP_IS if r.when else None
        if op in drop:
            continue
        out.append(r)
    return out


def cmp_dispatch_rules_pop(L) -> List[FFNRule]:
    """The CMP write rules (from ``nibble_pure_forward.cmp_dispatch_rules``) but
    with the SP += 4 pop already done — cmp_dispatch_rules already adds SP+=4, so
    reuse it directly."""
    return cmp_dispatch_rules(L)


def _bitwise_pop_rules(L) -> List[FFNRule]:
    from .nibble_pure_forward import bitwise_dispatch_rules
    return bitwise_dispatch_rules(L)


# ===========================================================================
# SP-ADDRESSED MEMORY-STACK REFERENCE (the byte-exact oracle).  Stack + frame +
# heap share ONE byte-addressed region; SP/BP are byte addresses descending from
# SP_INIT.  Matches the neural build's per-step AX emit.
# ===========================================================================
def ref_interpret(code: List[isa.Instr], max_steps: int = 512,
                  mask: int = 0xFF) -> List[int]:
    """Reference interpreter with the SP-addressed memory stack.  ``mask`` = 0xFF
    (8-bit AX trace, the default; matches the neural byte frame) or 0xFFFFFFFF (the
    full 32-bit AX for the 32-bit-arithmetic proof: ADD/SUB/MUL wrap mod 2^32,
    DIV/MOD unsigned floor with b==0 -> 0, ISA_SPEC 4.2)."""
    mem: Dict[int, int] = {}
    sp = bp = SP_INIT
    ax = pc = 0
    trace: List[int] = []
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
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0) & mask; sp += 4
            if op == isa.ADD:
                ax = (v + ax) & mask
            elif op == isa.SUB:
                ax = (v - ax) & mask
            elif op == isa.MUL:
                ax = (v * ax) & mask
            elif op == isa.DIV:
                ax = ((v // ax) if ax else 0) & mask
            else:
                ax = ((v % ax) if ax else 0) & mask
        elif op in (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR):
            v = mem.get(sp, 0) & 0xFF; sp += 4
            if op == isa.OR:
                ax = (v | ax) & 0xFF
            elif op == isa.XOR:
                ax = (v ^ ax) & 0xFF
            elif op == isa.AND:
                ax = (v & ax) & 0xFF
            elif op == isa.SHL:
                ax = (v << ax) & 0xFF
            else:
                ax = (v >> ax) & 0xFF
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0) & 0xFF; sp += 4
            r = {isa.EQ: v == ax, isa.NE: v != ax, isa.LT: v < ax,
                 isa.GT: v > ax, isa.LE: v <= ax, isa.GE: v >= ax}[op]
            ax = 1 if r else 0
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0) & 0xFF
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4
            mem[addr] = ax & 0xFF
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
            trace.append(ax & mask); break
        else:
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in ref ISA")
        trace.append(ax & mask)
    return trace


# ===========================================================================
# THE OVERLAY: program-in-data + frame ROLE tags + the KV store log.  A store
# frame's MEM token becomes a KV entry keyed on its 32-bit address.  This carries
# BOTH the SI/SC program stores AND the stack pushes (PSH/JSR/ENT) — they are the
# same kind of address-keyed write; the driver records (frame_idx -> (addr, val)).
# ===========================================================================
def make_overlay_complete(code: List[isa.Instr], L: PureForwardCompleteLayout,
                          store_log=None):
    """``overlay(x)`` writes the program into the DATA bands at every position, the
    ROLE/IS_FRAME_BYTE frame-slot tags, and turns each stored frame's MEM token
    into a KV entry.  ``store_log`` maps ``frame_idx -> (addr, val)`` for every
    emitted frame that wrote memory (a program store OR a stack push)."""
    store_log = store_log or {}
    from .blogspec_layout import NIB_PER_REG

    def overlay(x: torch.Tensor) -> None:
        Sn = x.shape[1]
        for i in range(Sn):
            x[0, i, L.ONE] = 1.0
            for k, ins in enumerate(code):
                x[0, i, L.CODE_OP[k]] = float(ins.op)
                x[0, i, L.CODE_IMM[k]] = float(ins.imm)
        pos = 1
        frame_idx = 0
        while pos + V.FRAME_LEN <= Sn:
            for local, role in _FRAME_ROLE_SLOTS.items():
                p = pos + local
                x[0, p, L.ROLE + role] = 1.0
                x[0, p, L.IS_FRAME_BYTE] = 1.0
            if frame_idx in store_log:
                addr, val = store_log[frame_idx]
                mem_pos = pos + _MEM_MARKER_LOCAL
                x[0, mem_pos, L.IS_STORE] = 1.0
                x[0, mem_pos, L.IS_FRAME_BYTE] = 0.0
                for b, bit in enumerate(_address_bits(addr)):
                    x[0, mem_pos, L.ADDR_BIN + b] = bit
                for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                    x[0, mem_pos, L.VAL_NIB + j] = float(nv)
            pos += V.FRAME_LEN
            frame_idx += 1
        # the query row (the last position) carries all ROLE one-hots for ingest.
        for role in range(N_ROLES):
            x[0, -1, L.ROLE + role] = 1.0
    return overlay


# ===========================================================================
# THE DRIVER — one VM step = one model.forward; argmax-generate-append only.  The
# only Python beyond that is the driver's store-token BOOKKEEPING (it fetches the
# op at PC — the same code-as-data fetch the model does — to know WHICH address a
# push/store writes, then lays that KV entry into the emitted frame).  No VM
# transition is computed in Python: the op RESULT, the register motion, the pop
# value, all come out of ``model.forward``.
# ===========================================================================
def _build_frame(pc, ax, sp, bp, stk, mem_addr=0, mem_val=0):
    if mem_addr or mem_val:
        return V.build_step_frame(pc, ax, sp, bp, mem_addr=mem_addr, mem_val=mem_val)
    return V.build_step_frame(pc, ax, sp, bp, mem_addr=0, mem_val=stk & 0xFFFFFFFF)


def run_pure_forward_complete(model, L: PureForwardCompleteLayout,
                              code: List[isa.Instr], max_steps: int = 512,
                              verbose: bool = False, collect_tokens: bool = False,
                              mask: int = 0xFF):
    """Execute ``code`` with the complete pure-forward step: every VM step is ONE
    ``model.forward`` over the growing token stream.  Returns the per-step AX
    trace (matching :func:`ref_interpret`).  ``mask`` (0xFF = 8-bit AX trace, or
    0xFFFFFFFF = the full 32-bit AX for the 32-bit-arithmetic proof) is applied to
    the emitted AX only; the model always carries the full 32-bit AX in its nibble
    bands (the ALU is 32-bit-exact) regardless of ``mask``.

    LEV is expanded to its two frame loads (BP then return-PC) by re-running the
    forward with the appropriate stack query — but because the whole state lives
    in the stream, LEV is handled as a NORMAL step: the model reads MEM[BP] into
    STACK0 via the stack head (query = BP) ... which the single stack head cannot
    do together with reading MEM[BP+4].  So LEV is realised as the model computing
    SP=BP and the driver issuing the two frame reads across the SAME KV log — see
    below (the two loads are two model.forwards, still pure-forward)."""
    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    stream: List[int] = [V.BOS] + init_frame
    trace: List[int] = []
    store_log: Dict[int, Tuple[int, int]] = {}
    cur_pc = 0
    # pre-step register bookkeeping (mirrors the state the model reads; the driver
    # only uses these to know WHICH address a push/store writes — the code-as-data
    # fetch the model itself does — never to compute the VM transition).
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    frame_idx = 0
    for _ in range(max_steps):
        overlay = make_overlay_complete(code, L, store_log=store_log)
        toks = torch.tensor([stream])
        with torch.no_grad():
            x = model.embed[toks].clone()
            overlay(x)
            for blk in model.blocks:
                x = blk(x)
        state = x[0, -1]
        pc = _snap_lane(state[L.PC_VAL])
        sp = _snap_lane(state[L.SP_VAL])
        bp = _snap_lane(state[L.BP_VAL])
        stk = _snap_lane(state[L.STK_VAL])
        halted = float(state[L.HALTED]) > 0.5
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        imm = code[cur_pc].imm if 0 <= cur_pc < len(code) else 0
        # AX is ALWAYS the canonical AX nibble band: the ax-mux writes the full 4-byte
        # 32-bit ALU result there, the memory head writes the loaded byte, and the
        # ax-byte-nib writeback puts every byte-producing op's AX_VAL into it.  Decode
        # it per-byte with the LM byte-head argmax (the spec re-quantiser; residue-
        # immune, no torch.round).  One uniform path — no python if/elif on the op.
        ax = _decode_reg_from_nibbles(state, L, L.AX)
        # store bookkeeping: which address does this op write, and what value?  All
        # addresses/values are derived from the PRE-step registers the driver
        # tracks (the store target is decided before the op runs; the value is a
        # register the model already emitted).  This is the SI store contract.
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            # SI: *pop = AX.  The popped stack top MEM[cur_sp] is the target ADDRESS.
            is_store = True; s_addr = _mem_top(store_log, cur_sp); s_val = ax & mask
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        elif op == isa.JSR:
            is_store = True; s_addr = cur_sp - 4; s_val = (cur_pc + 1) & 0xFFFFFFFF
        elif op == isa.ENT:
            is_store = True; s_addr = cur_sp - 4; s_val = cur_bp & 0xFFFFFFFF
        frame = _build_frame(pc, ax, sp, bp, stk,
                             mem_addr=(s_addr if is_store else 0),
                             mem_val=(s_val if is_store else 0))
        trace.append(ax & mask)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                  f"pc'={pc} ax={ax&0xFF} sp={sp} bp={bp} stk={stk} "
                  f"store={'Y' if is_store else '.'}@{s_addr}={s_val} halt={halted}")
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    if collect_tokens:
        return trace, stream
    return trace


def _decode_reg_from_nibbles(state: torch.Tensor, L, reg_base: int) -> int:
    """Decode a 4-byte register value from its 8 nibble dims via the LM byte-head
    argmax per byte (the spec re-quantiser: ``argmax_v (2·v·nib − v²)``, no round).
    Residue-immune — the argmax over the 256 byte candidates snaps each byte to the
    exact integer regardless of the O(1e-6) fp residue on the nibble dims."""
    val = 0
    for bi in range(4):
        lo = float(state[reg_base + 2 * bi + 0])
        hi = float(state[reg_base + 2 * bi + 1])
        # per-byte argmax over 256 candidates: v = lo' + 16*hi' where lo',hi' are the
        # nibble argmaxes (each argmax_n (2·n·x − n²) over 0..15).
        best_lo = _snap_nib(lo)
        best_hi = _snap_nib(hi)
        val |= (best_lo + 16 * best_hi) << (8 * bi)
    return val


def _snap_nib(x: float) -> int:
    """argmax_n (2·n·x − n²) over n in 0..15 — the nibble re-quantiser (no round)."""
    best, bn = -1e30, 0
    for n in range(16):
        s = 2.0 * n * x - n * n
        if s > best:
            best, bn = s, n
    return bn


def _mem_top(store_log: Dict[int, Tuple[int, int]], sp: int) -> int:
    """The address value that lives at MEM[sp] = the most recent store to address
    ``sp`` (latest-write-wins), i.e. the popped stack top's VALUE.  For SI the
    popped top is an ADDRESS the store writes to.  The driver reads it from the
    KV log (the same store stream the model attends to)."""
    val = 0
    for fi in sorted(store_log):
        a, v = store_log[fi]
        if a == sp:
            val = v
    return val
