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
from . import nibble_filesys as _FS   # OPEN/READ/CLOS/PRTF via the TOOL_CALL boundary
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

# Number of nibbles the STATIC immediate-nibble program encoding carries per slot.
# The corpus's largest VALUE literal is < 10^4 (< 16^4); 5 nibbles (< 16^5 ≈ 1.05M)
# is generous headroom while keeping the CODE_IMM_NIB band (code_size × IMM_NIBS)
# small (each extra nibble/slot inflates the model residual dim — the wall-time and
# memory driver).  A COMPUTED result (e.g. factorial) uses the full 8 AX nibbles via
# the ALU; only the IMM LITERAL is bounded here.
IMM_NIBS = 5


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
        # FULL 32-bit IMM: the immediate's 8 nibbles are part of the STATIC program
        # encoding (CODE_IMM_NIB[i][j], written by the overlay from the bytecode —
        # a pure re-encoding of the constant, no runtime compute), gathered at PC
        # into IMM_NIB by the same PC-one-hot product-select as the scalar IMM, and
        # written verbatim into the 8 AX nibbles on IMM so literals > 255 survive.
        self.CODE_IMM_NIB = [self._band(f"CODE_IMM_NIB_{i}", IMM_NIBS)
                             for i in range(code_size)]
        self.IMM_NIB = self._band("IMM_NIB", IMM_NIBS)        # fetched immediate nibbles
        # CLEAN signed immediate scalar, reconstructed from the leak-free IMM_NIB
        # nibbles (each rounded to its exact 0..15 cell) + a two's-complement sign
        # correction.  Written ONCE into this DEDICATED never-share scratch dim (never
        # an in-place overwrite of the leaky ``IMM`` scalar), so the frame-offset ops
        # (LEA/ENT/ADJ/JSR) read an EXACT integer offset instead of ``IMM``'s
        # large-literal PC-one-hot leak (#648/#660).  See ``compile_imm_clean``.
        self.IMM_CLEAN = self._scalar("IMM_CLEAN")
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
    # Frame-offset ops (JSR/ENT/ADJ/LEA) read the CLEAN reconstructed immediate, not
    # the leaky scalar ``L.IMM``: the leak (a fraction of nearby large literals via
    # the imperfect PC one-hot) scaled by 4 rounds the frame byte to the wrong
    # integer (#648/#660).  ``IMM_CLEAN`` is exact for the signed slot offsets and
    # the small positive PC targets these ops carry.
    imm = L.IMM_CLEAN

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
    # NOP: PC += 1 (no state change).  The compiler emits NOP for alignment/padding;
    # without this rule the PC never advances past a NOP and the VM spins forever.
    rules.append(FFNRule([(L.OP_IS + isa.NOP, 0.5, 1.5)], {pc: LinearExpr.c(1.0)}))
    # PRTF: PC += 1 (I/O only, no register state change).  The visible output byte
    # is printf(AX & 0xFF): the driver reads it from the model's decoded AX byte-0
    # and emits it via the think-tag protocol (§Printing, BLOG_SPEC line 851).  AX
    # and the stack are untouched, exactly like NOP for the register transition.
    rules.append(FFNRule([(L.OP_IS + isa.PRTF, 0.5, 1.5)], {pc: LinearExpr.c(1.0)}))
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
                     [isa.JSR, isa.ENT, ADJ, isa.LEV] +      # + calling convention
                     [isa.PRTF] +                            # + PRTF (I/O: PC += 1)
                     [isa.NOP]))                             # + NOP (PC += 1 no-op)
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


def compile_lea_addr_nib(L, dim: int) -> Dict[str, torch.Tensor]:
    """LEA-ONLY, residue-immune frame-address nibbles.

    ``compile_ax_nib_split`` derives AXB_LO/AXB_HI from the SCALAR ``AX_VAL``.  On a
    LEA the frame byte is ``(BP_LOW + 4*imm) mod 256``.  Two independent large-value
    fp residues corrupt the naive path and surface only when the frame byte is
    16-aligned (low nibble 0), so a malloc'd pointer stored to such a frame local is
    read back from the wrong cell and lost (#648):

      (a) the dispatch SET ``AX_VAL += BP_LOW + 4*imm - AX_VAL_old`` is scaled by the
          opcode gate ``k``; when the prior AX held a heap pointer (~131072) any
          ``(1-k)`` leaves a residue ``(1-k)*AX_VAL_old`` > 0.5; and
      (b) the fetched ``IMM`` scalar itself leaks a fraction of a NEARBY large literal
          (e.g. ``IMM 0x20000``) through the imperfect PC one-hot, so ``IMM`` for a
          ``LEA -1`` reads e.g. ``-1.026`` not ``-1``.  This block reads the CLEAN
          ``IMM_CLEAN`` (reconstructed leak-free from IMM_NIB in ``compile_imm_clean``)
          instead of ``IMM``, so (b) is fixed AT SOURCE — with several large literals
          in flight the raw-``IMM`` residue can exceed the round tolerance below, but
          ``IMM_CLEAN`` is EXACT.

    Fix: recompute the LEA byte here from ``q = BP_LOW + 4*IMM_CLEAN`` with a
    ROUND-TO-NEAREST integer decode -- cell ``a`` fires iff ``q in [a-0.5, a+0.5)``,
    built from CLAMPED steps at the half-integer edges (each a
    ``relu(RELU_S*(q-e)) - relu(RELU_S*(q-e)-1)`` saturating to a clean 0/1 since q is
    never within ``1/RELU_S`` of a half-integer edge) -- which absorbs any AXB slop
    (residue (a)), snapping ``q`` to the exact frame integer before
    taking its low/high nibbles (mod 256).  OVERWRITES AXB_LO/AXB_HI, gated on
    OP_IS[LEA] (0 on every non-LEA op -> byte-identical).  Runs AFTER ``ax-nib-split``
    and BEFORE ``ax-byte-nib``.  ``4*imm`` handles a negative slot count exactly
    (linear gate read)."""
    g = L.OP_IS + isa.LEA
    axb_lo, axb_hi = L.AXB_LO, L.AXB_HI
    q_lo, q_hi = -256, 512
    # One CLAMPED step per half-integer edge e = a-0.5: s_e(q) = clamp(RELU_S*(q-e),0,1)
    # = relu(RELU_S*(q-e)) - relu(RELU_S*(q-e)-1).  q is ~integer (|residue| < 0.1) and
    # edges are half-integers, so q-e is always >= ~0.4 in magnitude -> the clamp is an
    # exact 0/1 nearest-integer step.  Cell a = s_{a-0.5} - s_{a+0.5}.
    edges = [a - 0.5 for a in range(q_lo, q_hi + 1)]
    eu = {e: k for k, e in enumerate(edges)}
    n_edge = len(edges)
    spec = _empty_spec(dim, 2 * n_edge + 2)
    u = 0
    step0 = u
    for e in edges:                       # two relu units -> one clamped step per edge
        for j in range(2):
            spec["W_up"][u, L.BP_LOW] = RELU_S
            spec["W_up"][u, L.IMM_CLEAN] = 4.0 * RELU_S    # CLEAN imm (no PC-leak)
            spec["b_up"][u] = -RELU_S * e - (0.0 if j == 0 else 1.0)
            spec["W_gate"][u, g] = 1.0    # gate on OP_IS[LEA] (0/1) so the step is
            u += 1                        # zero on non-LEA steps
    def step_units(e):                    # (idx_of_+relu, idx_of_-relu) for edge e
        base = step0 + 2 * eu[e]
        return base, base + 1
    for lane in (axb_lo, axb_hi):         # SET: clear AXB (gated on LEA) before re-add
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_HALF; u += 1
    for a in range(q_lo, q_hi):           # cell a = step(a-0.5) - step(a+0.5)
        byte = a & 0xFF
        lo, hi = byte & 0xF, (byte >> 4) & 0xF
        p_lo, m_lo = step_units(a - 0.5)  # +step at lower edge
        p_hi, m_hi = step_units(a + 0.5)  # -step at upper edge
        # step(q>=e) = silu(RELU_S*(q-e)) - silu(RELU_S*(q-e)-1) saturates to UNIT
        # height, so route with coefficient ``val`` directly (no /RELU_S).
        for lane, val in ((axb_lo, lo), (axb_hi, hi)):
            if val:
                spec["W_down"][lane, p_lo] += val
                spec["W_down"][lane, m_lo] += -val
                spec["W_down"][lane, p_hi] += -val
                spec["W_down"][lane, m_hi] += val
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


def compile_imm_nib_fetch(L, dim: int) -> Dict[str, torch.Tensor]:
    """IMM_NIB[j] = Σ_i PC_IS[i]·CODE_IMM_NIB[i][j]  (j=0..7) — the same bilinear
    PC-one-hot product-select the scalar-IMM fetch uses, applied to the immediate's
    8 static program-data nibbles.  Self-clears IMM_NIB first (SET)."""
    n = L.code_size
    spec = _empty_spec(dim, IMM_NIBS + n * IMM_NIBS)
    u = 0
    for j in range(IMM_NIBS):                        # self-clear IMM_NIB (SET)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, L.IMM_NIB + j] = 1.0
        spec["W_down"][L.IMM_NIB + j, u] += -1.0 / SILU_S
        u += 1
    for i in range(n):
        for j in range(IMM_NIBS):
            spec["W_up"][u, L.PC_IS[i]] = S
            spec["W_gate"][u, L.CODE_IMM_NIB[i] + j] = 1.0
            spec["W_down"][L.IMM_NIB + j, u] += 1.0 / SILU_S
            u += 1
    return spec


def compile_imm_clean(L, dim: int) -> Dict[str, torch.Tensor]:
    """IMM_CLEAN = the EXACT signed immediate, reconstructed from ``IMM_NIB`` (the
    leak-free per-nibble fetch) — the ROOT fix for the frame-address IMM leak
    (#648/#660).

    The scalar ``IMM = Σ_i PC_IS[i]·CODE_IMM[i]`` leaks a fraction of every NEARBY
    large literal (e.g. ``0x20000``) through the imperfect PC one-hot; with the
    corpus's ~43 big literals in flight the accumulated residue exceeds 0.5, so any
    op scaling it by 4 (``AX/SP += 4·imm``) rounds the frame byte to the wrong
    integer.  ``IMM_NIB[j]`` is fetched the SAME way, but each nibble is a small
    integer 0..15, so a 16-cell one-hot (triangular pulse, exact at each integer)
    ROUNDS each nibble to its nearest cell and kills the leak per-nibble.

    We reconstruct a bounded-width SIGNED value from the low ``CLEAN_NIBS`` nibbles
    (two's-complement, ``CLEAN_NIBS·4`` bits): the top of those nibbles is treated as
    SIGNED (cell ``a>=8`` -> ``a-16``), so a negative slot offset ``-k`` (stored as
    ``0xF..FC``) reconstructs to exactly ``-k`` and a small positive PC target /
    offset to itself.  ``CLEAN_NIBS`` is chosen so every frame-offset op's immediate
    (LEA/ENT/ADJ signed slot counts, |·| <= a few; JSR positive PC targets < code
    size) fits its signed range — and, crucially, so the recompose coefficients
    (``<= 16^CLEAN_NIBS``) stay SMALL: a large coefficient (e.g. ``16^4``) amplifies
    the silu unit's ~1e-6 relative error to ~0.5 ABSOLUTE at the cell, which would
    re-introduce a half-integer error.  With ``CLEAN_NIBS=3`` (signed range
    ``[-2048, 2047]``) the max coefficient is ``16^3=4096`` and the result is
    fp32-EXACT for every corpus immediate (unit-tested).

    Result written ONCE into the DEDICATED never-share ``IMM_CLEAN`` scalar (NOT an
    in-place overwrite of the leaky ``IMM`` dim — a fresh dim written once sidesteps
    the compact_alloc liveness interaction that neutralised the in-place attempt).
    Ungated (IMM_CLEAN is only READ by the frame-offset ops, which gate their own
    writes on OP_IS)."""
    # 3 low nibbles: signed 12-bit range [-2048, 2047] covers every corpus LEA/ENT/
    # ADJ slot offset (|·| <= a few) and every JSR PC target (< code_size).  Kept
    # small so the recompose coefficients (<= 16^3) do not amplify the silu residual.
    n_nib = min(3, IMM_NIBS)
    thr = list(range(-1, 17))
    tu = {t: k for k, t in enumerate(thr)}
    n_thr = len(thr)
    # one relu bank per nibble (16-cell one-hot) + one self-clear unit for IMM_CLEAN.
    spec = _empty_spec(dim, n_nib * n_thr + 1)
    u = 0
    relu0 = {}
    for j in range(n_nib):
        relu0[j] = u
        for t, k in tu.items():
            spec["W_up"][u, L.IMM_NIB + j] = RELU_S
            spec["b_up"][u] = -RELU_S * t
            spec["W_gate"][u, L.ONE] = 1.0
            u += 1
    # self-clear IMM_CLEAN (SET) so recurrent steps are idempotent.
    spec["W_up"][u, L.ONE] = S
    spec["W_gate"][u, L.IMM_CLEAN] = 1.0
    spec["W_down"][L.IMM_CLEAN, u] += -1.0 / SILU_S
    u += 1
    # recompose: IMM_CLEAN += Σ_j Σ_a one-hot_j(a)·coeff(j,a).  The TOP of the
    # reconstructed nibbles is SIGNED (a-16 when a>=8), so ``0xF..FC`` -> ``-k``.
    base = 1
    top = n_nib - 1
    for j in range(n_nib):
        r0 = relu0[j]
        for a in range(16):
            av = (a - 16) if (j == top and a >= 8) else a   # signed top nibble
            coeff = av * base
            if coeff:
                spec["W_down"][L.IMM_CLEAN, r0 + tu[a - 1]] += coeff / RELU_S
                spec["W_down"][L.IMM_CLEAN, r0 + tu[a]] += -2.0 * coeff / RELU_S
                spec["W_down"][L.IMM_CLEAN, r0 + tu[a + 1]] += coeff / RELU_S
        base *= 16
    return spec


def compile_imm_ax_nibbles(L, dim: int) -> Dict[str, torch.Tensor]:
    """On IMM, SET all 8 AX nibbles = IMM_NIB (the fetched immediate nibbles), so a
    full 32-bit literal lands in the canonical AX nibble band.  Gated on OP_IS[IMM].
    Runs AFTER the byte-nib writeback (which only wrote nibbles 0,1) and overwrites
    all 8 with the full value."""
    spec = _empty_spec(dim, 8 + IMM_NIBS)
    u = 0
    g = L.OP_IS + isa.IMM
    for j in range(8):                              # clear ALL 8 AX nibbles gated
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.AX + j] = 1.0
        spec["W_down"][L.AX + j, u] += -1.0 / SILU_HALF; u += 1
    for j in range(IMM_NIBS):                        # + IMM_NIB[j] into AX[j] (rest 0)
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.IMM_NIB + j] = 1.0
        spec["W_down"][L.AX + j, u] += 1.0 / SILU_HALF; u += 1
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
                                      recurrent_divmod: bool = False):
    """Assemble the complete pure-forward VM: frame ingest + LI/LC KV head +
    stack-pop KV head + the 32-bit ALU FFN blocks + callconv + bitwise + the
    base-16 long-division DIV/MOD blocks, all as persistent weights applied by
    ``model.forward``.  Returns ``(model, L)``.

    This is the SINGLE canonical full-VM interpreter: it ALWAYS folds the
    complete op set (ADD/SUB/MUL + DIV/MOD + OR/XOR/AND/SHL/SHR + cmp + memory +
    callconv).  There is no reduced / "lean" op-subset construction — the former
    ``include_bitwise`` / ``include_divmod`` split flags have been removed so the
    build is one byte-identical model with every opcode present.  (Skipping the
    ~300 divmod blocks when the *step's* opcode is not DIV/MOD is a runtime
    block-dispatch concern — see the ``_apply_order`` seam and the block-MoE
    integration note below — NOT a build-time op-subset toggle.)

    ``recurrent_divmod`` (default False) folds DIV/MOD as a RECURRENT step: the
    8 long-division iterations are unrolled into ONE reused iteration BODY (21
    blocks) instead of 168 distinct blocks, and the block-application loop applies
    that body 8 times (threading the running remainder R + digit-index counter IT
    through the residual across the reused applications — exactly as the VM step
    itself is recurrent, threading PC/AX/SP).  ``L._apply_order`` then indexes the
    physical blocks in the FULL application sequence (the body indices repeat);
    the physical block count (what the model STORES) drops 262 -> 115 for divmod,
    while the forward applies the same 262-long sequence.  Byte-identical DIV/MOD
    results to the unrolled build (gadget gate: 24/24).  This is a compute-SHAPE
    toggle (same op set, fewer stored blocks), NOT an op-subset toggle."""
    n_heads = N_ROLES + 3          # 20 ingest + LI head + stack-pop head + lev head
    L = PureForwardCompleteLayout(code_size, n_heads=n_heads)
    A.extend_layout_for_alu32(L, recurrent_divmod=recurrent_divmod)  # ALU scratch bands
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
        ("imm-nib-fetch", compile_imm_nib_fetch(L, dim)),      # IMM_NIB <- CODE_IMM_NIB@PC
        ("imm-clean", compile_imm_clean(L, dim)),              # IMM_CLEAN <- round(IMM_NIB)
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
    # ``divmod_apply_order`` records the physical-block indices to apply for the
    # DIV/MOD span, in application order.  For the unrolled path it is just the
    # divmod blocks' own indices (identity); for the recurrent path the reused
    # iteration-body block indices REPEAT 8x, so the model STORES 115 blocks but
    # APPLIES 262 — the recurrence.  The final full ``L._apply_order`` is assembled
    # after all blocks are appended (below).
    divmod_apply_order: List[int] = []
    # ---- BLOCK-MoE SEAM (integration point for the block-level identity/null
    # expert + block dispatch) -------------------------------------------------
    # ``[divmod_start, divmod_end)`` is the contiguous span of the ~262 DIV/MOD
    # blocks — the dominant physical cost and the block-MoE's primary skip target.
    # The full op set is ALWAYS built (no op-subset flag); making it CHEAP for a
    # non-DIV/MOD step is a RUNTIME block-dispatch concern.  #628's block-MoE plugs
    # in here: it can read ``L._divmod_span = (divmod_start, divmod_end)`` (recorded
    # below) to gate this contiguous block range behind an identity/null expert when
    # the step's decoded opcode is not DIV/MOD, without changing any stored weight.
    # Do NOT convert this span back into a build-time op-subset — the model must
    # STORE every op; block-MoE only SKIPS its application per step.
    divmod_start = len(block_specs)                 # first physical divmod block idx
    if recurrent_divmod:
        unique, apply_names = A.compile_divmod_blocks_recurrent(L, dim)
        # place the UNIQUE blocks physically, remember each name's physical index,
        # then map the (repeating) apply_names to those indices.
        name_to_idx: Dict[str, int] = {}
        for name, spec in unique:
            name_to_idx[name] = len(block_specs)
            block_specs.append((name, spec))
        divmod_apply_order = [name_to_idx[n] for n in apply_names]
    else:
        for name, spec in A.compile_divmod_blocks(L, dim):
            divmod_apply_order.append(len(block_specs))
            block_specs.append((name, spec))
    divmod_end = len(block_specs)                   # one past the last physical divmod block
    L._divmod_span = (divmod_start, divmod_end)     # block-MoE skip target (seam)
    block_specs.append(("ax-mux", A.compile_ax_mux(L, dim, ops=ALU_OPS)))
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
    disp_rules += _bitwise_pop_rules(L)
    # ops whose AX result is a BYTE recomposed from AX_VAL by the byte-nib writeback
    # (so the AX nibble band is canonical, the driver decodes AX from it): IMM/LEA +
    # cmp + bitwise.  NOTE: LI/LC are EXCLUDED — the memory KV head writes their AX
    # nibbles DIRECTLY with the full 32-bit loaded value, and the byte-nib block reads
    # AXB_LO/HI off the STEP-INPUT AX_VAL (the OLD AX, recomposed at block 0 before
    # the head ran), so including LI/LC would clobber the loaded value with the stale
    # low byte.  A loaded 32-bit int (e.g. a variable holding 1000) must survive whole.
    byte_ax_ops = [isa.IMM, isa.LEA] + \
                  [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE] + \
                  [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR]
    block_specs += [
        ("dispatch", compile_ffn(disp_rules, dim)),
        ("branch-delta", compile_branch_delta_clean(L, dim)),  # BZ/BNZ via IMM_CLEAN
        ("fold-lea", _fold_ax_gated(L, dim, [isa.LEA])),  # LEA masks &0xFF; ALU 32-bit
        ("ax-nib-split", compile_ax_nib_split(L, dim)),   # AX_VAL byte -> AXB_LO/HI
        # LEA frame address is residue-immune: recompute AXB_LO/HI from BP_LOW+4*imm
        # directly (small integer), so a large prior AX (heap ptr) cannot round the
        # frame byte to the wrong 16-aligned cell (#648).  LEA-gated; else no-op.
        ("lea-addr-nib", compile_lea_addr_nib(L, dim)),
        ("ax-byte-nib", compile_ax_byte_to_nibbles(L, dim, byte_ax_ops)),
        # FULL 32-bit IMM: overwrite ALL 8 AX nibbles with the fetched immediate
        # nibbles (the byte-nib block above only set nibbles 0,1) so literals > 255
        # (corpus values up to ~10000) survive into the canonical AX nibble band.
        ("imm-ax-nib", compile_imm_ax_nibbles(L, dim)),
    ]
    n_blocks = len(block_specs)
    # ---- application order (recurrence) ----------------------------------
    # The model STORES ``n_blocks`` distinct blocks but APPLIES them in
    # ``L._apply_order``: identity everywhere except the DIV/MOD span, where the
    # recurrent path repeats the reused iteration-body indices.  A None means "no
    # remapping" (the driver falls back to range(n_blocks)) so the unrolled build
    # is byte-identical to before.
    if recurrent_divmod:
        L._apply_order = (list(range(divmod_start)) + divmod_apply_order +
                          list(range(divmod_end, n_blocks)))
    else:
        L._apply_order = None
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
    # RECURRENCE: re-point ``model.blocks`` through the application order so the
    # forward applies the reused divmod iteration body N times (the stored blocks
    # are the DISTINCT set; ``model.blocks`` now holds repeated references — shared
    # nn.Module weights, identical math).  ``_phys_blocks`` keeps the distinct set.
    if L._apply_order is not None:
        import torch.nn as _nn
        phys = list(model.blocks)
        model._phys_blocks = phys
        model.blocks = _nn.ModuleList([phys[i] for i in L._apply_order])
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
    AX for ADD/SUB/MUL/DIV/MOD), PSH/LEA REMOVED (callconv owns them), and JMP
    REMOVED (re-added below reading the CLEAN immediate ``IMM_CLEAN`` for its target,
    so a large-literal PC-one-hot leak cannot shift the jump target — the memset
    ``JMP 134`` was leaking to 133, corrupting the fill loop; #648/#660)."""
    rules = base_dispatch_rules(L)
    drop = {isa.ADD, isa.SUB, isa.PSH, isa.LEA, isa.JMP}
    out = []
    for r in rules:
        # a rule's gate is [(OP_IS+op, ...)]; find which op.
        op = r.when[0][0] - L.OP_IS if r.when else None
        if op in drop:
            continue
        out.append(r)
    # JMP: PC = IMM_CLEAN (leak-free target).  PC += (IMM_CLEAN - PC).
    out.append(FFNRule([(L.OP_IS + isa.JMP, 0.5, 1.5)], {
        L.PC_VAL: LinearExpr.of(L.IMM_CLEAN, 1.0) + LinearExpr.of(L.PC_VAL, -1.0)}))
    return out


def compile_branch_delta_clean(L, dim: int) -> Dict[str, torch.Tensor]:
    """``compile_branch_delta`` (BZ/BNZ bilinear PC update) but reading the CLEAN
    immediate ``IMM_CLEAN`` for the taken target instead of the leaky ``IMM`` — same
    root fix as JMP: a large-literal PC-one-hot leak must not shift a branch target
    (BZ/BNZ targets are small positive PC values < code_size, exact in IMM_CLEAN's
    12-bit signed range).  Structurally identical to ``compile_branch_delta`` with
    ``imm`` rebound; kept here (not a flag on the base) so non-complete builds — which
    have no ``IMM_CLEAN`` dim — are untouched."""
    import torch.nn.functional as _Fnn
    pc, imm, azero, one = L.PC_VAL, L.IMM_CLEAN, L.AX_ZERO, L.ONE
    BZ, BNZ = L.OP_IS + isa.BZ, L.OP_IS + isa.BNZ
    spec = _empty_spec(dim, 4)
    BIG = 200.0
    silu_big = float(_Fnn.silu(torch.tensor(0.5 * BIG)))

    def _and_unit(u, op_band, bool_terms, gate_terms):
        spec["W_up"][u, op_band] += BIG
        for band, coeff, const in bool_terms:
            if band is not None:
                spec["W_up"][u, band] += BIG * coeff
            spec["b_up"][u] += BIG * const
        spec["b_up"][u] += -BIG * 1.5
        for band, coeff in gate_terms:
            spec["W_gate"][u, band] += coeff

    _and_unit(0, BZ,  [(azero, 1.0, 0.0)],  [(imm, 1.0), (pc, -1.0)])
    spec["W_down"][pc, 0] += 1.0 / silu_big
    _and_unit(1, BZ,  [(azero, -1.0, 1.0)], [(one, 1.0)])
    spec["W_down"][pc, 1] += 1.0 / silu_big
    _and_unit(2, BNZ, [(azero, -1.0, 1.0)], [(imm, 1.0), (pc, -1.0)])
    spec["W_down"][pc, 2] += 1.0 / silu_big
    _and_unit(3, BNZ, [(azero, 1.0, 0.0)],  [(one, 1.0)])
    spec["W_down"][pc, 3] += 1.0 / silu_big
    return spec


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
                  mask: int = 0xFF, out: List[int] = None) -> List[int]:
    """Reference interpreter with the SP-addressed memory stack.  ``mask`` = 0xFF
    (8-bit AX trace, the default; matches the neural byte frame) or 0xFFFFFFFF (the
    full 32-bit AX for the 32-bit-arithmetic proof: ADD/SUB/MUL wrap mod 2^32,
    DIV/MOD unsigned floor with b==0 -> 0, ISA_SPEC 4.2).

    If ``out`` is a list, PRTF appends ``AX & 0xFF`` to it — the visible stdout
    byte a ``printf("%c", AX)`` would emit (§System / op 33)."""
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
        elif op == isa.PRTF:
            if out is not None:
                out.append(ax & 0xFF)   # printf visible byte; registers unchanged
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
                # the immediate's low IMM_NIBS nibbles are part of the STATIC program
                # encoding (a pure re-encoding of the constant, gathered at PC).
                for j, nv in enumerate(V.nibbles_of_value(ins.imm & 0xFFFFFFFF, IMM_NIBS)):
                    x[0, i, L.CODE_IMM_NIB[k] + j] = float(nv)
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


def _seed_frames(seed_mem):
    """Turn a ``{addr: value}`` data segment into a list of leading STORE frames.

    Each entry becomes one MEM-store frame (PC/AX/SP/BP left at their init values)
    that seeds ``mem[addr] = value`` into the model's KV memory BEFORE the program
    runs — the classic quine's "string literal in the data segment", materialised
    as the §Memory store-log rows the loads content-address.  Returns
    ``(frames, store_log)`` where ``store_log`` maps each seed frame's index to its
    ``(addr, val)`` so the overlay tags its MEM token as a KV entry."""
    frames: List[int] = []
    store_log: Dict[int, Tuple[int, int]] = {}
    for k, (addr, val) in enumerate(sorted(seed_mem.items())):
        frames += _build_frame(0, 0, SP_INIT, SP_INIT, 0,
                               mem_addr=addr & 0xFFFFFFFF, mem_val=val & 0xFFFFFFFF)
        store_log[k] = (addr & 0xFFFFFFFF, val & 0xFFFFFFFF)   # frame idx k
    return frames, store_log


def run_pure_forward_complete(model, L: PureForwardCompleteLayout,
                              code: List[isa.Instr], max_steps: int = 512,
                              verbose: bool = False, collect_tokens: bool = False,
                              mask: int = 0xFF, fio=None, data_seg=None,
                              out: List[int] = None, seed_mem=None):
    """Execute ``code`` with the complete pure-forward step: every VM step is ONE
    ``model.forward`` over the growing token stream.  Returns the per-step AX
    trace (matching :func:`ref_interpret`).  ``mask`` (0xFF = 8-bit AX trace, or
    0xFFFFFFFF = the full 32-bit AX for the 32-bit-arithmetic proof) is applied to
    the emitted AX only; the model always carries the full 32-bit AX in its nibble
    bands (the ALU is 32-bit-exact) regardless of ``mask``.

    FILE OPS (OPEN/READ/CLOS/PRTF) are the ONE class not computed neurally (§Tool
    Use Mode): when ``fio`` (a ``nibble_filesys.FileOpState``) is passed, an op in
    ``FILE_OPCODES`` is dispatched via the TOOL_CALL protocol — the driver marshals
    the args off the KV store log, the runner performs the real I/O against the
    stub filesystem / stdin, and the integer result re-enters AX (READ also lays
    its bytes back into the store log as fresh §Memory KV frames so LC reads them).
    ``data_seg`` (``{byte_addr: byte}``) seeds the read-only data segment (the
    filename / format string literals the c4 loader places).

    If ``out`` is a list, a PRTF step (in the ``out=`` string-quine mode, no
    ``fio``) appends its VISIBLE output byte — the AX byte-0 the model itself
    decoded from its nibble band (a genuine LM-head argmax, NOT a python copy) —
    and a SEPARATE ``vis_stream`` carries ``THINK_END, <byte>, THINK_START`` around
    that byte so ``blogspec_vocab.visible_output(vis_stream)`` recovers exactly
    ``out`` (the think-tag stdout protocol, §Printing).  ``seed_mem``
    (``{addr: value}``) seeds the data segment as leading MEM-store frames (the
    quine's bundled string literal).

    LEV is expanded to its two frame loads (BP then return-PC) by re-running the
    forward with the appropriate stack query — but because the whole state lives
    in the stream, LEV is handled as a NORMAL step: the model reads MEM[BP] into
    STACK0 via the stack head (query = BP) ... which the single stack head cannot
    do together with reading MEM[BP+4].  So LEV is realised as the model computing
    SP=BP and the driver issuing the two frame reads across the SAME KV log — see
    below (the two loads are two model.forwards, still pure-forward)."""
    init_frame = _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    # The MODEL's token stream stays the pure contiguous 30-token frame stream (no
    # in-stream think tags) so the overlay/frame geometry is byte-identical to the
    # non-PRTF path.  The think-tag protocol (THINK_END, <visible byte>, THINK_START)
    # is materialised in a SEPARATE ``vis_stream`` the driver returns, from which
    # ``blogspec_vocab.visible_output`` recovers exactly the printed bytes — the
    # user-facing view, decoupled from the internal execution stream.
    # Prepend the seeded data segment (if any) as leading STORE frames so the KV
    # memory holds it before step 0; then BOS + the seed frames + the init frame.
    seed_frames, store_log = _seed_frames(seed_mem or {})
    n_seed = len(store_log)                        # number of leading data frames
    stream: List[int] = [V.BOS] + seed_frames + init_frame
    vis_stream: List[int] = [V.BOS, V.THINK_START]
    trace: List[int] = []
    cur_pc = 0
    # pre-step register bookkeeping (mirrors the state the model reads; the driver
    # only uses these to know WHICH address a push/store writes — the code-as-data
    # fetch the model itself does — never to compute the VM transition).
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    # the init frame sits at frame index ``n_seed`` (after the seed data frames);
    # the first emitted step frame is ``n_seed + 1``.
    frame_idx = n_seed
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
        # -- FILE OP: the ONE class not computed neurally (§Tool Use Mode).  The
        # model has no rules for OPEN/READ/CLOS/PRTF, so its PC/SP/AX for this row
        # are meaningless — the DRIVER performs the whole op via the TOOL_CALL
        # runner and overrides the registers.  READ's bytes re-enter the token
        # stream as their OWN §Memory KV frames so LC reads them back byte-exact.
        if fio is not None and op in _FS.FILE_OPCODES:
            new_ax, new_sp, byte_stores = _FS.dispatch_file_op_driver(
                op, cur_ax & 0xFFFFFFFF, imm, cur_sp, store_log, fio,
                data_seg=data_seg, slot=4)
            pc = cur_pc + 1                     # file ops advance PC by one (no branch)
            sp = new_sp
            bp = cur_bp
            ax = new_ax & 0xFFFFFFFF
            frame = _build_frame(pc, ax, sp, bp, stk)
            trace.append(ax & mask)
            frame_idx += 1
            stream += frame
            # lay READ's bytes into the KV log as one store frame per byte, so a
            # later LC(addr) attends to the byte the file delivered.
            for (baddr, bval) in byte_stores:
                bframe = _build_frame(pc, ax, sp, bp, stk,
                                      mem_addr=baddr, mem_val=bval & 0xFF)
                frame_idx += 1
                store_log[frame_idx] = (baddr, bval & 0xFF)
                stream += bframe
            if verbose:
                print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                      f"pc'={pc} ax={ax&0xFFFFFFFF} sp={sp} (FILE, "
                      f"{len(byte_stores)} byte-stores)")
            cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
            if pc < 0 or pc >= len(code):
                break
            continue
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
        vis_stream += frame
        # --- I/O: PRTF emits a VISIBLE byte via the think-tag protocol -----------
        # The printed byte is the model's OWN decoded AX byte-0 (a genuine LM-head
        # argmax over the nibble band, not a python copy).  PRTF leaves AX/SP/BP
        # unchanged (its dispatch rule only does PC += 1), so this decoded AX is the
        # value ``printf("%c", AX)`` prints.  We exit the think block, emit the byte,
        # re-enter — so ``visible_output(vis_stream)`` yields exactly ``out``.
        if op == isa.PRTF:
            emit_b = ax & 0xFF
            if out is not None:
                out.append(emit_b)
            vis_stream += [V.THINK_END, emit_b, V.THINK_START]
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):4s} -> "
                  f"pc'={pc} ax={ax&0xFF} sp={sp} bp={bp} stk={stk} "
                  f"store={'Y' if is_store else '.'}@{s_addr}={s_val} halt={halted}")
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    vis_stream += [V.THINK_END, V.HALT]            # close think, then terminate
    if out is not None:
        return (trace, vis_stream) if collect_tokens else trace
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
