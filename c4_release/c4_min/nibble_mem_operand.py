"""nibble_mem_operand.py — the MEMORY-OPERAND addressing mode (C4_MEM_OPERAND).

The idea (and the thing this proves)
====================================
The interpreted pure-forward VM (``nibble_pure_forward_complete``) runs ONE
bytecode instruction per ``model.forward``.  Reading a memory operand costs a
whole VM step per load: ``IMM addr`` (put the address in AX, 1 step) + ``LI``
(the §Memory softmax1-CAM read, 1 step).  A single multiply-accumulate

    acc = acc + mem[a] * mem[b]

therefore compiles to ~10 VM steps: ``IMM a; LI; PSH; IMM b; LI; MUL; PSH;
IMM acc; LI; ADD`` — the LOADS dominate, and a length-N dot product is ~10N
model.forwards.

But the §Memory CAM read is done by a memory HEAD that runs EARLY in the block
stack (``mem-cam`` = physical block 7 of the complete model), while the
ADD/MUL compute runs LATE (``alu-expand`` = block 14, MUL = block 23+).  So a
read the head does in the early blocks of an instruction's OWN forward is fully
available to that same instruction's compute in the later blocks — no separate
``LI`` step is needed.  This module adds a single fused opcode

    MAC [addr_a], [addr_b]        # acc = acc + mem[addr_a] * mem[addr_b]

whose forward:
  * block 1            — snapshot the incoming accumulator (AX) into MAC_ACC
                         (before anything overwrites AX);
  * MAC-prep (pre-CAM) — on MAC set MAC_IS, expand the two frame addresses into
                         the two query-bit bands, clear STACK0 and AX so the two
                         CAM writes are clean SETs;
  * block 7 (mem-cam)  — TWO extra §Memory CAM heads (identical CAM to LI/SI)
                         read mem[addr_a] -> STACK0 and mem[addr_b] -> AX,
                         keyed on the addresses the FRAME carries (NOT on a
                         computed AX), enabled by MAC_IS;
  * block 14+ (MUL)    — the existing nibble-schoolbook MUL computes
                         STACK0 * AX = mem[a] * mem[b] into MUL_RES;
  * MAC-accumulate     — AX = MUL_RES + MAC_ACC  (32-bit carry chain), the fused
                         accumulate;
  * dispatch           — PC += 1 (MAC is a straight-line instruction).

So ``MAC [a],[b]`` is ONE VM step (one model.forward): the two reads and the
multiply-accumulate all fold into that single instruction's forward.  A dot
product is a chain of MACs (the accumulator round-trips through AX in the frame
between steps) — N model.forwards instead of ~10N.

Gating
======
Everything is behind ``C4_MEM_OPERAND`` (default OFF via ``mem_operand_enabled``).
With the flag OFF this module is never imported on the build path and the
complete-VM model + golden are byte-identical.  With the flag ON the builder
allocates the MAC bands, adds the MAC-prep / accumulate FFN blocks, bakes the
two extra CAM heads, and widens the opcode decode to MAC — all additive, all
MAC-gated, so every NON-MAC opcode is byte-identical to the flag-OFF build too
(the MAC bands are 0 on every non-MAC step; the two MAC heads sink to the
softmax1 zero-vector when MAC_IS=0, exactly like the LI head on a non-load).

The result is byte-exact vs numpy fixed-point and vs the SP-addressed
``ref_interpret`` reference (extended here with the MAC semantics).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import blogspec_vocab as V
from .nibble_vm import S, RELU_S, SILU_S, SILU_HALF, _empty_spec, _load_ffn, _zero_attn
from . import nibble_alu32 as A
from .blogspec_memory import ADDR_BITS
from .blogspec_layout import NIB_PER_REG
from . import nibble_pure_forward as PF
from . import nibble_pure_forward_complete as PFC
from .nibble_pure_forward import (
    N_ROLES, SP_INIT, _FRAME_ROLE_SLOTS, _MEM_MARKER_LOCAL, _address_bits,
    MEM_HEAD_CHANNELS, bake_frame_ingest, _bake_pf_memory_head,
    _flag_from_ops, _concat_specs,
)

# The fused multiply-accumulate opcode.  40 is the first free opcode value
# (isa.NUM_OPS = 40 covers 0..39); MAC decodes into its OWN dedicated OP_IS[MAC]
# lane (the layout widens OP_IS to include it ONLY under the flag) so the
# flag-OFF layout is untouched.
MAC = 40


def mem_operand_enabled() -> bool:
    """The memory-operand MAC (C4_MEM_OPERAND).  DEFAULT OFF.

    OFF -> this module is never on the build path; the complete-VM model and the
    golden are byte-identical.  ON -> the MAC opcode + its two early CAM reads +
    the fused accumulate are added (all MAC-gated, additive)."""
    return os.environ.get("C4_MEM_OPERAND", "0") != "0"


# ===========================================================================
# LAYOUT: the complete layout + the MAC scratch bands.  A dedicated OP_IS[MAC]
# decode lane (the ONLY widening of a base band, flag-gated), two 32-bit
# query-bit bands for the two operand addresses, the two address nibble bands
# the frame carries, the accumulator snapshot, and the private accumulate scratch.
# Every base / complete band offset is preserved (the MAC bands are appended
# after the complete layout's own dims), so a NON-MAC step is byte-identical.
# ===========================================================================
class MemOperandLayout(PFC.PureForwardCompleteLayout):
    def __init__(self, code_size: int, n_heads: int):
        super().__init__(code_size, n_heads=n_heads)
        self._off = self.D
        # Widen OP_IS by one lane so OP_IS[MAC] (=OP_IS+40) is a valid dim.  The base
        # band is OP_IS..OP_IS+39; this appends slot 40 at the END of the residual
        # (NOT contiguous with OP_IS), and we point OP_IS[MAC] reads there via a
        # small alias: the decode writes OP_IS + MAC, which must resolve to this dim.
        # Simplest: allocate a scalar and record its position; OP_IS + MAC is used
        # ONLY by our own MAC code, so we store it as MAC_OP_IS and translate.
        self.MAC_OP_IS = self._scalar("MAC_OP_IS")       # OP_IS[MAC] decode lane
        self.MAC_IS = self._scalar("MAC_IS")             # thresholded MAC indicator (CAM enable)
        self.MAC_A_QRY_BIN = self._band("MAC_A_QRY_BIN", ADDR_BITS)  # addr_a bits (query A)
        self.MAC_B_QRY_BIN = self._band("MAC_B_QRY_BIN", ADDR_BITS)  # addr_b bits (query B)
        self.MAC_A_ADDR_NIB = self._band("MAC_A_ADDR_NIB", 8)  # addr_a nibbles (frame)
        self.MAC_B_ADDR_NIB = self._band("MAC_B_ADDR_NIB", 8)  # addr_b nibbles (frame)
        self.MAC_ACC = self._band("MAC_ACC", 8)          # incoming accumulator nibbles
        self.MAC_PA = self._band("MAC_PA", 4)            # product bytes (MUL_RES recomposed)
        self.MAC_PB = self._band("MAC_PB", 4)            # accumulator bytes (MAC_ACC recomposed)
        self.MAC_SUM_C = self._band("MAC_SUM_C", 5)      # accumulate carry chain
        self.MAC_SUM_RES = self._band("MAC_SUM_RES", 8)  # 32-bit fused result nibbles
        while self._off % n_heads != 0:
            self._scalar(f"_mopad{self._off}")
        self.D = self._off

    def op_is_mac(self) -> int:
        """The physical residual dim that carries OP_IS[MAC] (out-of-band lane)."""
        return self.MAC_OP_IS


# ===========================================================================
# MAC opcode decode: OP_IS[MAC] (the MAC_OP_IS lane) = (OP_VAL == MAC), the same
# triangular-pulse decode.  Runs alongside the base opcode-decode block.
# ===========================================================================
def _mac_opcode_decode(L: MemOperandLayout, dim: int) -> Dict[str, torch.Tensor]:
    op_is_mac = L.op_is_mac()
    thr = [MAC - 1, MAC, MAC + 1]
    spec = _empty_spec(dim, len(thr) + 1)
    tu = {}
    for k, t in enumerate(thr):
        spec["W_up"][k, L.OP_VAL] = RELU_S
        spec["b_up"][k] = -RELU_S * t
        spec["W_gate"][k, L.ONE] = 1.0
        tu[t] = k
    cu = len(thr)                                    # self-clear the lane (SET)
    spec["W_up"][cu, L.ONE] = S
    spec["W_gate"][cu, op_is_mac] = 1.0
    spec["W_down"][op_is_mac, cu] += -1.0 / SILU_S
    spec["W_down"][op_is_mac, tu[MAC - 1]] += 1.0 / RELU_S
    spec["W_down"][op_is_mac, tu[MAC]] += -2.0 / RELU_S
    spec["W_down"][op_is_mac, tu[MAC + 1]] += 1.0 / RELU_S
    return spec


def _mac_flag(L: MemOperandLayout, dim: int) -> Dict[str, torch.Tensor]:
    """MAC_IS := a THRESHOLDED indicator of OP_IS[MAC] (the CAM-enable flag),
    residue-immune (a large-PC decode residue snaps to a clean 0/1 before it
    reaches the CAM role-gate).  Mirrors ``_flag_from_ops`` but reads the
    out-of-band MAC_OP_IS lane."""
    op_is_mac = L.op_is_mac()
    spec = _empty_spec(dim, 3)
    # self-clear MAC_IS (SET).
    spec["W_up"][0, L.ONE] = S; spec["W_gate"][0, L.MAC_IS] = 1.0
    spec["W_down"][L.MAC_IS, 0] += -1.0 / SILU_S
    RAMP = 1.0 / (0.2 * RELU_S)
    for k, (lo, sign) in enumerate(((0.4, +1.0), (0.6, -1.0)), start=1):
        spec["W_up"][k, op_is_mac] = RELU_S
        spec["b_up"][k] = -RELU_S * lo
        spec["b_gate"][k] = 1.0
        spec["W_down"][L.MAC_IS, k] += sign * RAMP
    return spec


# ===========================================================================
# MAC-prep (pre-CAM): expand the two frame addresses -> the two query bins, clear
# STACK0 + AX so each CAM head write is a clean SET.  MAC-gated.
# ===========================================================================
def _mac_prep(L: MemOperandLayout, dim: int) -> Dict[str, torch.Tensor]:
    specs = []
    # expand both operand addresses (frame nibbles) -> their 32-bit query bins.
    specs.append(PFC.compile_nibble_addr_expand(L, L.MAC_A_ADDR_NIB, L.MAC_A_QRY_BIN,
                                                 dim, n_nibbles=8))
    specs.append(PFC.compile_nibble_addr_expand(L, L.MAC_B_ADDR_NIB, L.MAC_B_QRY_BIN,
                                                 dim, n_nibbles=8))
    # clear STACK0 (operand-A dest) + AX (operand-B dest), MAC-gated.
    specs.append(_clear_band_gated_mac(L, L.STACK0, 8, dim))
    specs.append(_clear_band_gated_mac(L, L.AX, 8, dim))
    return _concat_specs(specs, dim)


def _clear_band_gated_mac(L, base, n, dim) -> Dict[str, torch.Tensor]:
    """Clear band dims base..base+n-1 when MAC is active (SET -old).  Reads the
    out-of-band MAC_OP_IS lane."""
    g = L.op_is_mac()
    spec = _empty_spec(dim, n)
    u = 0
    for j in range(n):
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, base + j] = 1.0
        spec["W_down"][base + j, u] += -1.0 / SILU_HALF
        u += 1
    return spec


def _mac_acc_snapshot(L: MemOperandLayout, dim: int) -> Dict[str, torch.Tensor]:
    """On MAC: MAC_ACC nibbles := AX nibbles (the incoming accumulator), BEFORE the
    MAC-prep clears AX / the CAM overwrites it.  Runs right after ingest.  Gated on
    MAC (SET; 0 on non-MAC ops -> byte-identical)."""
    g = L.op_is_mac()
    spec = _empty_spec(dim, 8 * 2)
    u = 0
    for j in range(8):
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.MAC_ACC + j] = 1.0
        spec["W_down"][L.MAC_ACC + j, u] += -1.0 / SILU_HALF; u += 1
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.AX + j] = 1.0
        spec["W_down"][L.MAC_ACC + j, u] += 1.0 / SILU_HALF; u += 1
    return spec


# ===========================================================================
# The two MAC CAM heads (identical §Memory CAM to LI/SI), keyed on the
# frame-carried operand addresses, enabled by MAC_IS, values -> STACK0 / AX.
# ===========================================================================
def _bake_mac_head(attn, L: MemOperandLayout, head: int, qry_bin: int, dst_base: int) -> None:
    from .blogspec_memory import EFF, BIAS, MEM_ALIBI_SLOPE, PEN_GATE
    hs = attn.scale
    smag = (EFF / hs) ** 0.5
    qb = (BIAS / hs) ** 0.5
    kb = (BIAS / hs) ** 0.5
    PEN = PEN_GATE
    p = (PEN / hs) ** 0.5
    attn.alibi_slopes[head] = MEM_ALIBI_SLOPE
    HD = attn.head_dim
    base = head * HD
    for b in range(ADDR_BITS):
        attn.W_k[base + b, L.ADDR_BIN + b] = 2.0 * smag
        attn.W_k[base + b, L.ONE] = -smag
        attn.W_q[base + b, qry_bin + b] = 2.0 * smag       # QUERY = operand address
        attn.W_q[base + b, L.ONE] = -smag
    cB = base + ADDR_BITS
    attn.W_q[cB, L.MAC_IS] = -qb           # ZFOD bias enabled by MAC_IS
    attn.W_k[cB, L.IS_STORE] = kb
    cR = base + ADDR_BITS + 1
    attn.W_q[cR, L.MAC_IS] = p              # store-role penalty
    attn.W_k[cR, L.ONE] = -p
    attn.W_k[cR, L.IS_STORE] = p
    cL = base + ADDR_BITS + 2               # LOAD-enable: non-MAC query -> softmax1 sink
    c = (PEN / hs) ** 0.5
    attn.W_q[cL, L.ONE] = c
    attn.W_q[cL, L.MAC_IS] = -c
    attn.W_k[cL, L.ONE] = -c
    for j in range(NIB_PER_REG):
        attn.W_v[base + ADDR_BITS + 3 + j, L.VAL_NIB + j] = 1.0
        attn.W_o[dst_base + j, base + ADDR_BITS + 3 + j] = 1.0


# ===========================================================================
# MAC-accumulate: AX = MUL_RES + MAC_ACC (fused accumulate), a 32-bit per-byte
# add carry chain into a PRIVATE band (never disturbs the shared ALU add lanes),
# then SET AX := MAC_SUM_RES, MAC-gated.
# ===========================================================================
def _mac_accumulate_blocks(L: MemOperandLayout, dim: int) -> List[Tuple[str, Dict]]:
    a = L.ALU32
    blocks: List[Tuple[str, Dict]] = []

    # (1) recompose: MAC_PA byte i = MUL_RES nib(2i)+16*nib(2i+1);
    #                MAC_PB byte i = MAC_ACC nib(2i)+16*nib(2i+1).
    A._ONE = L.ONE
    spec = _empty_spec(dim, 4 * 4)
    u = 0
    for i in range(4):
        u = A._clear(spec, u, L.MAC_PA + i)
        u = A._ident(spec, u, {a.MUL_RES + 2 * i: 1.0, a.MUL_RES + 2 * i + 1: 16.0},
                     0.0, L.MAC_PA + i, 1.0)
        u = A._clear(spec, u, L.MAC_PB + i)
        u = A._ident(spec, u, {L.MAC_ACC + 2 * i: 1.0, L.MAC_ACC + 2 * i + 1: 16.0},
                     0.0, L.MAC_PB + i, 1.0)
    blocks.append(("mac-acc-expand", A._truncate(spec, u, dim)))

    # (2..5) the per-byte add carry chain: MAC_SUM_RES = MAC_PA + MAC_PB.
    for i in range(4):
        blk = A._byte_add_block(L, dim, L.MAC_PA, L.MAC_PB, L.MAC_SUM_C,
                                L.MAC_SUM_RES, i, cin_const=0.0)
        blocks.append((f"mac-acc-add-b{i}", blk))

    # (6) AX nibbles := MAC_SUM_RES, MAC-gated (SET).  MUL's ax-mux does NOT fire for
    # MAC (MAC not in ALU_OPS), so AX is clean to receive the fused result.
    g = L.op_is_mac()
    spec = _empty_spec(dim, 8 * 2)
    u = 0
    for j in range(8):
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.AX + j] = 1.0
        spec["W_down"][L.AX + j, u] += -1.0 / SILU_HALF; u += 1
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.MAC_SUM_RES + j] = 1.0
        spec["W_down"][L.AX + j, u] += 1.0 / SILU_HALF; u += 1
    blocks.append(("mac-acc-writeback", spec))
    return blocks


# ===========================================================================
# BUILD: the complete pure-forward model + the MAC opcode.  Reuses the complete
# block list verbatim, inserting the MAC blocks at the right seams and baking the
# two MAC CAM heads onto the mem-cam block (block 7).
# ===========================================================================
def build_mem_operand_model(code_size: int = 32):
    """Build the complete pure-forward VM extended with the fused memory-operand
    MAC opcode.  Returns ``(model, L)``.  Requires C4_MEM_OPERAND (the caller
    should have set it); the layout / blocks / heads are all MAC-additive so every
    non-MAC opcode is byte-identical to ``build_pure_forward_complete_model``."""
    from .blogspec_model import Transformer
    from .nibble_vm import (
        compile_nibble_to_scalar, compile_pc_fetch, compile_code_select, compile_ffn,
    )
    from .nibble_pure_forward import compile_mem_prep, memory_dispatch_rules
    from .dsl import FFNRule, LinearExpr

    # +2 heads for the two MAC operand-read CAM heads (on top of the complete build's
    # 20 ingest + LI + stack-pop + lev = 23).
    n_heads = N_ROLES + 5
    L = MemOperandLayout(code_size, n_heads=n_heads)
    A.extend_layout_for_alu32(L, recurrent_divmod=False)
    from . import nibble_bitwise as _bw
    _bw.extend_layout_for_bitwise(L)
    if _bw.tight_shift_enabled():
        for _op in (isa.SHL, isa.SHR):
            _bw.extend_layout_for_tight_shift(L, _op)
    while L._off % n_heads != 0:
        L._scalar(f"_bwpad{L._off}")
    L.D = L._off
    min_dim = n_heads * MEM_HEAD_CHANNELS
    if L.D < min_dim:
        target = -(-min_dim // n_heads) * n_heads
        while L._off < target:
            L._scalar(f"_hdpad{L._off}")
        L.D = L._off
    dim = L.D
    A._ONE = L.ONE

    block_specs: List[Tuple[str, Dict]] = [
        ("ingest+recompose", compile_nibble_to_scalar(L, dim)),
        ("pc-fetch",    compile_pc_fetch(L, dim)),
        ("code-select", compile_code_select(L, dim)),
        ("opcode-decode", PFC.compile_opcode_decode_pfc(L, dim)),
        ("mac-decode",  _mac_opcode_decode(L, dim)),          # OP_IS[MAC] one-hot
        ("mac-flag",    _mac_flag(L, dim)),                   # MAC_IS thresholded
        # snapshot the incoming accumulator (AX) into MAC_ACC AFTER the opcode is
        # decoded (so the MAC gate is live) but BEFORE mac-prep clears AX.
        ("mac-acc-snap", _mac_acc_snapshot(L, dim)),
        ("imm-nib-fetch", PFC.compile_imm_nib_fetch(L, dim)),
        ("imm-clean", PFC.compile_imm_clean(L, dim)),
        ("mem-prep", compile_mem_prep(L, dim)),
        ("mac-prep", _mac_prep(L, dim)),                      # MAC query bins + clears
        ("mem-cam",  compile_nibble_to_scalar(L, dim)),       # ATTN = LI head + 2 MAC heads
        ("pop-addr", PFC.compile_pop_addr(L, dim)),
        ("stack-prep", PFC.compile_stack_prep(L, dim)),
        ("lev-addr4", PFC._force_bit2(L, L.LEV_QRY_BIN, dim)),
        ("stack-pop-cam", PFC.compile_stk_recompose(L, dim)),
        ("cmp-compute", PFC.compile_cmp_compute(L, dim)),
        ("cmp-finalize", PFC.compile_cmp_signed_finalize(L, dim)),
        ("alu-expand", A.compile_expand(L, dim)),
    ]
    for name, spec in A.compile_addsub_blocks(L, dim):
        block_specs.append((name, spec))
    for name, spec in A.compile_mul_blocks(L, dim):
        block_specs.append((name, spec))
    for name, spec in A.compile_divmod_blocks(L, dim):
        block_specs.append((name, spec))
    block_specs.append(("ax-mux", A.compile_ax_mux(L, dim, ops=PFC.ALU_OPS)))
    # the MAC fused-accumulate blocks run AFTER MUL wrote MUL_RES and AFTER ax-mux
    # (which does not fire for MAC), so AX is clean to receive MUL_RES + MAC_ACC.
    for name, spec in _mac_accumulate_blocks(L, dim):
        block_specs.append((name, spec))
    from .nibble_unified import build_bitwise_blocks, _bw_recompose_spec
    for name, spec in build_bitwise_blocks(L, dim):
        block_specs.append((name, spec))
    block_specs.append(("bw-recompose", _bw_recompose_spec(
        L, dim, (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR))))

    disp_rules = PFC._base_rules_minus_alu(L)
    disp_rules += memory_dispatch_rules(L)
    disp_rules += PFC.cmp_dispatch_rules_pop(L)
    disp_rules += PFC.callconv_dispatch_rules(L)
    disp_rules += PFC.alu32_housekeeping_rules(L)
    disp_rules += PFC._bitwise_pop_rules(L)
    # MAC dispatch: PC += 1 (straight-line).  Gated on the out-of-band OP_IS[MAC] lane.
    disp_rules.append(FFNRule([(L.op_is_mac(), 0.5, 1.5)], {L.PC_VAL: LinearExpr.c(1.0)}))

    byte_ax_ops = [isa.IMM, isa.LEA] + \
                  [isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE] + \
                  [isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR]
    block_specs += [
        ("dispatch", compile_ffn(disp_rules, dim)),
        ("branch-delta", PFC.compile_branch_delta_clean(L, dim)),
        ("fold-lea", PFC._fold_ax_gated(L, dim, [isa.LEA])),
        ("ax-nib-split", PFC.compile_ax_nib_split(L, dim)),
        ("lea-q-reduce", PFC.compile_lea_q_reduce(L, dim)),
    ]
    if PFC._lea_q_snap_enabled():
        block_specs += [("lea-q-snap", PFC.compile_lea_q_snap(L, dim))]
    block_specs += [
        ("lea-addr-nib", PFC.compile_lea_addr_nib(L, dim)),
        ("ax-byte-nib", PFC.compile_ax_byte_to_nibbles(L, dim, byte_ax_ops)),
        ("imm-ax-nib", PFC.compile_imm_ax_nibbles(L, dim)),
    ]
    n_blocks = len(block_specs)
    L._apply_order = None
    hidden = max(f["W_up"].shape[0] for _, f in block_specs)
    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, vocab=V.VOCAB, max_seq_len=16384)
    with torch.no_grad():
        PFC._bake_pure_embedding(model, L)
        for bi, (name, spec) in enumerate(block_specs):
            _zero_attn(model.blocks[bi].attn)
            _load_ffn(model.blocks[bi].ffn, spec, hidden)
        reg_bases = {"PC": L.PC, "AX": L.AX, "SP": L.SP, "BP": L.BP, "STACK0": L.STACK0}
        bake_frame_ingest(model.blocks[0].attn, L, reg_bases)
        mem_block = PFC._find(block_specs, "mem-cam")
        _bake_pf_memory_head(model.blocks[mem_block].attn, L, head=N_ROLES)
        stk_block = PFC._find(block_specs, "stack-pop-cam")
        PFC._bake_stack_pop_head(model.blocks[stk_block].attn, L, head=N_ROLES + 1)
        PFC._bake_lev_ret_head(model.blocks[stk_block].attn, L, head=N_ROLES + 2)
        # the two MAC operand-read CAM heads share the mem-cam block (block 7): read
        # mem[addr_a] -> STACK0 (operand A) and mem[addr_b] -> AX (operand B).
        _bake_mac_head(model.blocks[mem_block].attn, L, head=N_ROLES + 3,
                       qry_bin=L.MAC_A_QRY_BIN, dst_base=L.STACK0)
        _bake_mac_head(model.blocks[mem_block].attn, L, head=N_ROLES + 4,
                       qry_bin=L.MAC_B_QRY_BIN, dst_base=L.AX)
    L._block_names = [n for n, _ in block_specs]
    return model, L


# ===========================================================================
# The MAC-aware overlay: identical to the complete overlay, but also writes the
# two operand-address nibble bands on the frame's slot rows for a MAC step.  We
# reuse make_overlay_complete and add the MAC address nibbles on top.
# ===========================================================================
def make_overlay_mem_operand(code, L: MemOperandLayout, store_log=None, mac_addrs=None):
    """``mac_addrs`` maps ``frame_idx -> (addr_a, addr_b)`` for each emitted MAC
    step's frame, so the model's query heads read the operand addresses off the
    frame.  All other behaviour is the complete overlay."""
    base_overlay = PFC.make_overlay_complete(code, L, store_log=store_log)
    mac_addrs = mac_addrs or {}

    def overlay(x: torch.Tensor) -> None:
        base_overlay(x)
        # write the operand-address nibbles onto EVERY row's MAC address bands so
        # the ingest carries them to the query row (they are static per-step, like
        # the immediate).  We put them on the query row (last position) which the
        # ingest reads; simplest is to write them on all rows of the current frame.
        Sn = x.shape[1]
        # the CURRENT step's addresses are on the query row (the last position); the
        # driver passes them via ``mac_addrs`` keyed by the CURRENT frame index, but
        # the query row is always the last row, so write there.
        cur = mac_addrs.get("cur")
        if cur is not None:
            addr_a, addr_b = cur
            for j, nv in enumerate(V.nibbles_of_value(addr_a & 0xFFFFFFFF, 8)):
                x[0, -1, L.MAC_A_ADDR_NIB + j] = float(nv)
            for j, nv in enumerate(V.nibbles_of_value(addr_b & 0xFFFFFFFF, 8)):
                x[0, -1, L.MAC_B_ADDR_NIB + j] = float(nv)
    return overlay


# ===========================================================================
# The MAC-aware reference interpreter (numpy fixed-point oracle).  Same
# SP-addressed memory-stack semantics as ``PFC.ref_interpret``, plus the fused
# MAC opcode: acc = acc + mem[addr_a] * mem[addr_b], acc carried in AX.  ``mac_b``
# maps PC -> addr_b (the second operand address; addr_a rides Instr.imm).
# ===========================================================================
def ref_interpret_mac(code, mac_b: Dict[int, int], max_steps: int = 512,
                      mask: int = 0xFF, seed_mem: Optional[Dict[int, int]] = None):
    mem: Dict[int, int] = dict(seed_mem or {})
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
        if op == MAC:
            a = mem.get(imm & 0xFFFFFFFF, 0) & mask
            b = mem.get(mac_b.get(i, 0) & 0xFFFFFFFF, 0) & mask
            ax = (ax + a * b) & mask
        elif op == isa.IMM:
            ax = imm & mask
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
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0) & mask
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4
            mem[addr] = ax & mask
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            trace.append(ax & mask); break
        else:
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in MAC ref ISA")
        trace.append(ax & mask)
    return trace


# ===========================================================================
# The MAC-aware driver: one VM step = one model.forward.  Identical to
# ``run_pure_forward_complete`` except it also feeds the current MAC step's two
# operand addresses to the overlay, and treats MAC as a straight-line op (PC+=1,
# AX = the model-decoded fused result).
# ===========================================================================
def run_mem_operand(model, L: MemOperandLayout, code, mac_b: Dict[int, int],
                    max_steps: int = 512, mask: int = 0xFF,
                    seed_mem: Optional[Dict[int, int]] = None, verbose: bool = False):
    """Run ``code`` (which may contain MAC opcodes) through the mem-operand model.
    ``mac_b`` maps PC -> the second operand address for each MAC; the first rides
    ``Instr.imm``.  ``seed_mem`` seeds the KV memory before step 0.  Returns the
    per-step AX trace (matching ``ref_interpret_mac``)."""
    from .nibble_pure_forward import _snap_lane
    seed_frames, store_log = PFC._seed_frames(seed_mem or {})
    n_seed = len(store_log)
    stream: List[int] = [V.BOS] + seed_frames + PFC._build_frame(0, 0, SP_INIT, SP_INIT, 0)
    trace: List[int] = []
    cur_pc = 0
    cur_sp = cur_bp = SP_INIT
    cur_ax = 0
    frame_idx = n_seed
    for _ in range(max_steps):
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        imm = code[cur_pc].imm if 0 <= cur_pc < len(code) else 0
        mac_addrs = {}
        if op == MAC:
            mac_addrs = {"cur": (imm & 0xFFFFFFFF, mac_b.get(cur_pc, 0) & 0xFFFFFFFF)}
        overlay = make_overlay_mem_operand(code, L, store_log=store_log, mac_addrs=mac_addrs)
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
        ax = PFC._decode_reg_from_nibbles(state, L, L.AX)
        # store bookkeeping (same store contract as the complete driver).
        s_addr = s_val = 0
        is_store = False
        if op in (isa.SI, isa.SC):
            is_store = True; s_addr = PFC._mem_top(store_log, cur_sp); s_val = ax & mask
        elif op == isa.PSH:
            is_store = True; s_addr = cur_sp - 4; s_val = ax & mask
        frame = PFC._build_frame(pc, ax, sp, bp, stk,
                                 mem_addr=(s_addr if is_store else 0),
                                 mem_val=(s_val if is_store else 0))
        trace.append(ax & mask)
        frame_idx += 1
        if is_store:
            store_log[frame_idx] = (s_addr, s_val)
        stream += frame
        if verbose:
            nm = isa.NAMES.get(op, "MAC" if op == MAC else str(op))
            print(f"  step pc={cur_pc} op={nm:4s} -> "
                  f"pc'={pc} ax={ax & mask} sp={sp} halt={halted}")
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    return trace
