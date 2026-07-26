"""nibble_exact_steps.py — the MEMORY-OPERAND ALU family (C4_EXACT_STEPS).

What this extends (and why)
===========================
``nibble_mem_operand`` proved the accfa388 fold for ONE fused op: ``MAC [a],[b]``
reads two addressed operands in the EARLY blocks of its OWN forward (the §Memory
CAM runs at block 7) and does the multiply-accumulate in the LATE blocks (MUL at
block 23), so the two loads + the multiply + the accumulate collapse into a SINGLE
``model.forward`` — no separate ``IMM addr; LI`` load step per operand.

This module GENERALISES that fold to the rest of the addressed-operand ALU ops.
An addressed binary ALU op in the interpreted VM costs FOUR forwards:

    IMM addr ; LI ; PSH ; <OP>          #  AX = mem[addr] <op> AX   (interpreted)

— the ``IMM addr`` puts the address in AX (1 forward), ``LI`` reads mem into AX (1),
``PSH`` stages it as the ALU's popped operand (1), and ``<OP>`` finally computes.
The memory-operand form does it in ONE:

    <OP>M [addr]                        #  AX = mem[addr] <op> AX   (1 forward)

The mechanism reuses EVERYTHING the MAC already built:
  * the same §Memory CAM read at block 7 (``_bake_mac_head``), keyed on the
    FRAME-carried operand address, value -> ``STACK0`` (the ALU's operand-A);
  * the EXISTING unconditional 32-bit ALU blocks, which every step compute
    ``A <op> B`` with ``A = STACK0``, ``B = AX`` into each op's OWN result band
    (``ADD_RES``/``SUB_RES``/``MUL_RES``/``DIV_RES``/``MOD_RES``) — so once the CAM
    put ``mem[addr]`` in STACK0 and AX still holds the accumulator, EVERY result
    band already holds ``mem[addr] <op> AX``, for free;
  * a gated write-back that copies the selected op's result band into AX;
  * a straight-line ``PC += 1`` dispatch (a memory-operand op does NOT pop the
    stack — there is no PSH — so SP is untouched).

Semantics (operand order): ``<OP>M [addr]`` computes ``mem[addr] <op> AX`` — the
SAME operand order as the base ISA's ``PSH v; <OP>`` (which is ``v <op> AX`` =
popped-operand ``<op>`` AX).  For ADD/MUL this equals ``AX <op> mem[addr]``; for
SUB/DIV/MOD it is the non-commutative ``mem[addr] - AX`` etc., exactly matching a
``PSH mem[addr]; SUB`` in the base ISA.

Scope / honest limits
=====================
The folded family is the FIVE arithmetic ops ADD/SUB/MUL/DIV/MOD (opcodes 41-45),
which leave their result in a NEUTRAL per-op band the write-back can select.  The
bitwise ops (AND/OR/XOR) write their result DIRECTLY into AX gated on the base
``OP_IS[op]`` (no neutral result band), so a clean memory-operand fold for them
would need a NEW ungated bitwise-result band; that is deliberately out of scope
here (it would touch the bitwise bake a sibling is editing).  See the module
report for the full fold/no-fold verdict.

Gating
======
Everything is behind ``C4_EXACT_STEPS`` (default OFF via ``exact_steps_enabled``).
The build path is a SUPERSET of the ``C4_MEM_OPERAND`` build (it turns MAC on too),
so with ``C4_EXACT_STEPS`` OFF this module is never imported and the golden is
byte-identical.  With it ON, the family is additive and per-op-gated: every band is
0 on a non-memory-operand step, and the five extra §Memory CAM reads sink to the
softmax1 zero-vector when their op is not active (exactly like the MAC head), so
every OTHER opcode is byte-identical to the mem-operand build too.
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
from . import nibble_pure_forward_complete as PFC
from . import nibble_mem_operand as MO
from .nibble_pure_forward import (
    N_ROLES, SP_INIT, MEM_HEAD_CHANNELS, bake_frame_ingest, _bake_pf_memory_head,
)

# The memory-operand ALU opcodes.  MAC (nibble_mem_operand) is 40; these are the
# next five free values.  Each decodes into its OWN out-of-band OP_IS lane (like
# MAC_OP_IS), so the base opcode band 0..39 is untouched.
ADDM, SUBM, MULM, DIVM, MODM = 41, 42, 43, 44, 45

# op -> (base ALU op it reuses, result-band attribute on L.ALU32)
_MOP_ALU = {
    ADDM: (isa.ADD, "ADD_RES"),
    SUBM: (isa.SUB, "SUB_RES"),
    MULM: (isa.MUL, "MUL_RES"),
    DIVM: (isa.DIV, "DIV_RES"),
    MODM: (isa.MOD, "MOD_RES"),
}
MOPS = list(_MOP_ALU.keys())

NAMES = {ADDM: "ADDM", SUBM: "SUBM", MULM: "MULM", DIVM: "DIVM", MODM: "MODM"}


def exact_steps_enabled() -> bool:
    """The memory-operand ALU family (C4_EXACT_STEPS).  DEFAULT OFF.

    OFF -> never on the build path; the complete-VM model + golden are
    byte-identical.  ON -> the five ``<OP>M [addr]`` opcodes + their early CAM read
    + the gated result write-back are added (all per-op-gated, additive)."""
    return os.environ.get("C4_EXACT_STEPS", "0") != "0"


# ===========================================================================
# LAYOUT: the mem-operand layout + the memory-operand-ALU scratch bands.  One
# out-of-band OP_IS lane per <OP>M op, one shared CAM-enable flag, one operand
# query-bit band + its frame address nibbles.  All appended AFTER the mem-operand
# + ALU + bitwise bands so every prior offset is preserved.
# ===========================================================================
class ExactStepsLayout(MO.MemOperandLayout):
    def __init__(self, code_size: int, n_heads: int):
        super().__init__(code_size, n_heads=n_heads)
        # NOTE: the ALU / bitwise layout extensions happen in the BUILD (after this
        # ctor), so we allocate our bands from a hook the build calls post-ALU.  Keep
        # the ctor identical to MemOperandLayout; ``extend_for_exact_steps`` appends.

    def extend_for_exact_steps(self) -> None:
        """Allocate the memory-operand-ALU bands.  Called by the build AFTER the ALU
        + bitwise layout extensions so the ALU result bands already exist."""
        self.MOP_OP_IS = {op: self._scalar(f"MOP_OP_IS_{NAMES[op]}") for op in MOPS}
        self.MOP_IS = self._scalar("MOP_IS")               # shared CAM-enable flag
        self.MOP_QRY_BIN = self._band("MOP_QRY_BIN", ADDR_BITS)  # operand addr bits
        self.MOP_ADDR_NIB = self._band("MOP_ADDR_NIB", 8)  # operand addr nibbles (frame)

    def mop_op_is(self, op: int) -> int:
        return self.MOP_OP_IS[op]


# ===========================================================================
# DECODE: OP_IS[<OP>M] one-hot into the out-of-band lane, per op.  Same triangular
# pulse as the base opcode decode / the MAC decode.
# ===========================================================================
def _mop_opcode_decode(L: ExactStepsLayout, dim: int) -> Dict[str, torch.Tensor]:
    # one 4-unit decode (3 thresholds + a self-clear) per op, concatenated.
    n = len(MOPS)
    spec = _empty_spec(dim, n * 4)
    u = 0
    for op in MOPS:
        lane = L.mop_op_is(op)
        thr = [op - 1, op, op + 1]
        base_u = u
        tu = {}
        for t in thr:
            spec["W_up"][u, L.OP_VAL] = RELU_S
            spec["b_up"][u] = -RELU_S * t
            spec["W_gate"][u, L.ONE] = 1.0
            tu[t] = u
            u += 1
        # self-clear the lane (SET), then the triangular pulse.
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, lane] = 1.0
        spec["W_down"][lane, u] += -1.0 / SILU_S
        spec["W_down"][lane, tu[op - 1]] += 1.0 / RELU_S
        spec["W_down"][lane, tu[op]] += -2.0 / RELU_S
        spec["W_down"][lane, tu[op + 1]] += 1.0 / RELU_S
        u += 1
    return spec


def _mop_flag(L: ExactStepsLayout, dim: int) -> Dict[str, torch.Tensor]:
    """MOP_IS := a THRESHOLDED OR of all the <OP>M decode lanes (the CAM-enable flag
    for the shared operand-read head).  Residue-immune: a large-PC decode residue on
    any lane snaps to a clean 0/1 before it reaches the CAM role-gate.  Mirrors the
    MAC ``_mac_flag`` but ORs over the five lanes."""
    spec = _empty_spec(dim, 1 + 2 * len(MOPS))
    u = 0
    # self-clear MOP_IS (SET).
    spec["W_up"][u, L.ONE] = S
    spec["W_gate"][u, L.MOP_IS] = 1.0
    spec["W_down"][L.MOP_IS, u] += -1.0 / SILU_S
    u += 1
    RAMP = 1.0 / (0.2 * RELU_S)
    for op in MOPS:
        lane = L.mop_op_is(op)
        # a clamped step(lane >= 0.5): relu(lane-0.4) - relu(lane-0.6), each op adds
        # 0/1; since at most one lane is 1 at a time the OR is just the sum.
        for lo, sign in ((0.4, +1.0), (0.6, -1.0)):
            spec["W_up"][u, lane] = RELU_S
            spec["b_up"][u] = -RELU_S * lo
            spec["b_gate"][u] = 1.0
            spec["W_down"][L.MOP_IS, u] += sign * RAMP
            u += 1
    return spec


# ===========================================================================
# PREP (pre-CAM): expand the operand address (frame nibbles) -> its query bin, and
# clear STACK0 (the CAM-read destination = ALU operand-A) so the CAM write is a
# clean SET.  Gated on MOP_IS.  AX is LEFT ALONE (it holds the ALU operand-B).
# ===========================================================================
def _mop_prep(L: ExactStepsLayout, dim: int) -> Dict[str, torch.Tensor]:
    from .nibble_pure_forward import _concat_specs
    specs = []
    specs.append(PFC.compile_nibble_addr_expand(L, L.MOP_ADDR_NIB, L.MOP_QRY_BIN,
                                                 dim, n_nibbles=8))
    specs.append(_clear_band_gated_mop(L, L.STACK0, 8, dim))
    return _concat_specs(specs, dim)


def _clear_band_gated_mop(L, base, n, dim) -> Dict[str, torch.Tensor]:
    """Clear band dims base..base+n-1 when a <OP>M op is active (SET -old).  Gated on
    the shared MOP_IS thresholded flag (1 iff any <OP>M lane is live)."""
    g = L.MOP_IS
    spec = _empty_spec(dim, n)
    u = 0
    for j in range(n):
        spec["W_up"][u, g] = S
        spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, base + j] = 1.0
        spec["W_down"][base + j, u] += -1.0 / SILU_HALF
        u += 1
    return spec


# ===========================================================================
# WRITE-BACK: AX_nib[c] := <OP>_RES[c] for the active <OP>M op (SET), gated on the
# out-of-band MOP_OP_IS lane.  The base ax-mux (compile_ax_mux) does NOT fire for
# these ops (they are not in ALU_OPS), so AX is clean to receive the fused result.
# ===========================================================================
def _mop_writeback(L: ExactStepsLayout, dim: int) -> Dict[str, torch.Tensor]:
    a = L.ALU32
    spec = _empty_spec(dim, 8 * 2 * len(MOPS))
    u = 0
    for op in MOPS:
        g = L.mop_op_is(op)
        band = getattr(a, _MOP_ALU[op][1])
        for c in range(8):
            # clear old AX nibble c (gated), then add RES[c] (gated).
            spec["W_up"][u, g] = S
            spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, L.AX + c] = 1.0
            spec["W_down"][L.AX + c, u] += -1.0 / SILU_HALF
            u += 1
            spec["W_up"][u, g] = S
            spec["b_up"][u] = -S * 0.5
            spec["W_gate"][u, band + c] = 1.0
            spec["W_down"][L.AX + c, u] += 1.0 / SILU_HALF
            u += 1
    return spec


# ===========================================================================
# BUILD: the mem-operand model (which already has MAC) + the memory-operand-ALU
# family.  Mirrors ``build_mem_operand_model`` but registers the extra decode /
# flag / prep / write-back blocks and one extra CAM head reading into STACK0.
# ===========================================================================
def build_exact_steps_model(code_size: int = 32):
    """Build the complete pure-forward VM + MAC + the memory-operand ALU family.
    Returns ``(model, L)``.  Requires C4_EXACT_STEPS (which also implies
    C4_MEM_OPERAND).  All blocks / heads are additive + per-op-gated, so every
    non-memory-operand opcode is byte-identical to the mem-operand build."""
    # ensure the mem-operand machinery is enabled (this build is a superset).
    os.environ.setdefault("C4_MEM_OPERAND", "1")
    from .blogspec_model import Transformer
    from .nibble_vm import (
        compile_nibble_to_scalar, compile_pc_fetch, compile_code_select, compile_ffn,
    )
    from .nibble_pure_forward import compile_mem_prep, memory_dispatch_rules
    from .dsl import FFNRule, LinearExpr

    # heads: complete build's 23 + 2 MAC heads (N_ROLES+3, +4) + 1 memory-operand-ALU
    # operand-read head (N_ROLES+5).  N_ROLES + 6.
    n_heads = N_ROLES + 6
    L = ExactStepsLayout(code_size, n_heads=n_heads)
    A.extend_layout_for_alu32(L, recurrent_divmod=False)
    from . import nibble_bitwise as _bw
    _bw.extend_layout_for_bitwise(L)
    if _bw.tight_shift_enabled():
        for _op in (isa.SHL, isa.SHR):
            _bw.extend_layout_for_tight_shift(L, _op)
    # our memory-operand-ALU bands (after the ALU result bands exist).
    L.extend_for_exact_steps()
    while L._off % n_heads != 0:
        L._scalar(f"_espad{L._off}")
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
        ("mac-decode",  MO._mac_opcode_decode(L, dim)),
        ("mac-flag",    MO._mac_flag(L, dim)),
        ("mop-decode",  _mop_opcode_decode(L, dim)),       # <OP>M one-hots
        ("mop-flag",    _mop_flag(L, dim)),                # MOP_IS thresholded
        ("mac-acc-snap", MO._mac_acc_snapshot(L, dim)),
        ("imm-nib-fetch", PFC.compile_imm_nib_fetch(L, dim)),
        ("imm-clean", PFC.compile_imm_clean(L, dim)),
        ("mem-prep", compile_mem_prep(L, dim)),
        ("mac-prep", MO._mac_prep(L, dim)),
        ("mop-prep", _mop_prep(L, dim)),                   # <OP>M query bin + STACK0 clear
        ("mem-cam",  compile_nibble_to_scalar(L, dim)),    # ATTN = LI + 2 MAC + 1 MOP heads
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
    # the MAC fused-accumulate blocks (unchanged from the mem-operand build).
    for name, spec in MO._mac_accumulate_blocks(L, dim):
        block_specs.append((name, spec))
    # the memory-operand-ALU write-back: AX := <OP>_RES for the active <OP>M op.  Runs
    # AFTER the ALU wrote every RES band and AFTER ax-mux (which does not fire for
    # <OP>M), so AX is clean to receive mem[addr] <op> AX.
    block_specs.append(("mop-writeback", _mop_writeback(L, dim)))
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
    # MAC dispatch (unchanged): PC += 1.
    disp_rules.append(FFNRule([(L.op_is_mac(), 0.5, 1.5)], {L.PC_VAL: LinearExpr.c(1.0)}))
    # <OP>M dispatch: PC += 1 (straight-line, NO stack pop -> SP untouched).  Gated on
    # each out-of-band MOP_OP_IS lane.
    for op in MOPS:
        disp_rules.append(FFNRule([(L.mop_op_is(op), 0.5, 1.5)],
                                  {L.PC_VAL: LinearExpr.c(1.0)}))

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
        # the two MAC operand-read CAM heads (unchanged).
        MO._bake_mac_head(model.blocks[mem_block].attn, L, head=N_ROLES + 3,
                          qry_bin=L.MAC_A_QRY_BIN, dst_base=L.STACK0)
        MO._bake_mac_head(model.blocks[mem_block].attn, L, head=N_ROLES + 4,
                          qry_bin=L.MAC_B_QRY_BIN, dst_base=L.AX)
        # the memory-operand-ALU operand-read head: mem[addr] -> STACK0 (ALU operand-A),
        # keyed on the frame-carried operand address, enabled by MOP_IS.  Reuses the
        # MAC head bake verbatim (same §Memory CAM), but its enable flag is MOP_IS.
        _bake_mop_head(model.blocks[mem_block].attn, L, head=N_ROLES + 5,
                       qry_bin=L.MOP_QRY_BIN, dst_base=L.STACK0)
    L._block_names = [n for n, _ in block_specs]
    return model, L


def _bake_mop_head(attn, L: ExactStepsLayout, head: int, qry_bin: int,
                   dst_base: int) -> None:
    """The memory-operand-ALU operand-read CAM head.  IDENTICAL to
    ``nibble_mem_operand._bake_mac_head`` except the CAM-enable flag is ``MOP_IS``
    (not MAC_IS): read ``mem[addr]`` (addr from the frame's query bin) -> STACK0 when
    a <OP>M op is active, else sink to the softmax1 zero-vector."""
    from .blogspec_memory import EFF, BIAS, MEM_ALIBI_SLOPE, PEN_GATE
    hs = attn.scale
    smag = (EFF / hs) ** 0.5
    qb = (BIAS / hs) ** 0.5
    kb = (BIAS / hs) ** 0.5
    p = (PEN_GATE / hs) ** 0.5
    attn.alibi_slopes[head] = MEM_ALIBI_SLOPE
    HD = attn.head_dim
    base = head * HD
    for b in range(ADDR_BITS):
        attn.W_k[base + b, L.ADDR_BIN + b] = 2.0 * smag
        attn.W_k[base + b, L.ONE] = -smag
        attn.W_q[base + b, qry_bin + b] = 2.0 * smag
        attn.W_q[base + b, L.ONE] = -smag
    cB = base + ADDR_BITS
    attn.W_q[cB, L.MOP_IS] = -qb              # ZFOD bias enabled by MOP_IS
    attn.W_k[cB, L.IS_STORE] = kb
    cR = base + ADDR_BITS + 1
    attn.W_q[cR, L.MOP_IS] = p                # store-role penalty
    attn.W_k[cR, L.ONE] = -p
    attn.W_k[cR, L.IS_STORE] = p
    cL = base + ADDR_BITS + 2                 # LOAD-enable: non-<OP>M query -> sink
    c = (PEN_GATE / hs) ** 0.5
    attn.W_q[cL, L.ONE] = c
    attn.W_q[cL, L.MOP_IS] = -c
    attn.W_k[cL, L.ONE] = -c
    for j in range(NIB_PER_REG):
        attn.W_v[base + ADDR_BITS + 3 + j, L.VAL_NIB + j] = 1.0
        attn.W_o[dst_base + j, base + ADDR_BITS + 3 + j] = 1.0


# ===========================================================================
# The memory-operand-ALU reference interpreter (numpy fixed-point oracle).  Extends
# the MAC reference with the five <OP>M ops: AX = mem[addr] <op> AX (addr = imm).
# ===========================================================================
def ref_interpret_exact(code, mac_b: Optional[Dict[int, int]] = None,
                        max_steps: int = 512, mask: int = 0xFF,
                        seed_mem: Optional[Dict[int, int]] = None):
    mem: Dict[int, int] = dict(seed_mem or {})
    mac_b = mac_b or {}
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
        if op in _MOP_ALU:
            v = mem.get(imm & 0xFFFFFFFF, 0) & mask      # mem[addr] = operand-A
            base = _MOP_ALU[op][0]
            if base == isa.ADD:
                ax = (v + ax) & mask
            elif base == isa.SUB:
                ax = (v - ax) & mask
            elif base == isa.MUL:
                ax = (v * ax) & mask
            elif base == isa.DIV:
                ax = ((v // ax) if ax else 0) & mask
            else:
                ax = ((v % ax) if ax else 0) & mask
        elif op == MO.MAC:
            a = mem.get(imm & 0xFFFFFFFF, 0) & mask
            b = mem.get(mac_b.get(i, 0) & 0xFFFFFFFF, 0) & mask
            ax = (ax + a * b) & mask
        elif op == isa.IMM:
            ax = imm & mask
        elif op == isa.PSH:
            sp -= 4; mem[sp] = ax & mask
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0) & mask; sp += 4
            if op == isa.ADD: ax = (v + ax) & mask
            elif op == isa.SUB: ax = (v - ax) & mask
            elif op == isa.MUL: ax = (v * ax) & mask
            elif op == isa.DIV: ax = ((v // ax) if ax else 0) & mask
            else: ax = ((v % ax) if ax else 0) & mask
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0) & mask
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0); sp += 4; mem[addr] = ax & mask
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
            raise NotImplementedError(
                f"op {isa.NAMES.get(op, NAMES.get(op, op))} not in exact-steps ref ISA")
        trace.append(ax & mask)
    return trace


# ===========================================================================
# OVERLAY: the mem-operand overlay + the <OP>M operand-address nibbles on the query
# row (the same static-per-step address ingest the MAC uses).
# ===========================================================================
def make_overlay_exact(code, L: ExactStepsLayout, store_log=None,
                       mac_addrs=None, mop_addr=None):
    base_overlay = MO.make_overlay_mem_operand(code, L, store_log=store_log,
                                               mac_addrs=mac_addrs)

    def overlay(x: torch.Tensor) -> None:
        base_overlay(x)
        if mop_addr is not None:
            for j, nv in enumerate(V.nibbles_of_value(mop_addr & 0xFFFFFFFF, 8)):
                x[0, -1, L.MOP_ADDR_NIB + j] = float(nv)
    return overlay


# ===========================================================================
# DRIVER: one VM step = one model.forward.  Extends the mem-operand driver with the
# <OP>M operand-address feed and their straight-line PC += 1.
# ===========================================================================
def run_exact_steps(model, L: ExactStepsLayout, code,
                    mac_b: Optional[Dict[int, int]] = None,
                    max_steps: int = 512, mask: int = 0xFF,
                    seed_mem: Optional[Dict[int, int]] = None, verbose: bool = False):
    """Run ``code`` (which may contain MAC and <OP>M opcodes) through the exact-steps
    model.  ``mac_b`` maps PC -> the second MAC operand address (first rides
    ``Instr.imm``); a <OP>M's single operand address rides ``Instr.imm``.  Returns
    the per-step AX trace (matching :func:`ref_interpret_exact`)."""
    from .nibble_pure_forward import _snap_lane
    mac_b = mac_b or {}
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
        mop_addr = None
        if op == MO.MAC:
            mac_addrs = {"cur": (imm & 0xFFFFFFFF, mac_b.get(cur_pc, 0) & 0xFFFFFFFF)}
        elif op in _MOP_ALU:
            mop_addr = imm & 0xFFFFFFFF
        overlay = make_overlay_exact(code, L, store_log=store_log,
                                     mac_addrs=mac_addrs, mop_addr=mop_addr)
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
            nm = isa.NAMES.get(op, NAMES.get(op, "MAC" if op == MO.MAC else str(op)))
            print(f"  step pc={cur_pc} op={nm:4s} -> pc'={pc} ax={ax & mask} "
                  f"sp={sp} halt={halted}")
        cur_pc, cur_sp, cur_bp, cur_ax = pc, sp, bp, ax
        if halted or pc < 0 or pc >= len(code):
            break
    return trace
