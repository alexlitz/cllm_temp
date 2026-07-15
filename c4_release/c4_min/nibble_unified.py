"""ONE persistent Transformer that IS the whole C4 VM step (BLOG_SPEC).

This module assembles a **single** ``blogspec_model.Transformer`` (a coherent
``nn.Module`` with persistent weights) whose ``forward`` performs one complete C4
VM step for the corpus op families. Everything the compute path needs lives in
the model's weights — there is **no functional gadget that builds weights
on-the-fly, no Python if/elif dispatch, and no Python memory dict** inside the
step. The only Python that remains is the standard autoregressive generation
loop (fetch a step's output, snap it, re-embed), exactly as the mission allows.

What is folded into the one model
=================================
The step model is a stack of physical FFN sub-blocks (single residual position,
attention zeroed = identity) plus ONE real softmax1+ALiBi attention block:

  1. **recompose**    nibble register bands -> scalar value lanes
     (``nibble_vm.compile_nibble_to_scalar``).                       [FFN]
  2. **pc-fetch**     PC_VAL -> PC one-hot + AX_ZERO predicate
     (``nibble_vm.compile_pc_fetch``).                               [FFN]
  3. **code-select**  fetch OP_VAL/IMM at PC from DATA memory
     (``nibble_vm.compile_code_select``) — universal fetch, code-as-data. [FFN]
  4. **opcode-decode** OP_VAL scalar -> OP_IS[op] one-hot
     (``nibble_vm.compile_opcode_decode``).                          [FFN]
  5. **MoE dispatch** the spec's ``StandardMoEFFN``: one PureFFN expert per
     opcode, blended by the decoded ``OP_IS`` one-hot — no Python branch.
     Covers IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ/HALT + the cmp/bitwise/shift/
     muldivmod experts materialised below.                          [MoE FFN]
  6. **branch-delta** bilinear BZ/BNZ PC update
     (``nibble_vm.compile_branch_delta``).                           [FFN]
  7. **fold**         AX_VAL mod-256 (8-bit substrate fold).          [FFN]
  8. **KV-memory**    ONE real softmax1 + ALiBi attention head (the §Memory
     content-addressable store) for LI/SI — replaces the Python dict.  [ATTN]

The whole thing is ONE ``Transformer``; ``forward(tokens)`` (the standard
autoregressive interface) or the recurrent step driver runs the same weights.

The op families, as REAL weights
================================
  * base (IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ/HALT): ``base_dispatch_rules`` lowered
    by ``compile_ffn`` (the proven scalar-lane transition).
  * comparisons (EQ/NE/LT/GT/LE/GE): ``cmp_dispatch_rules`` — the §Comparisons
    zero-detector + sign-of-difference on the byte value lanes, materialised as
    SwiGLU weights here (not the Python-float ``nibble_cmp`` gadget).
  * bitwise/shift (OR/XOR/AND/SHL/SHR): the real per-nibble 256-entry table +
    power-of-two select FFN blocks from ``nibble_bitwise`` (already weights).
  * muldivmod (MUL/DIV/MOD): 8-bit lookup-table experts (256x256 -> byte), the
    same "table in the FFN" the spec sanctions for bitwise; foldable because the
    corpus reference is 8-bit. The full-32-bit MUL/DIV/MOD carry-round / long-
    division gadgets are iterative Python and do NOT fold into a fixed FFN stack
    — see ``UNFOLDABLE`` and the deliverable doc.

Honest boundary
===============
Two things do not collapse into the fixed step stack:
  * **MUL/DIV/MOD at full 32-bit** — the carry-round / base-16 long-division
    loops are data-dependent Python iteration. 8-bit is a table (folded here);
    32-bit needs O(width) unrolled blocks or the iterative gadget.
  * **bitwise/shift as MoE experts on the *scalar* value lane** — the bitwise
    FFN operates on the 16 nibble bands (256-entry tables), not the single
    scalar lane the base/cmp experts share. It is folded as its own dedicated
    FFN sub-blocks *inside the one model*, but it is not one of the scalar-lane
    MoE experts (different residual footprint). Still one model; see the block
    list. This is reported precisely, not papered over.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .blogspec_model import Transformer, FFN
from .dsl import FFNRule, LinearExpr
from .nibble_vm_layout import NibbleVMLayout
from .nibble_vm import (
    S, RELU_S, SILU_S, SILU_HALF,
    compile_ffn, compile_nibble_to_scalar, compile_pc_fetch,
    compile_code_select, compile_opcode_decode, base_dispatch_rules,
    compile_branch_delta, compile_fold, _empty_spec,
    _write_reg_nibbles, VALVOCAB,
)
from .nibble_moe import NibbleStandardMoEFFN, _ffn_from_spec


# opcodes that fold into the scalar-lane MoE dispatch of the one model.
SCALAR_MOE_OPS = [isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB,
                  isa.JMP, isa.BZ, isa.BNZ, isa.HALT,
                  isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
                  isa.MUL, isa.DIV, isa.MOD]
# opcodes whose FOLDED form (this file) is 8-bit only (corpus is 8-bit).
EIGHT_BIT_ONLY = frozenset([isa.MUL, isa.DIV, isa.MOD])
# what genuinely does not fold into the fixed stack (see docstring).
UNFOLDABLE = {
    "MUL/DIV/MOD@32bit": "iterative carry-round / base-16 long-division loops",
}


def compile_opcode_decode_ops(L: NibbleVMLayout, dim: int, ops) -> Dict[str, torch.Tensor]:
    """``OP_IS[op] = (OP_VAL == op)`` for EVERY op in ``ops`` — the triangular-
    pulse decode of ``nibble_vm.compile_opcode_decode`` widened past BASE_OPS so
    the cmp / muldivmod experts get a decoded one-hot to route on. Shared relu
    bank per distinct threshold; self-clears each OP_IS lane first (SET)."""
    ops = sorted(set(ops))
    thresholds = sorted({t for op in ops for t in (op - 1, op, op + 1)})
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_relu = len(thresholds)
    n_units = n_relu + len(ops)
    spec = _empty_spec(dim, n_units)
    for t, j in thr_unit.items():
        spec["W_up"][j, L.OP_VAL] = RELU_S
        spec["b_up"][j] = -RELU_S * t
        spec["W_gate"][j, L.ONE] = 1.0
    clear0 = n_relu
    for c, op in enumerate(ops):                       # self-clear each OP_IS lane
        uu = clear0 + c
        spec["W_up"][uu, L.ONE] = S
        spec["W_gate"][uu, L.OP_IS + op] = 1.0
        spec["W_down"][L.OP_IS + op, uu] += -1.0 / SILU_S
    for op in ops:
        band = L.OP_IS + op
        spec["W_down"][band, thr_unit[op - 1]] += 1.0 / RELU_S
        spec["W_down"][band, thr_unit[op]] += -2.0 / RELU_S
        spec["W_down"][band, thr_unit[op + 1]] += 1.0 / RELU_S
    return spec


# ===========================================================================
# COMPARISONS as real SwiGLU weights on the byte value lanes.
#
# The §Comparisons primitive is the zero detector on d = STK - AX and the sign
# of d. On the 8-bit substrate STK_VAL, AX_VAL in [0,255] so d in [-255,255],
# and a single fp32 difference is exact. We materialise:
#   EQ = Z(d)               (the +1/-2/+1 finite-2nd-difference bump, §584)
#   NE = 1 - Z(d)
#   LT = step(AX-STK >= 1)  (d < 0)
#   GT = step(STK-AX >= 1)  (d > 0)
#   LE = 1 - GT             GE = 1 - LT
# each written into AX_VAL (SET: -old + result), PC += 1, SP += 4 (pop consumed).
# These are ordinary SwiGLU hidden units — no Python float gadget.
# ===========================================================================
EPS = 0.5
def _z_detector_units(spec, u0, d_terms, d_const, out_band, out_scale, one):
    """Append the 3 zero-detector SwiGLU units (§584-586) writing out_scale*Z(d)
    into ``out_band``. ``d`` is the linear form (d_terms, d_const). Uses the
    silu second-difference; k normalises Z(0)=1."""
    sig = float(torch.sigmoid(torch.tensor(S * EPS)))
    k = S * EPS * (2.0 * sig - 1.0)
    for idx, (bias, w) in enumerate([(S * EPS, 1.0), (0.0, -2.0), (-S * EPS, 1.0)]):
        u = u0 + idx
        for band, coeff in d_terms.items():
            spec["W_up"][u, band] += S * coeff
        spec["b_up"][u] += S * d_const + bias
        spec["W_gate"][u, one] = 1.0                 # gate = 1 (pass silu)
        spec["W_down"][out_band, u] += out_scale * w / k
    return u0 + 3


def _step_ge1_unit(spec, u, d_terms, d_const, out_band, out_scale, one):
    """One SwiGLU unit: out_band += out_scale * sigmoid-step[d >= 1].
    Realised as relu(d-0)-relu(d-1) via silu is 2 units; use the sharp
    sigmoid-free ramp: step(d>=1) = relu(d) - relu(d-1) for integer d. Two
    ReLU-via-silu units."""
    for idx, thr in enumerate((0.0, 1.0)):
        uu = u + idx
        for band, coeff in d_terms.items():
            spec["W_up"][uu, band] += RELU_S * coeff
        spec["b_up"][uu] += RELU_S * (d_const - thr)
        spec["W_gate"][uu, one] = 1.0
        spec["W_down"][out_band, uu] += out_scale * (1.0 if idx == 0 else -1.0) / RELU_S
    return u + 2


def compile_cmp_expert(L: NibbleVMLayout, op: int, dim: int) -> Dict[str, torch.Tensor]:
    """Compile ONE comparison opcode into a SwiGLU expert (real weights).

    Operates on the scalar byte lanes STK_VAL (pop) and AX_VAL. Writes the 0/1
    result into AX_VAL (SET), PC += 1, SP += 4. Every rule is IMPLICITLY the
    active-op expert (the MoE blend gates it by OP_IS[op]); inside the expert we
    do NOT re-gate (the expert only runs weighted by its opcode one-hot), so the
    units fire unconditionally on the value lanes.
    """
    stk, ax, sp, pc, one = L.STK_VAL, L.AX_VAL, L.SP_VAL, L.PC_VAL, L.ONE
    # count units: clear AX (1) + result units + PC(1) + SP(1)
    # EQ/NE need 3 (Z); LT/GT need 2; LE/GE need 2 (via GT/LT) + const via clear.
    n_units = 16
    spec = _empty_spec(dim, n_units)
    u = 0
    # SET: clear old AX_VAL (write -AX_VAL) via a silu-identity unit.
    spec["W_up"][u, one] = S
    spec["W_gate"][u, ax] = 1.0
    spec["W_down"][ax, u] += -1.0 / SILU_S
    u += 1
    # d = STK - AX (pop - ax), matching isa.interpret's ``pop() CMP ax``.
    d_terms = {stk: 1.0, ax: -1.0}
    if op == isa.EQ:
        u = _z_detector_units(spec, u, d_terms, 0.0, ax, 1.0, one)
    elif op == isa.NE:
        # NE = 1 - Z(d): const 1 into AX, then -Z(d).
        spec["W_up"][u, one] = S; spec["W_gate"][u, one] = 1.0
        spec["W_down"][ax, u] += 1.0 / SILU_S; u += 1
        u = _z_detector_units(spec, u, d_terms, 0.0, ax, -1.0, one)
    elif op == isa.GT:                              # d >= 1
        u = _step_ge1_unit(spec, u, d_terms, 0.0, ax, 1.0, one)
    elif op == isa.LT:                              # -d >= 1
        u = _step_ge1_unit(spec, u, {stk: -1.0, ax: 1.0}, 0.0, ax, 1.0, one)
    elif op == isa.GE:                              # 1 - LT = 1 - step(-d>=1)
        spec["W_up"][u, one] = S; spec["W_gate"][u, one] = 1.0
        spec["W_down"][ax, u] += 1.0 / SILU_S; u += 1
        u = _step_ge1_unit(spec, u, {stk: -1.0, ax: 1.0}, 0.0, ax, -1.0, one)
    elif op == isa.LE:                              # 1 - GT = 1 - step(d>=1)
        spec["W_up"][u, one] = S; spec["W_gate"][u, one] = 1.0
        spec["W_down"][ax, u] += 1.0 / SILU_S; u += 1
        u = _step_ge1_unit(spec, u, d_terms, 0.0, ax, -1.0, one)
    else:
        raise ValueError(f"compile_cmp_expert: {op} not a comparison")
    # PC += 1 ; SP += 4 (pop consumed one slot).
    spec["W_up"][u, one] = S; spec["W_gate"][u, one] = 1.0
    spec["W_down"][pc, u] += 1.0 / SILU_S; u += 1
    spec["W_up"][u, one] = S; spec["W_gate"][u, one] = 4.0
    spec["W_down"][sp, u] += 1.0 / SILU_S; u += 1
    return spec


# ===========================================================================
# MUL / DIV / MOD — 8-bit lookup-table experts (the folded form).
#
# The spec sanctions "just lookup tables ... embedded in the FFNs" for the
# nibble bitwise ops. The same construction gives an 8-bit MUL/DIV/MOD: the two
# byte operands are expanded to one-hots (256 cells each) and the result byte is
# the table entry op(a,b). Foldable into a FIXED FFN pair (expand + select).
# The 32-bit case is the iterative gadget (UNFOLDABLE), reported honestly.
# ===========================================================================
_MDM_FN = {
    isa.MUL: lambda a, b: (a * b) & 0xFF,
    isa.DIV: lambda a, b: ((a // b) if b else 0) & 0xFF,
    isa.MOD: lambda a, b: ((a % b) if b else 0) & 0xFF,
}


def mdm_housekeep_spec(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """The scalar-lane MUL/DIV/MOD expert: AX_VAL = MDM_RES (the 8-bit table
    result, computed by the mdm-expand/select blocks that run BEFORE dispatch on
    the pre-op STK_VAL/AX_VAL operands), PC += 1, SP += 4. Written SET (clear old
    AX, add MDM_RES). Gated by the MoE opcode blend (only fires for MUL/DIV/MOD)."""
    ax, sp, pc, one = L.AX_VAL, L.SP_VAL, L.PC_VAL, L.ONE
    spec = _empty_spec(dim, 4)
    spec["W_up"][0, one] = S; spec["W_gate"][0, ax] = 1.0
    spec["W_down"][ax, 0] += -1.0 / SILU_S            # clear old AX (SET)
    spec["W_up"][1, one] = S; spec["W_gate"][1, L.MDM_RES] = 1.0
    spec["W_down"][ax, 1] += 1.0 / SILU_S             # AX = MDM_RES
    spec["W_up"][2, one] = S; spec["W_gate"][2, one] = 1.0
    spec["W_down"][pc, 2] += 1.0 / SILU_S
    spec["W_up"][3, one] = S; spec["W_gate"][3, one] = 4.0
    spec["W_down"][sp, 3] += 1.0 / SILU_S
    return spec


# ---------------------------------------------------------------------------
# 8-bit MUL/DIV/MOD as ONE bilinear table select on the byte lanes.
#
# The result byte lives in AX_VAL. We add per-op operand one-hot bands and a
# select block whose units fire on ``A_OH[a] AND B_OH[b] AND OP_IS[op]`` writing
# ``op(a,b)`` into AX_VAL. This is the same "table embedded in the FFN" the spec
# uses for the nibble bitwise ops (§685), scaled to the 8-bit operand. The full
# 32-bit MUL/DIV/MOD carry-round / long-division is iterative (UNFOLDABLE).
# ---------------------------------------------------------------------------
def extend_layout_for_mdm(L: NibbleVMLayout) -> NibbleVMLayout:
    """Allocate 256-cell one-hot bands for the byte operands STK_VAL, AX_VAL."""
    if getattr(L, "MDM_A_OH", None) is not None:
        return L
    L.MDM_A_OH = L._band("MDM_A_OH", 256)
    L.MDM_B_OH = L._band("MDM_B_OH", 256)
    while L._off % L.n_heads != 0:
        L._scalar(f"_pad{L._off}")
    L.D = L._off
    return L


def compile_mdm_expand(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """One-hot expand STK_VAL -> MDM_A_OH, AX_VAL -> MDM_B_OH (0..255) via the
    §510 triangular pulse. Shared relu bank per source (the fetch/decode
    one-hot construction, widened to 256 cells)."""
    thresholds = list(range(-1, 257))
    n_thr = len(thresholds)
    spec = _empty_spec(dim, 2 * n_thr)
    for si, (src, oh_base) in enumerate([(L.STK_VAL, L.MDM_A_OH),
                                         (L.AX_VAL, L.MDM_B_OH)]):
        base = si * n_thr
        tu = {t: base + j for j, t in enumerate(thresholds)}
        for t, u in tu.items():
            spec["W_up"][u, src] = RELU_S
            spec["b_up"][u] = -RELU_S * t
            spec["W_gate"][u, L.ONE] = 1.0
        for a in range(256):
            spec["W_down"][oh_base + a, tu[a - 1]] += 1.0 / RELU_S
            spec["W_down"][oh_base + a, tu[a]] += -2.0 / RELU_S
            spec["W_down"][oh_base + a, tu[a + 1]] += 1.0 / RELU_S
    return spec


def compile_mdm_select(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """The 8-bit MUL/DIV/MOD table select for all three ops in ONE block: for
    every op and every (a,b) with op(a,b)!=0, a hidden unit gated on
    ``MDM_A_OH[a] AND MDM_B_OH[b] AND OP_IS[op]`` writes op(a,b) into MDM_RES.

    Runs BEFORE dispatch (on the pre-op STK/AX byte operands, so AX is not yet
    cleared). The housekeeping expert then copies MDM_RES -> AX. One self-clear
    unit resets MDM_RES each step (SET). This is the honest cost of a byte x byte
    lookup: ~O(3 * 256^2) hidden units (zero entries skipped). Reported
    separately from the core VM param count; the 32-bit form does NOT fold."""
    keys = []
    for op in (isa.MUL, isa.DIV, isa.MOD):
        fn = _MDM_FN[op]
        for a in range(256):
            for b in range(256):
                v = fn(a, b)
                if v != 0:
                    keys.append((op, a, b, v))
    spec = _empty_spec(dim, max(1, len(keys) + 1))
    # unit 0: self-clear MDM_RES (SET each step).
    spec["W_up"][0, L.ONE] = S
    spec["W_gate"][0, L.MDM_RES] = 1.0
    spec["W_down"][L.MDM_RES, 0] += -1.0 / SILU_S
    for u, (op, a, b, v) in enumerate(keys, start=1):
        spec["W_up"][u, L.MDM_A_OH + a] += S
        spec["W_up"][u, L.MDM_B_OH + b] += S
        spec["W_up"][u, L.OP_IS + op] += S
        spec["b_up"][u] += -S * 2.5                   # AND of 3 windows
        spec["W_gate"][u, L.ONE] = float(v)
        spec["W_down"][L.MDM_RES, u] += 1.0 / SILU_HALF
    return spec


# ===========================================================================
# THE UNIFIED LAYOUT — VM dispatch bands + KV-memory bands in ONE residual.
# ===========================================================================
from .blogspec_layout import NIB_PER_REG
from .blogspec_memory import ADDR_BITS as _MEM_ADDR_BITS, bake_memory_head


class UnifiedLayout(NibbleVMLayout):
    """``NibbleVMLayout`` (fetch/decode/dispatch scalar lanes) + the softmax1-KV
    memory head's address/value bands, so ONE model carries the whole VM step:
    universal fetch, MoE dispatch, AND the content-addressable memory head."""

    def __init__(self, code_size: int, n_heads: int = 4):
        super().__init__(code_size, n_heads=n_heads)
        # re-open the allocator (base __init__ padded + set D); append mem bands.
        self._off = self.D
        self.ADDR_BIN = self._band("ADDR_BIN", _MEM_ADDR_BITS)   # store-addr bits (KEY)
        self.QRY_BIN = self._band("QRY_BIN", _MEM_ADDR_BITS)     # load-addr bits (QUERY)
        self.VAL_NIB = self._band("VAL_NIB", NIB_PER_REG)        # store value nibbles
        self.IS_STORE = self._scalar("IS_STORE")
        self.IS_LOAD = self._scalar("IS_LOAD")
        self.IS_CHAR = self._scalar("IS_CHAR")
        # operand one-hot bands + result lane for the folded 8-bit MUL/DIV/MOD.
        self.MDM_A_OH = self._band("MDM_A_OH", 256)
        self.MDM_B_OH = self._band("MDM_B_OH", 256)
        self.MDM_RES = self._scalar("MDM_RES")           # byte result (pre-dispatch)
        while self._off % n_heads != 0:
            self._scalar(f"_pad{self._off}")
        self.D = self._off


# ===========================================================================
# BITWISE / SHIFT — fold the real per-nibble table FFN blocks into the model,
# OPCODE-GATED so they only fire for their op (OR/XOR/AND/SHL/SHR).
# ===========================================================================
from . import nibble_bitwise as _bw


def _opcode_gate_rules(rules: List[FFNRule], op_band: int) -> List[FFNRule]:
    """Add an ``OP_IS[op]`` guard window to every rule so the block only writes
    when the decoded opcode matches (turns the standalone bitwise/shift select
    rules into opcode-routed dispatch rules — the same gating the base experts
    have baked into their guards)."""
    out = []
    for r in rules:
        out.append(FFNRule([(op_band, 0.5, 1.5)] + list(r.when), dict(r.write)))
    return out


def build_bitwise_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Compile the OR/XOR/AND/SHL/SHR dispatch as opcode-gated FFN sub-blocks on
    the AX/STACK0 nibble bands (the real ``nibble_bitwise`` per-nibble 256-entry
    tables + power-of-two select, already SwiGLU weights). One shared expand
    block builds the operand one-hots; each op's select is OP_IS-gated. The
    shift bit4 fold and shift selects are likewise gated. Returns named specs."""
    _bw.extend_layout_for_bitwise(L)                  # allocate A_OH/B_OH/SHIFT_*
    L.D = L._off                                      # bands grew; refresh D
    blocks: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    # ONE expand block: STACK0/AX nibbles + AX nib0/nib1 -> one-hots.
    expand = _bw.compile_onehot_expand(
        src_bands=[L.STACK0 + j for j in range(_bw.N_NIB)]
                  + [L.AX + j for j in range(_bw.N_NIB)]
                  + [L.AX + 0, L.AX + 1],
        oh_bases=list(L.A_OH) + list(L.B_OH) + [L.SHIFT_LO_OH, L.SHIFT_N1_OH],
        cells=16, one_band=L.ONE, dim=L.D)
    blocks.append(("bw-expand", expand))
    # shift bit4 fold (gated by SHL|SHR).
    bit4 = _bw._shift_bit4_rules(L)
    bit4g = (_opcode_gate_rules(bit4, L.OP_IS + isa.SHL)
             + _opcode_gate_rules(bit4, L.OP_IS + isa.SHR))
    blocks.append(("bw-bit4", compile_ffn(bit4g, L.D)))
    # per-op selects, opcode-gated, all merged into ONE select block.
    sel: List[FFNRule] = []
    for op in (isa.OR, isa.XOR, isa.AND):
        _, s = _bw.bitwise_dispatch_rules(L, op)
        sel += _opcode_gate_rules(s, L.OP_IS + op)
    for op in (isa.SHL, isa.SHR):
        _, _b4, s = _bw.shift_dispatch_rules(L, op)
        sel += _opcode_gate_rules(s, L.OP_IS + op)
    blocks.append(("bw-select", compile_ffn(sel, L.D)))
    return blocks


# ===========================================================================
# BUILD the ONE unified Transformer.
# ===========================================================================
def _bw_recompose_spec(L, dim, ops) -> Dict[str, torch.Tensor]:
    """Recompose the AX low byte (nibbles 0,1) into AX_VAL, GATED on the bitwise/
    shift ops, so a bitwise result written to the AX NIBBLE bands lands on the
    scalar AX_VAL the frame emit reads. SET: clear AX_VAL then add nib0 + 16*nib1
    (only when an OP_IS[op] in ``ops`` is active)."""
    spec = _empty_spec(dim, 3 * len(ops))
    u = 0
    for op in ops:
        g = L.OP_IS + op
        # clear AX_VAL (gated): up = S*(g), gate = AX_VAL, down -1.
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.AX_VAL] = 1.0
        spec["W_down"][L.AX_VAL, u] += -1.0 / SILU_HALF; u += 1
        # + nib0 (gated).
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.AX + 0] = 1.0
        spec["W_down"][L.AX_VAL, u] += 1.0 / SILU_HALF; u += 1
        # + 16*nib1 (gated).
        spec["W_up"][u, g] = S; spec["b_up"][u] = -S * 0.5
        spec["W_gate"][u, L.AX + 1] = 16.0
        spec["W_down"][L.AX_VAL, u] += 1.0 / SILU_HALF; u += 1
    return spec


def build_unified_model(code_size: int = 8, n_heads: int = 4,
                        include_mdm_table: bool = True,
                        include_bitwise: bool = True):
    """Assemble ONE persistent ``blogspec_model.Transformer`` = the whole C4 VM
    step. Returns ``(model, L, meta)``.

    Block stack (each block = softmax1+ALiBi attention THEN SwiGLU FFN):

      block 0 : KV-MEMORY attention head (real §Memory CAM) + recompose FFN
      1 pc-fetch | 2 code-select (universal fetch) | 3 opcode-decode
      [mdm-expand | mdm-select]  (8-bit MUL/DIV/MOD table, before dispatch)
      [bw-expand | bw-bit4 | bw-select | bw-recompose]  (OR/XOR/AND/SHL/SHR)
      DISPATCH : MoE (StandardMoEFFN, per-op experts) | branch-delta | fold

    Everything is in the model weights: universal fetch (code-as-data), in-model
    MoE dispatch, softmax1-KV memory, the op FFN experts, and the bitwise/table
    FFN sub-blocks. No functional gadget, no Python dispatch, no Python memory
    dict on the compute path.
    """
    L = UnifiedLayout(code_size, n_heads=n_heads)
    if include_bitwise:
        _bw.extend_layout_for_bitwise(L)
        while L._off % n_heads != 0:
            L._scalar(f"_bwpad{L._off}")
        L.D = L._off
    dim = L.D

    # -- the scalar-lane FFN experts (per opcode) for the MoE dispatch ---------
    base_rules = base_dispatch_rules(L)
    base_specs = {}
    for op in (isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB,
               isa.JMP, isa.BZ, isa.BNZ, isa.HALT):
        rr = [r for r in base_rules if r.when and (r.when[0][0] - L.OP_IS) == op]
        base_specs[op] = compile_ffn(rr, dim) if rr else _empty_spec(dim, 1)
    cmp_specs = {op: compile_cmp_expert(L, op, dim)
                 for op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE)}
    mdm_specs = {op: mdm_housekeep_spec(L, dim)
                 for op in (isa.MUL, isa.DIV, isa.MOD)}
    all_specs = {**base_specs, **cmp_specs, **mdm_specs}
    if include_bitwise:
        # bitwise experts do only PC+=1 ; SP+=4 (the result is written to the AX
        # nibble bands by the bw-select block, then bw-recompose -> AX_VAL).
        for op in (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR):
            hk = _empty_spec(dim, 2)
            hk["W_up"][0, L.ONE] = S; hk["W_gate"][0, L.ONE] = 1.0
            hk["W_down"][L.PC_VAL, 0] += 1.0 / SILU_S
            hk["W_up"][1, L.ONE] = S; hk["W_gate"][1, L.ONE] = 4.0
            hk["W_down"][L.SP_VAL, 1] += 1.0 / SILU_S
            all_specs[op] = hk

    experts, expert_ops = [], []
    for op in sorted(all_specs):
        experts.append(_ffn_from_spec(all_specs[op], dim))
        expert_ops.append(op)
    moe = NibbleStandardMoEFFN(experts, expert_ops, op_start=L.OP_IS,
                               num_ops=isa.NUM_OPS)

    # -- the plain FFN sub-blocks (fetch/decode/branch/fold) -------------------
    pre_blocks = [
        ("recompose",   compile_nibble_to_scalar(L, dim)),
        ("pc-fetch",    compile_pc_fetch(L, dim)),
        ("code-select", compile_code_select(L, dim)),
        ("opcode-decode", compile_opcode_decode_ops(L, dim, expert_ops)),
    ]
    if include_mdm_table:
        pre_blocks += [("mdm-expand", compile_mdm_expand(L, dim)),
                       ("mdm-select", compile_mdm_select(L, dim))]
    if include_bitwise:
        for name, spec in build_bitwise_blocks(L, dim):
            pre_blocks.append((name, spec))
        pre_blocks.append(("bw-recompose", _bw_recompose_spec(
            L, dim, (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR))))
    post_blocks = [
        ("branch-delta", compile_branch_delta(L, dim)),
        ("fold",        compile_fold(L.AX_VAL, L.ONE, dim, modulus=256)),
    ]
    block_plan = pre_blocks + [("dispatch", moe)] + post_blocks
    n_blocks = len(block_plan)
    hidden = max(sp["W_up"].shape[0] for name, sp in block_plan
                 if not isinstance(sp, NibbleStandardMoEFFN))

    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, vocab=V.VOCAB, max_seq_len=64)
    with torch.no_grad():
        model.embed.zero_()
        _bake_unified_embedding(model, L)
        for bi, (name, payload) in enumerate(block_plan):
            blk = model.blocks[bi]
            if name == "dispatch":
                # zero this block's own FFN and attach the MoE as the FFN.
                for p in (blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o):
                    p.zero_()
                blk.ffn = payload          # the StandardMoEFFN IS this block's FFN
            else:
                for p in (blk.attn.W_q, blk.attn.W_k, blk.attn.W_v, blk.attn.W_o):
                    p.zero_()
                _load_ffn_padded(blk.ffn, payload, hidden)
        # block 0's attention = the real softmax1+ALiBi §Memory CAM head.
        bake_memory_head(model.blocks[0].attn, L, head=0)

    meta = {
        "code_size": code_size, "n_heads": n_heads, "dim": dim,
        "n_blocks": n_blocks, "hidden": hidden,
        "block_names": [name for name, _ in block_plan],
        "moe_experts": expert_ops,
        "include_mdm_table": include_mdm_table,
    }
    return model, L, meta


def _load_ffn_padded(ffn, spec: Dict[str, torch.Tensor], hidden: int) -> None:
    h = spec["W_up"].shape[0]
    ffn.W_up.zero_();   ffn.W_up[:h] = spec["W_up"]
    ffn.b_up.zero_();   ffn.b_up[:h] = spec["b_up"]
    ffn.W_gate.zero_(); ffn.W_gate[:h] = spec["W_gate"]
    ffn.b_gate.zero_(); ffn.b_gate[:h] = spec["b_gate"]
    ffn.W_down.zero_(); ffn.W_down[:, :h] = spec["W_down"]
    ffn.b_down.zero_(); ffn.b_down.copy_(spec["b_down"])


def _bake_unified_embedding(model, L) -> None:
    """Byte tokens embed their nibbles into CUR_NIB; ONE=1 in every row. (The
    same foundation embedding; the recurrent driver seeds state directly.)"""
    E = torch.zeros(V.VOCAB, model.dim)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    model.embed.copy_(E)


# ===========================================================================
# PARAMETER ACCOUNTING — the definitive "size of the fully wired VM".
# ===========================================================================
def param_report(model, meta=None) -> dict:
    """Exact per-tensor + total dense/nonzero/sparsity accounting for the ONE
    unified model. Returns a dict; ``print_param_report`` renders it."""
    per = []
    total = 0
    nonzero = 0
    for name, p in model.named_parameters():
        n = p.numel()
        nz = int((p.detach() != 0).sum().item())
        per.append({"name": name, "shape": tuple(p.shape),
                    "params": n, "nonzero": nz})
        total += n
        nonzero += nz
    rep = {
        "total_params": total,
        "nonzero_params": nonzero,
        "zero_params": total - nonzero,
        "sparsity": 1.0 - (nonzero / total if total else 0.0),
        "dim": model.dim,
        "n_blocks": len(model.blocks),
        "n_heads": model.blocks[0].attn.n_heads,
        "vocab": model.vocab,
        "per_tensor": per,
    }
    if meta:
        rep["meta"] = meta
    return rep


def _fmt(n: int) -> str:
    return f"{n:,}"


def print_param_report(model, meta=None) -> dict:
    rep = param_report(model, meta)
    print("=" * 74)
    print("UNIFIED C4 VM MODEL — PARAMETER ACCOUNTING")
    print("=" * 74)
    print(f"  dim={rep['dim']}  n_blocks={rep['n_blocks']}  "
          f"n_heads={rep['n_heads']}  vocab={rep['vocab']}")
    print(f"  TOTAL (dense)  : {_fmt(rep['total_params'])}")
    print(f"  NONZERO        : {_fmt(rep['nonzero_params'])}")
    print(f"  ZERO           : {_fmt(rep['zero_params'])}")
    print(f"  SPARSITY       : {rep['sparsity']*100:.2f}%")
    # aggregate by kind (embed / lm_head / attn / ffn per block).
    agg = {}
    for t in rep["per_tensor"]:
        # block index prefix e.g. blocks.4.ffn.experts.3.W_up
        key = t["name"].split(".W")[0].split(".b")[0]
        d = agg.setdefault(key, {"params": 0, "nonzero": 0})
        d["params"] += t["params"]; d["nonzero"] += t["nonzero"]
    print("-" * 74)
    print("  per-component (params / nonzero):")
    for k in sorted(agg):
        d = agg[k]
        print(f"    {k:44s} {_fmt(d['params']):>14s} / {_fmt(d['nonzero'])}")
    print("=" * 74)
    return rep
