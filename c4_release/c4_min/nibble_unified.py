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
    _write_reg_nibbles, VALVOCAB, vm_width32, maybe_cast_model_for_width,
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


# NOTE: the 256x256x3 dense MUL/DIV/MOD LOOKUP TABLE has been REMOVED entirely.
# ``mdm_housekeep_spec`` / ``extend_layout_for_mdm`` / ``compile_mdm_expand`` /
# ``compile_mdm_select`` (the byte×byte table gadget that materialised the ~45 GB /
# intermediate ~160465 fp32 wall) are GONE.  MUL/DIV/MOD now run ONLY through the
# efficient ``nibble_alu32`` fp32 FFN gadgets (byte MUL schoolbook + base-16 long
# division), used by the Qwen build in ``qwen_full_vm``.  ``_MDM_FN`` (the tiny
# 3-lambda truth table above) is kept purely as the ANALYTIC reference the fit
# configurator uses to count the removed table's width (tensor-free); nothing builds
# a table from it any more.


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
        # (The 256x256 MUL/DIV/MOD lookup-table operand one-hot bands have been
        # removed — the dense table is gone; MUL/DIV/MOD run only through the
        # efficient nibble_alu32 ALU in the Qwen build.)
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


def build_bitwise_blocks(L, dim, barrel_shift_ops=(isa.SHL, isa.SHR)
                         ) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Compile the OR/XOR/AND/SHL/SHR dispatch as opcode-gated FFN sub-blocks on
    the AX/STACK0 nibble bands. ALL five ops share ONE per-nibble bit-plane
    extraction (``bw-bitplanes``): OR/XOR/AND via the boolean per-bit combine
    (``AND=a·b``, ``OR=a+b-a·b``, ``XOR=a+b-2a·b`` on the shared planes) and
    SHL/SHR via the bitwise LOG-SHIFTER — the 5 conditional 2**k stages
    (``shift by 2**k`` gated on AX bit k), each a per-output-bit 2:1 mux built from
    the same boolean AND/OR machinery, plus an ``n >= 32 -> 0`` fold.  There is NO
    dense per-value table AND NO barrel select on the build path any more; the log
    stages read the SOURCE planes ``A_BIT`` and take the shift amount straight from
    the AX planes ``B_BIT[0..4]`` (no shift-amount one-hot expand, no MUL/DIV).
    Each op's rules are OP_IS-gated; the log stages of SHL and SHR share the
    ``SH_STAGE`` pipeline buffers (only one opcode fires at a time).  Returns named
    specs.

    ``barrel_shift_ops`` (default ``(SHL, SHR)``) selects WHICH shift ops the
    LOG-SHIFTER owns.  In the full (muldiv+bitwise) build with the legacy
    ``shift_via_mul`` on it is passed ``()`` — SHL/SHR are then computed via the
    NATIVE MUL/DIV gadgets (``nibble_alu32.compile_shift_pow2_route`` + the ax-mux),
    so the shifter blocks are DROPPED entirely and only the OR/XOR/AND combine
    remains.  The log-shifter is the DEFAULT and needs no muldiv, so it also works
    in a muldiv-less bitwise subset (where ``shift_via_mul`` is unavailable)."""
    _bw.extend_layout_for_bitwise(L)                  # allocate A_BIT/B_BIT/SH_STAGE
    L.D = L._off                                      # bands grew; refresh D
    blocks: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    # ONE shared, opcode-INDEPENDENT bit-plane extraction: STACK0/AX nibbles ->
    # A_BIT/B_BIT (4 planes each).  OR/XOR/AND read A_BIT+B_BIT; the log-shifter
    # reads the source planes A_BIT and the shift-amount bits B_BIT[0..4].
    planes = _bw.compile_bit_extract(
        src_bands=[L.STACK0 + j for j in range(_bw.N_NIB)]
                  + [L.AX + j for j in range(_bw.N_NIB)],
        bit_bases=[L.A_BIT + j * 4 for j in range(_bw.N_NIB)]
                  + [L.B_BIT + j * 4 for j in range(_bw.N_NIB)],
        one_band=L.ONE, dim=L.D)
    blocks.append(("bw-bitplanes", planes))
    # OR/XOR/AND per-op selects, opcode-gated, merged into ONE combine block.
    comb: List[FFNRule] = []
    for op in (isa.OR, isa.XOR, isa.AND):
        s = _bw.perbit_select_rules(L, op)               # shared per-bit gadget
        comb += _opcode_gate_rules(s, L.OP_IS + op)
    blocks.append(("bw-combine", compile_ffn(comb, L.D)))
    # SHL/SHR shifter.  DEFAULT: the TIGHT direct-8x8 nibble shifter
    # (``C4_TIGHT_SHIFT`` ON) — a 6-block DIRECT nibble pipeline (amount decode,
    # coarse select, fine product/peel, assemble) per direction reading the operand
    # straight from the STACK0 nibbles (NO bit-planes).  Each direction has PRIVATE
    # scratch bands so both run unconditionally; only the OUT->AX recompose is
    # OP_IS-gated (merged for SHL+SHR).  FALLBACK (``C4_TIGHT_SHIFT=0``): the
    # bit-granular LOG-SHIFTER — the shift is a PIPELINE (each mux stage reads the
    # prior stage's buffer), stages are SEPARATE sequential FFN blocks; SHL and SHR
    # share the SH_STAGE buffers, both OP_IS-gated.
    if barrel_shift_ops:
        if _bw.barrel_shift_enabled():
            # BARREL shifter (C4_BARREL_SHIFT=1): 4 blocks/direction (vs the tight
            # path's 6/8).  Private scratch per direction; recompose OP_IS-gated.
            for name, spec in _bw.unified_barrel_shift_blocks(
                    L, lambda: L.D, shift_ops=barrel_shift_ops):
                blocks.append((name, spec))
        elif _bw.tight_shift_enabled():
            for name, spec in _bw.unified_tight_shift_blocks(
                    L, lambda: L.D, shift_ops=barrel_shift_ops):
                blocks.append((name, spec))
        else:
            n_stage_blocks = _bw.LOG_STAGES + 2
            merged: List[List[FFNRule]] = [[] for _ in range(n_stage_blocks)]
            for op in barrel_shift_ops:                  # (SHL, SHR) unless via mul
                stage_blocks = _bw.shift_stage_blocks(L, op)
                for bi, sb in enumerate(stage_blocks):
                    merged[bi] += _opcode_gate_rules(sb, L.OP_IS + op)
            names = (["bw-shift-keep"]
                     + [f"bw-shift-log{k}" for k in range(_bw.LOG_STAGES)]
                     + ["bw-shift-recompose"])
            for name, rules in zip(names, merged):
                blocks.append((name, compile_ffn(rules, L.D)))
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
                        include_bitwise: bool = True):
    """Assemble ONE persistent ``blogspec_model.Transformer`` = the whole C4 VM
    step (base / cmp / bitwise + §Memory). Returns ``(model, L, meta)``.

    Block stack (each block = softmax1+ALiBi attention THEN SwiGLU FFN):

      block 0 : KV-MEMORY attention head (real §Memory CAM) + recompose FFN
      1 pc-fetch | 2 code-select (universal fetch) | 3 opcode-decode
      [bw-bitplanes | bw-combine | bw-shift-{keep,log0..4,recompose} | bw-recompose]
                                                        (OR/XOR/AND + LOG-SHIFTER SHL/SHR)
      DISPATCH : MoE (StandardMoEFFN, per-op experts) | branch-delta | fold

    Everything is in the model weights: universal fetch (code-as-data), in-model
    MoE dispatch, softmax1-KV memory, the op FFN experts, and the bitwise FFN
    sub-blocks. No functional gadget, no Python dispatch, no Python memory dict on
    the compute path.

    NOTE: MUL/DIV/MOD are NOT baked into this lean model — the 256x256x3 lookup
    table has been removed; the only MUL/DIV/MOD path is the efficient
    ``nibble_alu32`` ALU used by the Qwen build (``qwen_full_vm``).
    """
    # The unified build's CMP / bitwise / MUL-DIV lanes are single-scalar (NOT ported
    # to the fp32 two-limb AX/STACK0 representation), so PIN single-scalar for the
    # whole build+drive: the shared recompose/dispatch/branch/requant functions all
    # key off ``vm_two_limb()``, and this keeps them consistent regardless of the
    # ambient ``C4_VM_WIDTH32`` / ``C4_VM_TWO_LIMB`` flags.  The width-32 single-scalar
    # substrate still runs in fp64 here (its historical behaviour).
    from .nibble_vm import two_limb_mode
    with two_limb_mode(False):
        return _build_unified_model_impl(code_size, n_heads, include_bitwise)


def _build_unified_model_impl(code_size, n_heads, include_bitwise):
    L = UnifiedLayout(code_size, n_heads=n_heads)
    L.two_limb = False                       # STAMP: single-scalar (driver honours it)
    if include_bitwise:
        _bw.extend_layout_for_bitwise(L)
        # TIGHT shifter (C4_TIGHT_SHIFT, default ON): allocate its private per-op
        # scratch bands NOW, before ``dim`` is fixed below, so the tight shift blocks
        # (compiled inside build_bitwise_blocks at the fixed ``dim``) reference valid
        # residual dims.  OR/XOR/AND keep the shared A_BIT/B_BIT bit-planes.
        if _bw.tight_shift_enabled():
            for op in (isa.SHL, isa.SHR):
                _bw.extend_layout_for_tight_shift(L, op)
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
    all_specs = {**base_specs, **cmp_specs}
    if include_bitwise:
        # bitwise experts do only PC+=1 ; SP+=4 (the result is written to the AX
        # nibble bands by the bw-combine / log-shifter blocks, then bw-recompose).
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
    if include_bitwise:
        for name, spec in build_bitwise_blocks(L, dim):
            pre_blocks.append((name, spec))
        pre_blocks.append(("bw-recompose", _bw_recompose_spec(
            L, dim, (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR))))
    post_blocks = [
        ("branch-delta", compile_branch_delta(L, dim)),
        ("fold",        compile_fold(L.AX_VAL, L.ONE, dim)),   # width-aware (256/2^32)
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
    # width-32: run the whole step in fp64 (SUB +2^32 shift / 16^7 recompose /
    # per-byte requant exact to 2^32).  No-op under the 8-bit substrate.
    maybe_cast_model_for_width(model)

    meta = {
        "code_size": code_size, "n_heads": n_heads, "dim": dim,
        "n_blocks": n_blocks, "hidden": hidden,
        "block_names": [name for name, _ in block_plan],
        "moe_experts": expert_ops,
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
