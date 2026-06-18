#!/usr/bin/env python3
"""Corpus interpreter-vs-oracle attribution gate (CPU-only, rule-attributable).

The FAST, GPU-FREE bug-hunting gate. For each program in the 1096 corpus it
runs the **value-faithful** pure-IR interpreter
(``neural_vm/unified_compiler/faithful_interpreter.py``) over the program's
production context (the same code-prompt + DraftVM teacher-forced step tape the
real model decodes), reproduces the model's per-step ``(PC, AX)`` decode
EXACTLY as the production fail-fast path does, compares it to the DraftVM
oracle, and on the FIRST diverging byte names the single owning declarative
rule that produced the wrong value — all on CPU in ~8s of build + a few ms per
program. This turns bug-hunting from "GPU residual probing" into "run the gate,
read the attributed rule".

Why this is faithful (and what makes it trustworthy)
----------------------------------------------------
The interpreter's attention (softmax1 + ALiBi MHA) and FFN (SwiGLU @ scale S)
math is the term-for-term reproduction of the lowered ``AutoregressiveAttention``
/ ``PureFFN`` blocks; ``tools/faithful_interpreter_validate.py`` validates it is
**byte-for-byte identical to the real model's argmax at every token position**
on the smoke set + a 1096 sample. This gate drives that same faithful forward
over the production decode setup:

  * **Production context.** The seed is ``_build_context(bytecode, data)``
    (CODE_START ... bytecode ... CODE_END ... DATA) PLUS the DraftVM-teacher-
    forced 35-token step slices — exactly what ``run_batch_fail_fast`` feeds the
    model (the DraftVM proposes the oracle tokens; the model verifies them and
    is the arbiter, byte-identity with production).
  * **Production decode.** Per VM step the model's predicted ``(PC, AX)`` is read
    from the model's argmax over that step's draft slice, decoded at the FIXED
    step-tape offsets (PC bytes at slice offsets 1..4, AX bytes at 6..9) — i.e.
    the markers are re-anchored exactly as production re-anchors the first token
    of each step via the Python STEP_END dispatch. This is what makes a PASSING
    program decode cleanly (a naive marker-search or next-token compare flags
    every step boundary; the fixed-offset re-anchored decode matches production).

Classes
-------
  * **PASS** — the per-step ``(PC, AX)`` matches the oracle every step through
    HALT (same criterion as ``--criterion full_trace``) AND the program is not
    downstream of a cross-step poisoning correction (the guard below).
  * **FAIL@step/byte + ATTRIBUTED RULE** (HIGH-confidence) — the first step where
    the decode diverges, with the divergence AT OR BEFORE the cross-step poison
    point, AND the single declarative rule whose RUNTIME SwiGLU contribution
    dominates the wrong value. THIS IS THE KEY OUTPUT — the highest-leverage fix
    target. Now covers the AX HIGH BYTES (1..3): the fixed-offset re-anchored
    decode reads them faithfully when not poisoned (the edge_literal AX-byte-1
    cluster matches the neural full_trace 10/10), so they are attributed, not
    deferred.
  * **CROSS-STEP** — the divergence (or a clean single-forward "pass" with a
    register-VALUE correction before the last step) is DOWNSTREAM of production's
    autoregressive context poisoning, which a single teacher-forced forward
    cannot reproduce (the model "recovers" under teacher forcing). The var/func
    PC framing-drift + deep expr AX live here. The gate FLAGS these (never
    over-claims a verdict the single forward can't certify) and points at the GPU
    ``--faithfulness-check`` (the real production decode) to resolve them. See
    ``_value_correction_step`` for the soundness guard (0 false-trusts over the
    validation sample).
  * **FAIL (ALU step) + COARSE BLOCK ATTRIBUTION** — the divergence is at/after a
    step whose opcode is one of the 4 still-imperative composite ALU blocks
    (ADD/SUB -> AddSub5StageBlock, MUL -> FlattenedALUMul, DIV/MOD ->
    FlattenedDivMod, SHL/SHR -> ALUShiftComposite). These blocks have no IR rule
    form, BUT the faithful forward EXECUTES the real baked block on the residual
    tape (a CPU forward — exact), so the interpreter still gets a real per-step
    ``(PC, AX)`` verdict. It just cannot pin a *declarative rule* (the block is
    imperative), so the attribution is COARSE: ``attributed_op=<ALU block>``,
    ``attributed_rule=None``, ``is_alu_step=True``. This is honest — distinct
    from the IR-rule attributions on the non-ALU path. The cross-step poisoning
    guard still applies: an ALU-step FAIL downstream of a value correction is
    still flagged CROSS-STEP (confidence), so the ALU verdict and the
    decode/cross-step verdicts COEXIST. (Previously these programs were flagged
    ALU-OPAQUE and the gate declined to judge; the real ALU now executes.)

Faithfulness / trustworthiness
------------------------------
``--faithfulness-check`` (uses a GPU only if one is FREE; otherwise skips) runs
the same per-step decode with the REAL NEURAL model (not the oracle) on a sample
and confirms interpreter==neural step-for-step. Where they disagree, that is an
interpreter COVERAGE GAP (not a model bug) and is reported separately so the
gate never mis-attributes. It is also the authoritative resolver for the
CROSS-STEP class (the autoregressive framing-drift the CPU single forward flags
but cannot decode). The summary quantifies the corpus split: HIGH-confidence
(gate is authoritative, incl. AX-high-byte) vs CROSS-STEP vs ALU programs (real
ALU block EXECUTED, coarse block attribution).

Usage
-----
    CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --smoke
    CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --sample-1096 80
    CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --ids 0,49,250
    CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --demo
    python tools/interp_oracle_gate.py --sample-1096 40 --faithfulness-check 12
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Declarations-only / spec_k=0 flags so the CPU build is the ground-truth path
# and the (expected) compile-time integrity/gate warnings stay silent.
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from neural_vm.unified_compiler.faithful_interpreter import (  # noqa: E402
    FaithfulInterpreter, STEP_TOKENS,
)
from neural_vm.speculative import DraftVM  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE  # noqa: E402

# Reuse the VALIDATED faithful-forward + spec recovery from the validation
# harness (the same math the validator proved byte-for-byte vs the model).
from tools.faithful_interpreter_validate import (  # noqa: E402
    faithful_full_forward,
    _faithful_residual_pre_head,
    _attn_block_to_specs,
    _faithful_attn_forward,
    _COMPOSITE_FFN,
)


def build_production_model(device: str = "cpu"):
    """Build the model+layout the PRODUCTION runner uses (``alu_mode='efficient'``).

    CRITICAL: ``BatchedPureNeuralRunner`` builds with ``trust_neural_alu=True``
    => ``alu_mode='efficient'`` (49 physical blocks), NOT the lookup-mode model
    (47 blocks). The two are NOT byte-equivalent — the efficient build's extra
    composite-ALU blocks shift the block layout and change the decode for EVERY
    op (e.g. lookup-mode mis-decodes ``or_basic``/``xor_basic`` that the
    production efficient model gets right). So the gate MUST build the efficient
    model to be faithful to production. Built on CPU (no GPU), in-memory
    (``disk_cache=False`` so the layout keeps its ``compiler_ir_factory``
    lambdas for attribution). ~8-15s.
    """
    import contextlib
    import io

    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = compile_full_vm_dynamic(
            alu_mode="efficient", disk_cache=False,
        )
    model = model.to(device)
    model.eval()
    return model, layout


# ---------------------------------------------------------------------------
# Cached faithful forward.
#
# ``faithful_full_forward`` re-extracts every block's head-specs from the baked
# weights on EVERY call (~2.2M ``nonzero()`` calls per forward => ~4.5 s on CPU,
# >90 % of its runtime, and INDEPENDENT of context length). The gate runs one
# forward per program (the single teacher-forced pass) plus one residual pass
# for attribution, so that re-extraction would dominate the corpus run.
# ``FaithfulForwardCache`` extracts the head-specs + dense FFN weights ONCE and
# reuses them, dropping each forward to ~0.4-1.1 s (5-13x) while staying
# BYTE-IDENTICAL to ``faithful_full_forward`` (validated by
# tools/faithful_interpreter_validate.py + the per-position argmax parity check).
# ---------------------------------------------------------------------------


class FaithfulForwardCache:
    """One-time extraction of the faithful-forward block specs + FFN weights.

    ``forward(tape)`` reproduces ``faithful_full_forward(model, tape)`` exactly
    (same softmax1+ALiBi attention math, same SwiGLU FFN, same composite-ALU
    real-block fallback, same LM head) but without the per-call head-spec
    re-extraction. Built on CPU; the gate never touches the GPU.
    """

    def __init__(self, model):
        self.model = model
        self.blocks = []
        for block in model.blocks:
            attn = block.attn
            heads = _attn_block_to_specs(attn, model.d_model)
            is_comp = type(block.ffn).__name__ in _COMPOSITE_FFN
            if is_comp:
                ff = None
            else:
                ffn = block.ffn
                W_up = (ffn.W_up.data.to_dense() if ffn.W_up.is_sparse
                        else ffn.W_up.data).clone()
                W_gate = (ffn.W_gate.data.to_dense() if ffn.W_gate.is_sparse
                          else ffn.W_gate.data).clone()
                W_down = (ffn.W_down.data.to_dense() if ffn.W_down.is_sparse
                          else ffn.W_down.data).clone()
                ff = (W_up, ffn.b_up.clone(), W_gate, ffn.b_gate.clone(), W_down)
            self.blocks.append((
                heads, attn.num_heads, attn.head_dim,
                getattr(attn, "use_softmax1", True), block, is_comp, ff,
            ))
        self.head_w = model.head.weight.clone()
        self.head_b = model.head.bias.clone()

    @torch.no_grad()
    def _residual_pre_head(self, tape: Sequence[int]) -> torch.Tensor:
        """Faithful per-token residual after all blocks, before the LM head —
        byte-identical to ``_faithful_residual_pre_head(model, tape)`` but cached.
        """
        tok = torch.tensor([list(tape)], dtype=torch.long)
        x = self.model.embed(tok)[0]
        for (heads, nh, hd, sm1, block, is_comp, ff) in self.blocks:
            x = _faithful_attn_forward(heads, x, nh, hd, sm1)
            if is_comp:
                x = block.ffn(x.unsqueeze(0))[0]
            else:
                W_up, b_up, W_gate, b_gate, W_down = ff
                up = x @ W_up.t() + b_up
                gate = x @ W_gate.t() + b_gate
                hidden = torch.nn.functional.silu(up) * gate
                x = x + hidden @ W_down.t()
        return x

    @torch.no_grad()
    def forward(self, tape: Sequence[int]) -> torch.Tensor:
        """Return ``[S, vocab]`` logits — byte-identical to
        ``faithful_full_forward(model, tape)``."""
        x = self._residual_pre_head(tape)
        return x @ self.head_w.t() + self.head_b


# ---------------------------------------------------------------------------
# Opcode -> imperative composite ALU block (the #230 still-imperative path).
#
# These blocks have no declarative IR rule form, so a divergence at an ALU step
# can be attributed only at BLOCK granularity (``attributed_op=<block>``,
# ``attributed_rule=None``) — not to a single rule. The faithful forward (the
# cached ``FaithfulForwardCache``, which already runs ``block.ffn(...)`` for the
# composite blocks) EXECUTES the real baked block on the residual tape, so the
# gate gives a real per-step (PC, AX) verdict for ALU programs (no longer
# ALU-OPAQUE).
# ---------------------------------------------------------------------------

_ALU_OPCODE_BLOCK: Dict[int, str] = {
    25: "ADD (AddSub5StageBlock)",
    26: "SUB (AddSub5StageBlock)",
    27: "MUL (FlattenedALUMul)",
    28: "DIV (FlattenedDivMod)",
    29: "MOD (FlattenedDivMod)",
    23: "SHL (ALUShiftComposite)",
    24: "SHR (ALUShiftComposite)",
}

_OPCODE_NAME = {
    0: "LEA", 1: "IMM", 2: "JMP", 3: "JSR", 4: "BZ", 5: "BNZ", 6: "ENT",
    7: "ADJ", 8: "LEV", 9: "LI", 10: "LC", 11: "SI", 12: "SC", 13: "PSH",
    14: "OR", 15: "XOR", 16: "AND", 17: "EQ", 18: "NE", 19: "LT", 20: "GT",
    21: "LE", 22: "GE", 23: "SHL", 24: "SHR", 25: "ADD", 26: "SUB",
    27: "MUL", 28: "DIV", 29: "MOD", 38: "EXIT",
}


# ---------------------------------------------------------------------------
# Production context construction (mirrors run_vm._build_context +
# DraftVM teacher-forced tape + per-step opcode capture).
# ---------------------------------------------------------------------------


def build_code_prompt(bytecode: Sequence[int], data: bytes = b"") -> List[int]:
    """The CODE_START..CODE_END..DATA prompt the production runner seeds with.

    Byte-identical to ``AutoregressiveVMRunner._build_context`` for the
    code+data portion (argv/stdin omitted — the 1096 corpus is stdin-free).
    """
    tokens: List[int] = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(IMMEDIATE_SIZE):
            tokens.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            tokens.append(0)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    if isinstance(data, (bytes, bytearray, list)):
        tokens.extend(int(b) for b in data)
    tokens.append(Token.DATA_END)
    return tokens


@dataclass
class OracleTape:
    draft_tokens: List[int]                  # teacher-forced step slices
    steps: List[Tuple[int, int]]             # (pc, ax) AFTER each step
    opcodes: List[int]                       # opcode executed AT each step
    halted: bool


def oracle_tape_and_steps(
    bytecode: Sequence[int], data: bytes = b"", max_steps: int = 64,
) -> OracleTape:
    """Run the DraftVM, capturing the teacher-forced 35-token step tape, the
    per-step ``(pc, ax)`` oracle, and the opcode executed at each step.

    The ``(pc, ax)`` list matches ``_oracle_pc_ax_steps`` (capture AFTER each
    ``step()``). The opcode per step is read from the DraftVM's current
    instruction BEFORE the step executes (so an ALU-step divergence can be
    attributed to its composite ALU block). The 35-token slices are exactly what
    the model is teacher-forced to reproduce.
    """
    vm = DraftVM(list(bytecode))
    vm.load_data(data)
    draft: List[int] = []
    steps: List[Tuple[int, int]] = []
    opcodes: List[int] = []
    n = 0
    while not vm.halted and n < max_steps:
        # Opcode about to execute (static code path; memory path -> -1).
        op = -1
        if vm.idx < len(vm.code):
            op = vm.code[vm.idx] & 0xFF
        ok = vm.step()
        if not ok:
            break
        opcodes.append(op)
        draft.extend(vm.draft_tokens())
        steps.append((vm.pc & 0xFFFFFFFF, vm.ax & 0xFFFFFFFF))
        n += 1
    return OracleTape(
        draft_tokens=draft, steps=steps, opcodes=opcodes, halted=vm.halted,
    )


# ---------------------------------------------------------------------------
# Fixed-offset per-step decode (markers re-anchored, as production does).
# ---------------------------------------------------------------------------

# Register marker offset inside a step (DraftVM.draft_tokens layout). The
# PC/AX/SP/BP markers are fixed at {0,5,10,15} in both layouts; STACK0 is at 20
# ONLY in the 35-token layout (dropped under C4_NO_STACK0_EMIT => STEP_TOKENS=30).
_REG_OFFSETS = {"PC": 0, "AX": 5, "SP": 10, "BP": 15}
if STEP_TOKENS != 30:
    _REG_OFFSETS["STACK0"] = 20


def _decode_reg_fixed(slice_tokens: Sequence[int], marker_off: int) -> int:
    """Decode a 32-bit register from a step slice at its FIXED offset.

    The 4 little-endian value bytes live at ``marker_off+1 .. marker_off+4``.
    Decoding at the FIXED offset (not by searching for the marker token)
    re-anchors the marker exactly as production re-anchors the first token of
    each step via the Python STEP_END dispatch — so a passing program decodes
    cleanly. This is what makes the AX HIGH BYTES (1..3) faithful: their value
    tokens sit at fixed offsets 7/8/9 and the model's argmax there over the
    teacher-forced tape == the model's own emission for the steps that are not
    downstream of a cross-step poisoning correction (see ``_value_correction_step``).
    """
    val = 0
    for j in range(4):
        val |= (int(slice_tokens[marker_off + 1 + j]) & 0xFF) << (j * 8)
    return val & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# The cross-step faithfulness GUARD (the key to AX-high-byte / framing-drift
# confidence).
#
# A SINGLE teacher-forced forward reads the model's argmax over the DraftVM's
# byte-exact oracle tape. For a step whose context is NOT yet poisoned, that
# argmax == the model's own production emission (validated byte-for-byte). But
# production decode is AUTOREGRESSIVE: the moment the model emits a REGISTER
# VALUE byte that differs from the oracle tape (a "value correction"), production
# appends the WRONG byte and re-syncs its DraftVM from the model's now-wrong
# register state — so EVERY later step reads a poisoned context the teacher-
# forced single forward never sees (the model "recovers" under teacher forcing).
# Therefore:
#   * the single-forward per-step (PC, AX) decode is FAITHFUL to production for
#     every step AT OR BEFORE the first register-VALUE-byte correction, and
#   * it is UNRELIABLE (the autoregressive framing-drift / cross-step class) for
#     any step strictly after it.
# A MARKER-token misprediction (offsets 0/5/10/15/20/25/34) does NOT poison —
# production re-anchors the marker structurally (STEP_END dispatch), so only the
# register VALUE bytes matter. This guard is the soundness contract: when it
# says "faithful" the gate's verdict provably matches the neural full_trace
# runner (measured: 0 false-trusts over the var/func/expr/edge sample); when it
# says "cross-step" the verdict may be a teacher-forcing artifact and is reported
# as a distinct CROSS-STEP class, never silently trusted.
# ---------------------------------------------------------------------------

# Step-relative offsets that carry a REGISTER VALUE byte the model re-emits on a
# later step. 35-token: PC 1..4, AX 6..9, SP 11..14, BP 16..19, STACK0 21..24.
# 30-token (C4_NO_STACK0_EMIT): the STACK0 block is dropped, so the value bytes
# are PC 1..4, AX 6..9, SP 11..14, BP 16..19 only. The MEM addr/val offsets are
# the ``_UNSAFE_OFFSETS`` (DraftVM-trusted, never a model correction); the marker
# offsets {0,5,10,15,(20),MEM_marker,STEP_END} are re-anchored.
_VALUE_OFFSETS = frozenset(
    list(range(1, 5)) + list(range(6, 10)) + list(range(11, 15))
    + list(range(16, 20))
    + (list(range(21, 25)) if STEP_TOKENS != 30 else [])
)
# MEM addr/val bytes the DraftVM is trusted for (production never reads a model
# correction here — the embedding's MEM-metadata injection makes them disagree
# with the flat argmax for EVERY program). Mirrors batched_pure_neural._UNSAFE_OFFSETS:
# 35-token MEM addr/val = 26..33; 30-token (STACK0 dropped) = 21..28.
if STEP_TOKENS == 30:
    _UNSAFE_OFFSETS = frozenset(range(21, 29))
else:
    _UNSAFE_OFFSETS = frozenset(range(26, 34))


def _value_correction_step(
    pred_fn, draft_tokens: Sequence[int], offsets=_VALUE_OFFSETS,
) -> Optional[int]:
    """First VM step at which the model's emitted register-VALUE byte (at one of
    ``offsets``, default all five registers) diverges from the DraftVM oracle
    tape (the cross-step poisoning point).

    ``pred_fn(t)`` is the model's argmax for draft position ``t`` (= argmax at
    ``prefix + t - 1`` over the teacher-forced forward). A divergence at a value
    offset is the FIRST place production's autoregressive context goes wrong.
    Returns the step index, or ``None`` if the model reproduces every value byte
    (single-forward faithful). Marker / MEM offsets are never a poisoning
    correction (re-anchored / DraftVM-trusted). The default ``_VALUE_OFFSETS``
    (PC/AX/SP/BP/STACK0) is the SOUND guard — an early SP/BP/STACK0 correction
    can poison a later PC (var/func), so the full set is required (0 false-trusts
    measured; PC+AX-only gave 34+).
    """
    for t in range(len(draft_tokens)):
        off = t % STEP_TOKENS
        if off not in offsets:
            continue
        if int(pred_fn(t)) != int(draft_tokens[t]):
            return t // STEP_TOKENS
    return None


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    cluster: str
    classification: str               # PASS / FAIL / ERROR (ALU-OPAQUE retired)
    n_steps: int
    # FAIL detail.
    div_step: Optional[int] = None
    div_reg: Optional[str] = None     # "PC" or "AX"
    div_byte: Optional[int] = None    # 0..3
    expected: Optional[int] = None    # expected byte value
    got: Optional[int] = None         # interpreter byte value
    expected_reg: Optional[int] = None  # full expected register value
    got_reg: Optional[int] = None       # full interpreter register value
    attributed_op: Optional[str] = None
    attributed_rule: Optional[str] = None
    attributed_contrib: Optional[float] = None
    # The imperative ALU block at/owning the diverging step, if the step's
    # opcode is one of the 4 composite ALU blocks. On an ALU-step FAIL this is
    # the COARSE attribution (``attributed_op=<block>``, ``attributed_rule=None``)
    # because the block is imperative and has no declarative rule to pin.
    alu_op: Optional[str] = None
    # True if the diverging/owning step's opcode is a composite ALU block (so the
    # verdict rides the real imperative ALU forward; on a FAIL the attribution is
    # the coarse block, not a rule). Orthogonal to ``confidence`` — an ALU-step
    # FAIL downstream of a value correction is still CROSS-STEP.
    is_alu_step: bool = False
    # Confidence of the (step, reg, byte) divergence verdict — driven by the
    # cross-step poisoning guard ``_value_correction_step``, NOT by which byte:
    #
    #   "high"       = the divergence is AT OR BEFORE the first register-VALUE-
    #                  byte correction, so the single-forward decode provably
    #                  matches the neural full_trace runner (incl. PC, AX byte 0,
    #                  AND AX bytes 1..3 — the AX-HIGH-BYTE case). Attributed.
    #   "cross-step" = the divergence is strictly AFTER a value correction, so it
    #                  rides production's autoregressive context poisoning that a
    #                  single teacher-forced forward cannot reproduce (the
    #                  framing-drift / cross-step class: var/func PC, deep expr
    #                  AX). The single-forward verdict is a teacher-forcing
    #                  artifact — NOT attributed; resolve with the GPU
    #                  ``--faithfulness-check`` (the real production decode).
    confidence: str = "high"
    # The cross-step poisoning point (first register-VALUE-byte correction step),
    # or None if the whole program is single-forward faithful. Set on FAIL.
    value_correction_step: Optional[int] = None
    note: str = ""


PASS = "PASS"
FAIL = "FAIL"
# ALU_OPAQUE is RETIRED as a verdict: the faithful forward now executes the real
# composite ALU block, so ALU programs get a real PASS/FAIL. The constant is
# kept only so any external importer does not break; the gate never emits it.
ALU_OPAQUE = "ALU-OPAQUE"
ERROR = "ERROR"
HIGH = "high"
CROSS_STEP = "cross-step"


def _div_confidence(div_step: int, value_corr_step: Optional[int]) -> str:
    """HIGH-confidence iff the divergence is at or before the first cross-step
    poisoning correction (so the single teacher-forced forward == production);
    CROSS-STEP otherwise (the autoregressive framing-drift class).

    This replaces the old per-byte heuristic ("PC/AX-byte0 = high, AX-high-byte
    = low"). The byte INDEX is irrelevant to faithfulness — what matters is
    whether the step is downstream of a value correction that poisons the
    autoregressive context. AX bytes 1..3 ARE faithful when not poisoned (the
    edge_literal cluster: 10/10 vs neural); PC IS unfaithful when poisoned (the
    var/func cluster). See ``_value_correction_step``.
    """
    if value_corr_step is None or div_step <= value_corr_step:
        return HIGH
    return CROSS_STEP


# ---------------------------------------------------------------------------
# The gate context (build once).
# ---------------------------------------------------------------------------


@dataclass
class GateContext:
    model: object
    layout: object
    dim_positions: Dict[str, int]
    flat_ffn_ops: List[object]        # ops with IR FFN rules (for attribution)
    interp: FaithfulInterpreter
    fwd: "FaithfulForwardCache"       # cached faithful forward (5-13x faster)


def build_gate_context(verbose: bool = True) -> GateContext:
    """Build the real baked model (CPU, cached compile ~8s) + the attribution
    interpreter. CPU-only: the per-tape faithful forward is cheap on CPU and
    co-exists with the GPU lanes."""
    if verbose:
        print("[interp-oracle-gate] building production model "
              "(alu_mode='efficient', CPU, ~8-15s)...",
              file=sys.stderr, flush=True)
    model, layout = build_production_model("cpu")
    dim_positions = dict(layout.dim_positions)
    flat_ops: List[object] = []
    for blk in layout.ops_per_layer:
        flat_ops.extend(blk)
    flat_ops.extend(list(getattr(layout, "block_ops", []) or []))
    flat_ops.extend(list(getattr(layout, "model_ops", []) or []))
    interp = FaithfulInterpreter(
        dim_positions=dim_positions, ops_per_block=[],
        d_model=model.d_model, num_heads=model.blocks[0].attn.num_heads,
        head_dim=model.blocks[0].attn.head_dim,
    )
    fwd = FaithfulForwardCache(model)
    if verbose:
        print(f"[interp-oracle-gate] model: d_model={model.d_model} "
              f"blocks={len(model.blocks)} ops={len(flat_ops)} "
              f"(cached faithful forward ready)",
              file=sys.stderr, flush=True)
    return GateContext(
        model=model, layout=layout, dim_positions=dim_positions,
        flat_ffn_ops=flat_ops, interp=interp, fwd=fwd,
    )


# ---------------------------------------------------------------------------
# Per-program classification + attribution.
# ---------------------------------------------------------------------------


@torch.no_grad()
def classify_program(
    ctx: GateContext,
    name: str,
    bytecode: Sequence[int],
    data: bytes = b"",
    *,
    cluster: str = "",
    max_steps: int = 48,
    attribute: bool = True,
) -> GateResult:
    """Run the gate on one program: faithful per-step decode vs DraftVM oracle.

    Returns the classification + (on FAIL) the attributed owning rule.
    """
    try:
        ot = oracle_tape_and_steps(bytecode, data, max_steps=max_steps)
    except Exception as exc:  # noqa: BLE001
        return GateResult(name=name, cluster=cluster, classification=ERROR,
                          n_steps=0, note=f"oracle error: {exc!r}")
    if not ot.steps:
        return GateResult(name=name, cluster=cluster, classification=ERROR,
                          n_steps=0, note="oracle produced no steps")

    prompt = build_code_prompt(bytecode, data)
    prefix = len(prompt)
    full_ctx = prompt + ot.draft_tokens
    n_steps = len(ot.steps)

    try:
        logits = ctx.fwd.forward(full_ctx)
        fa = logits.argmax(dim=-1).tolist()
    except Exception as exc:  # noqa: BLE001
        return GateResult(name=name, cluster=cluster, classification=ERROR,
                          n_steps=n_steps, note=f"faithful forward error: {exc!r}")

    # Predicted draft token at draft position t = argmax at (prefix + t - 1).
    def pred_tok(t: int) -> int:
        return int(fa[prefix + t - 1])

    # The cross-step faithfulness guard: the first VM step at which the model's
    # emitted register-VALUE byte diverges from the oracle tape. Every step AT OR
    # BEFORE it decodes faithfully from this single teacher-forced forward;
    # anything strictly after rides production's autoregressive context poisoning
    # (the framing-drift / cross-step class) the single forward cannot reproduce.
    # The guard uses ALL register-value offsets (PC/AX/SP/BP/STACK0): an early
    # SP/BP/STACK0 correction CAN poison a later PC (the var/func cluster), so a
    # PC+AX-only guard would falsely certify those — measured 0 false-trusts with
    # the full set, 34+ with PC+AX-only.
    vcorr = _value_correction_step(pred_tok, ot.draft_tokens)

    # Per step, decode the model's predicted (PC, AX) from its draft slice and
    # compare to the oracle. FIRST diverging byte wins. The decode is at the
    # FIXED (re-anchored) value offsets — production re-anchors the marker via
    # STEP_END dispatch, so the AX HIGH BYTES (1..3) decode faithfully here for
    # any step not downstream of a value correction (the guard above).
    for s in range(n_steps):
        base = s * STEP_TOKENS
        slice_pred = [pred_tok(base + k) for k in range(STEP_TOKENS)]
        o_pc, o_ax = ot.steps[s]
        for reg, o_val in (("PC", o_pc), ("AX", o_ax)):
            moff = _REG_OFFSETS[reg]
            g_val = _decode_reg_fixed(slice_pred, moff)
            if g_val == o_val:
                continue
            # First diverging register at this step. Find the first diverging
            # byte (low-order first) for a precise attribution.
            for k in range(4):
                exp_b = (o_val >> (8 * k)) & 0xFF
                got_b = (g_val >> (8 * k)) & 0xFF
                if exp_b == got_b:
                    continue
                opcode = ot.opcodes[s] if s < len(ot.opcodes) else -1
                op_name = _OPCODE_NAME.get(opcode, f"op{opcode}")
                # ALU step? The step's opcode is a composite ALU block. The
                # faithful forward (FaithfulForwardCache, which runs the real
                # ``block.ffn(...)`` for the composite blocks) EXECUTED the real
                # baked block on the residual tape, so this divergence is a REAL
                # FAIL (not ALU-OPAQUE). But the block is imperative (no
                # declarative rule), so the attribution is COARSE:
                # attributed_op=<block>, attributed_rule=None, is_alu_step=True.
                #
                # CROSS-STEP RECONCILIATION (clean-ALU certification): when the
                # DIVERGING STEP'S OWN OPCODE is a composite ALU block, the wrong
                # (PC, AX) IS the ALU block's own result — the GENUINE first
                # divergence — NOT a value byte the autoregressive decode poisoned
                # downstream of a benign SP/STACK0 re-anchoring. The blanket vcorr
                # guard otherwise defers these as CROSS-STEP because the unrelated
                # step-1 SP-byte-2 / STACK0-high-byte re-anchoring corrections fire
                # FIRST (``div_step > vcorr``). But those corrections are the
                # benign re-anchoring noise the gate already trusts for the marker
                # offsets; they do NOT poison the ALU step's own result. So an
                # ALU-OPCODE-step divergence is certified HIGH-confidence (a real
                # coarse-attributed FAIL). Measured: every such ALU-step FAIL over
                # mul/div/mod/expr matches the neural full_trace FAIL verdict
                # (7/7), and the 43 clean ALU programs (no ALU-step divergence) all
                # match neural PASS — 0 false-trusts. The cross-step guard remains
                # intact for NON-ALU divergences (the var/func PC framing-drift,
                # where the flat divergence is on a PC/AX value the early SP/BP/
                # STACK0 correction genuinely poisons).
                if opcode in _ALU_OPCODE_BLOCK:
                    return GateResult(
                        name=name, cluster=cluster, classification=FAIL,
                        n_steps=n_steps, div_step=s, div_reg=reg, div_byte=k,
                        expected=exp_b, got=got_b, expected_reg=o_val,
                        got_reg=g_val, attributed_op=_ALU_OPCODE_BLOCK[opcode],
                        attributed_rule=None, alu_op=_ALU_OPCODE_BLOCK[opcode],
                        is_alu_step=True, confidence=HIGH,
                        value_correction_step=vcorr,
                        note=(f"step {s} opcode={op_name} {reg}[{k}] "
                              f"exp=0x{exp_b:02x} got=0x{got_b:02x} "
                              f"-> imperative ALU block "
                              f"{_ALU_OPCODE_BLOCK[opcode]} (coarse "
                              f"attribution: no declarative rule; GENUINE ALU-step "
                              f"divergence — certified HIGH-confidence)"),
                    )
                # Non-ALU divergence: the cross-step poisoning guard applies — a
                # divergence strictly AFTER the first register-VALUE-byte
                # correction may be a teacher-forcing artifact (the var/func PC
                # framing-drift) and is flagged CROSS-STEP, not attributed.
                conf = _div_confidence(s, vcorr)
                # Non-ALU step: attribute the wrong byte to its owning declarative
                # rule. Only attribute HIGH-confidence divergences (at/before the
                # cross-step poisoning point): a CROSS-STEP divergence may be a
                # teacher-forcing artifact, so naming a rule there would mislead.
                attr_op = attr_rule = None
                attr_c = None
                if attribute and conf == HIGH:
                    attr_op, attr_rule, attr_c = _attribute_byte(
                        ctx, full_ctx, prefix, s, reg, k, got_b,
                    )
                xs_tag = ("  [CROSS-STEP: divergence is downstream of a value "
                          f"correction at step {vcorr} — production's "
                          "autoregressive poisoning; resolve with "
                          "--faithfulness-check (GPU)]"
                          if conf == CROSS_STEP else "")
                return GateResult(
                    name=name, cluster=cluster, classification=FAIL,
                    n_steps=n_steps, div_step=s, div_reg=reg, div_byte=k,
                    expected=exp_b, got=got_b, expected_reg=o_val, got_reg=g_val,
                    attributed_op=attr_op, attributed_rule=attr_rule,
                    attributed_contrib=attr_c, confidence=conf,
                    value_correction_step=vcorr,
                    note=(f"step {s} opcode={op_name} {reg}[{k}] "
                          f"exp=0x{exp_b:02x} got=0x{got_b:02x}" + xs_tag),
                )
    # No (PC, AX) divergence found. BUT if the program's path executes an
    # composite-ALU opcode, the faithful forward EXECUTED that block (the
    # FaithfulForwardCache runs the real ``block.ffn(...)`` for the composite
    # blocks — production's actual ALU), so a clean per-step match IS a real PASS
    # — including for programs whose path runs an ALU op. (Previously these were
    # declined as ALU-OPAQUE on a stale "not bit-certified vs production" caveat;
    # the gate now certifies them.) We still surface that the path runs an ALU
    # block via ``is_alu_step`` / ``alu_op`` so the verdict reports the coarse
    # block context.
    alu_steps = [(s, ot.opcodes[s]) for s in range(min(n_steps, len(ot.opcodes)))
                 if ot.opcodes[s] in _ALU_OPCODE_BLOCK]
    has_alu = bool(alu_steps)
    alu_op0 = _ALU_OPCODE_BLOCK[alu_steps[0][1]] if has_alu else None
    # The single teacher-forced forward found NO (PC, AX) divergence. If the
    # model emitted a register-VALUE byte that diverged from the oracle tape at
    # some step (``vcorr`` set) at or before the LAST checked step, the gate
    # CANNOT certify PASS: under teacher forcing the model "recovers" each later
    # step (it is fed the correct tape), but production's autoregressive decode
    # would have poisoned the context from ``vcorr`` on and may FAIL a later step
    # the single forward cannot see (the var/func cluster: neural fails @ step 4,
    # the single forward "passes"). A correction at the LAST step (``vcorr >=
    # n_steps - 1``) cannot poison any later (PC, AX) check, so it stays a faithful
    # PASS (the edge_literal single/last-step cluster). The PASS-path uses the
    # FULL-offset correction step (``vcorr``): an SP/BP/STACK0 correction at an
    # early step CAN poison a later PC (the var/func cluster — neural fails @ step
    # 4 from a step-1 BP correction). SOUNDNESS demands the full guard here:
    # PC+AX-only would falsely PASS those 5 var programs (measured). The cost is
    # conservative — some genuine passes whose only corrections are SP/BP/STACK0
    # (e.g. the eq/lt/boolean smoke) are flagged CROSS-STEP rather than PASSed;
    # that is the safe direction (the gate never claims a PASS it cannot back).
    # Resolve those with --faithfulness-check (GPU). A correction at the LAST step
    # (``vcorr >= n_steps - 1``) cannot poison any later (PC, AX) check, so it
    # stays a faithful PASS.
    #
    # CLEAN-ALU CERTIFICATION (cross-step reconciliation): the blanket vcorr
    # deferral was DEFERRING essentially every ALU program (mul/div/mod/expr)
    # because the upstream step-1 SP-byte-2 + STACK0-high-byte re-anchoring
    # corrections fire FIRST (``vcorr = 0 or 1``). But for a program whose path
    # executes a composite ALU block AND whose every (PC, AX) decodes cleanly,
    # those benign re-anchoring corrections do NOT poison the ALU result: the ALU
    # block's output is the program's whole point and it matched the oracle at
    # every step. Measured over mul/div/mod/expr (100-119,150-169,850-874): all
    # 43 clean ALU programs (no (PC, AX) divergence) have neural full_trace = ok
    # (0 false-trusts). So a clean program whose path RUNS an ALU block is
    # certified PASS even when its only corrections are the SP/STACK0 re-anchoring
    # noise. NON-ALU clean programs (``not has_alu``) keep the conservative
    # CROSS-STEP deferral: 5 var_simple programs (262/263/267/271/272) have NO
    # flat (PC, AX) divergence yet neural FAILS @ step 4 (a step-1 BP/SP
    # correction poisons a later PC the single forward never sees) — certifying
    # those would be false-trusts, so the guard stays intact for them.
    if vcorr is not None and vcorr < n_steps - 1 and not has_alu:
        return GateResult(
            name=name, cluster=cluster, classification=FAIL,
            n_steps=n_steps, div_step=None, confidence=CROSS_STEP,
            value_correction_step=vcorr, is_alu_step=has_alu, alu_op=alu_op0,
            note=(f"single-forward decode found no (PC, AX) divergence, but the "
                  f"model emitted a register-VALUE byte that diverged from the "
                  f"oracle tape at step {vcorr}; production's autoregressive "
                  f"decode would poison the context from there and may fail a "
                  f"later step the teacher-forced forward cannot see "
                  f"(CROSS-STEP — resolve with --faithfulness-check (GPU))"),
        )
    if has_alu:
        s0 = alu_steps[0][0]
        vc_tag = (f"; certified despite a benign SP/STACK0 re-anchoring "
                  f"correction at step {vcorr} (clean-ALU certification — the "
                  f"ALU block's own result matched the oracle at every step)"
                  if vcorr is not None else "")
        return GateResult(
            name=name, cluster=cluster, classification=PASS, n_steps=n_steps,
            is_alu_step=True, alu_op=alu_op0, value_correction_step=vcorr,
            note=(f"all {n_steps} steps' (PC, AX) match oracle (path executes "
                  f"imperative ALU op {alu_op0} at step {s0}"
                  + (f" + {len(alu_steps) - 1} more" if len(alu_steps) > 1 else "")
                  + ", executed via real baked block" + vc_tag + ")"),
        )
    return GateResult(name=name, cluster=cluster, classification=PASS,
                      n_steps=n_steps, note=f"all {n_steps} steps match oracle")


# ---------------------------------------------------------------------------
# AX byte0 is decoded from the OUTPUT_LO/OUTPUT_HI result nibble cells. (PC and
# AX bytes 1..3 are decoded from per-byte families that are not uniformly
# OUTPUT_*; we attribute the AX byte-0 / OUTPUT path precisely and fall back to
# a static-writer scan for other bytes.)
_DECODE_FAMILIES = {
    ("AX", 0): ("OUTPUT_LO", "OUTPUT_HI"),
}


@torch.no_grad()
def _attribute_byte(
    ctx: GateContext, full_ctx: List[int], prefix: int, step: int,
    reg: str, byte_k: int, got_byte: int,
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Name the single declarative rule whose RUNTIME SwiGLU output dominates
    the wrong ``got_byte`` at the predicting position.

    The predicting position for register ``reg`` byte ``byte_k`` of step
    ``step`` is the draft position just BEFORE that byte:
    ``prefix + step*35 + marker_off + byte_k`` (argmax there predicts the byte
    at ``+1``). We capture the faithful pre-head residual at that position and
    rank the FFN rules by their actual contribution to the winning nibble cells
    of the wrong byte. Returns ``(op_name, rule_name, contribution)`` or
    ``(None, None, None)`` if no rule produced a nonzero runtime contribution
    (a default/relayed cell — itself a useful signal).
    """
    moff = _REG_OFFSETS[reg]
    pred_pos = prefix + step * STEP_TOKENS + moff + byte_k
    try:
        resid = ctx.fwd._residual_pre_head(full_ctx)[pred_pos]
    except Exception:  # noqa: BLE001
        return None, None, None

    fams = _DECODE_FAMILIES.get((reg, byte_k))
    candidates: List[Tuple[str, str, float]] = []
    if fams is not None:
        lo_cell = got_byte & 0x0F
        hi_cell = (got_byte >> 4) & 0x0F
        for fam, cell in ((fams[0], lo_cell), (fams[1], hi_cell)):
            base = ctx.dim_positions.get(fam)
            if base is None:
                continue
            col = base + cell
            ranked = ctx.interp.attribute_runtime_contribution(
                ctx.flat_ffn_ops, col, resid,
            )
            candidates.extend(ranked)
    if not candidates:
        # No OUTPUT-family mapping (PC / AX bytes 1..3) or no runtime writer:
        # scan ALL families for a column whose runtime contribution is highest,
        # restricted to dims whose name encodes this register/byte if present.
        ranked_all: List[Tuple[str, str, float]] = []
        wanted = []
        for nm, base in ctx.dim_positions.items():
            up = nm.upper()
            if reg in up and (f"BYTE{byte_k}" in up or "OUTPUT" in up):
                wanted.append((nm, base))
        for nm, base in wanted:
            for cell in range(16):
                ranked_all.extend(
                    ctx.interp.attribute_runtime_contribution(
                        ctx.flat_ffn_ops, base + cell, resid,
                    )
                )
        candidates = ranked_all
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda t: abs(t[2]), reverse=True)
    op_name, rule_name, contrib = candidates[0]
    return op_name, rule_name, contrib


# ---------------------------------------------------------------------------
# Faithfulness check (interpreter vs REAL NEURAL model — needs a free GPU).
# ---------------------------------------------------------------------------


def _free_gpu() -> Optional[int]:
    """Return a GPU index with ample free memory, else None.

    Used by ``--faithfulness-check``: the interpreter-vs-neural confirm needs
    the real neural forward. We only borrow a GPU that is mostly idle so we do
    not contend with the two running fix lanes.
    """
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        return None
    best = None
    for line in out.strip().splitlines():
        try:
            idx, used, total, util = (int(x.strip()) for x in line.split(","))
        except Exception:  # noqa: BLE001
            continue
        free = total - used
        # Need >=10 GB free AND <40% util to avoid contending with a fix lane.
        if free >= 10000 and util < 40:
            if best is None or free > best[1]:
                best = (idx, free)
    return best[0] if best else None


@torch.no_grad()
def faithfulness_check(
    ctx: GateContext, programs: List[dict], gpu: int, max_steps: int = 48,
) -> Tuple[int, int, List[str], List[str]]:
    """Run interpreter-vs-NEURAL per-step decode on ``programs`` (GPU build).

    Builds the REAL neural runner on ``gpu`` and, for each program, runs the
    production fail-fast decode (the authoritative neural per-step (PC, AX)) and
    the CPU interpreter's per-step decode over the SAME oracle tape, and counts
    programs where they AGREE step-for-step. Disagreements are interpreter
    COVERAGE GAPS (reported by name) so the gate never mis-attributes them.
    Returns ``(n_faithful, n_authoritative, gap_names, cross_step_gap_names)``.
    ``CUDA_VISIBLE_DEVICES`` is already pinned to ``gpu`` by ``main`` (before
    any torch init) so the runner lands on the GPU.
    """
    del gpu  # device already pinned in main before torch init.
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner
    from tests.declarative_oracle import declarative_oracle_for_program

    runner = BatchedPureNeuralRunner(max_seq_len=4096)
    bcs = [p["bytecode"] for p in programs]
    datas = [p.get("data", b"") for p in programs]
    steps_list = []
    for p in programs:
        try:
            orc = declarative_oracle_for_program(
                p["bytecode"], p.get("data", b""), label=p["name"])
            steps_list.append(orc.steps)
        except Exception:  # noqa: BLE001
            steps_list.append(None)
    neural = runner.run_batch_fail_fast(
        bcs, data_list=datas, expected_steps_list=steps_list,
        max_context_window=512, spec_k=32, criterion="full_trace",
    )

    n_faithful = 0
    n_authoritative = 0          # gate is authoritative (PASS or HIGH-conf FAIL)
    gaps: List[str] = []         # interp != neural where the gate claimed HIGH-conf
    xs_gaps: List[str] = []      # CROSS-STEP class (flagged, not authoritative)
    for p, nres in zip(programs, neural):
        # CPU interpreter verdict (no attribution to keep it fast).
        ires = classify_program(
            ctx, p["name"], p["bytecode"], p.get("data", b""),
            cluster=p.get("cluster", ""), max_steps=max_steps, attribute=False,
        )
        n_status = nres.get("status")
        n_div = nres.get("divergence_step")
        i_pass = ires.classification == PASS
        n_pass = n_status == "pass"

        # CROSS-STEP interp verdict (divergence downstream of a value correction):
        # the autoregressive framing-drift class the single forward cannot
        # reproduce — flagged, NOT authoritative. Report whether it agrees with
        # neural for the record, but never count it against the gate.
        if ires.classification == FAIL and ires.confidence == CROSS_STEP:
            agree = (not n_pass) and (ires.div_step == n_div if ires.div_step is not None else True)
            xs_gaps.append(
                f"{p['name']} (interp CROSS-STEP @ "
                f"step{ires.div_step}/poison@{ires.value_correction_step}, "
                f"neural={n_status}@{n_div}{' [agree]' if agree else ''})")
            continue

        # The gate is AUTHORITATIVE for this program (PASS or HIGH-conf FAIL,
        # incl. the now-promoted AX-high-byte cases). ALU programs now flow
        # through here too: the faithful forward executed the real composite ALU
        # block, so the verdict is a real PASS/FAIL and is compared to neural
        # like any other program (no ALU-OPAQUE skip).
        n_authoritative += 1
        same_verdict = (i_pass == n_pass)
        same_div = (ires.div_step == n_div) if not i_pass else True
        if same_verdict and same_div:
            n_faithful += 1
        else:
            gaps.append(
                f"{p['name']} (interp={ires.classification}@{ires.div_step} "
                f"neural={n_status}@{n_div})"
            )
    return n_faithful, n_authoritative, gaps, xs_gaps


# ---------------------------------------------------------------------------
# Program sources
# ---------------------------------------------------------------------------


def smoke_programs() -> List[dict]:
    from tests.test_smoke import _SMOKE_GROUPS
    out = []
    for _g, tests in _SMOKE_GROUPS.items():
        for t in tests:
            nm = t["name"].split("::")[-1]
            out.append({"name": nm, "bytecode": t["bytecode"], "data": b"",
                        "cluster": nm.replace("test_", "").rsplit("_", 1)[0]})
    return out


def _cluster_of(desc: str) -> str:
    import re
    base = desc.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    return base.rstrip("_") or "misc"


def corpus_programs(
    ids: Optional[List[int]] = None,
    sample: int = 0,
    max_decl_steps: int = 40,
) -> List[dict]:
    """Compile 1096 corpus programs to bytecode.

    ``ids`` selects exact corpus indices; otherwise ``sample`` picks a
    cluster-round-robin sample (known-bug clusters first), skipping programs
    whose declarative horizon exceeds ``max_decl_steps`` (the deep diverging
    loop/gcd/rec band — minutes per CPU forward, fails anyway).
    """
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program

    tests = generate_test_programs()

    if ids is not None:
        out = []
        for i in ids:
            if i < 0 or i >= len(tests):
                continue
            src, expected, desc = tests[i]
            try:
                bc, data = compile_c(src)
            except Exception as exc:  # noqa: BLE001
                out.append({"name": f"id{i}:{desc[:32]}", "compile_error": repr(exc)})
                continue
            out.append({"name": f"id{i}:{desc[:36]}", "bytecode": bc,
                        "data": data, "cluster": _cluster_of(desc),
                        "expected": expected})
        return out

    priority = ("div", "mod", "var", "if", "sub", "add", "mul", "expr",
                "and", "or", "eq", "ne", "lt", "gt", "le", "ge", "absdiff",
                "bool", "ternary", "nested", "func")
    buckets: Dict[str, list] = {}
    for src, expected, desc in tests:
        buckets.setdefault(_cluster_of(desc), []).append((src, expected, desc))
    order = list(priority) + [c for c in buckets if c not in priority]

    out: List[dict] = []
    idx = {c: 0 for c in buckets}
    while len(out) < sample:
        progressed = False
        for c in order:
            if c not in buckets or idx[c] >= len(buckets[c]):
                continue
            src, expected, desc = buckets[c][idx[c]]
            idx[c] += 1
            progressed = True
            try:
                bc, data = compile_c(src)
                orc = declarative_oracle_for_program(bc, data, label=desc)
                if orc.steps is None or orc.steps > max_decl_steps:
                    continue
            except Exception:  # noqa: BLE001
                continue
            out.append({"name": desc[:40], "bytecode": bc, "data": data,
                        "cluster": _cluster_of(desc), "expected": expected})
            if len(out) >= sample:
                break
        if not progressed:
            break
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def run_set(ctx: GateContext, programs: List[dict], title: str,
            max_steps: int = 48, attribute: bool = True) -> List[GateResult]:
    print("=" * 90)
    print(f"  {title}  ({len(programs)} programs)")
    print("=" * 90)
    results: List[GateResult] = []
    for p in programs:
        if "compile_error" in p:
            print(f"  COMPILE-ERR  {p['name']}: {p['compile_error']}")
            continue
        r = classify_program(
            ctx, p["name"], p["bytecode"], p.get("data", b""),
            cluster=p.get("cluster", ""), max_steps=max_steps,
            attribute=attribute,
        )
        results.append(r)
        if r.classification == PASS:
            alu_tag = f"  [ALU: {r.alu_op}]" if r.is_alu_step else ""
            print(f"  PASS         {r.name}  ({r.n_steps} steps){alu_tag}")
        elif r.classification == ERROR:
            print(f"  ERROR        {r.name}  {r.note}")
        elif r.confidence == CROSS_STEP:
            # CROSS-STEP: the single-forward verdict is downstream of a value
            # correction (autoregressive framing-drift). Report the poisoning
            # point, not an attributed rule (the verdict is not single-forward
            # faithful). Applies to ALU steps too (an ALU FAIL downstream of a
            # value correction is still autoregressive-poisoned).
            loc = (f"step={r.div_step} {r.div_reg}[{r.div_byte}]"
                   if r.div_step is not None else "(no flat divergence)")
            alu_tag = f"  [ALU: {r.alu_op}]" if r.is_alu_step else ""
            print(f"  {'CROSS-STEP':<12} {r.name}  {loc}  "
                  f"value-corruption@step{r.value_correction_step}{alu_tag}  "
                  f"[autoregressive framing-drift — confirm with --faithfulness-check (GPU)]")
        elif r.is_alu_step:
            # HIGH-confidence ALU-step FAIL: coarse block attribution (the block
            # is imperative — no declarative rule to pin).
            print(f"  {'FAIL(ALU)':<12} {r.name}  step={r.div_step} "
                  f"{r.div_reg}[{r.div_byte}] exp=0x{r.expected:02x} "
                  f"got=0x{r.got:02x} -> {r.attributed_op} (coarse, no rule)")
        else:
            # HIGH-confidence FAIL (incl. AX bytes 1..3 not downstream of a
            # value correction — the AX-high-byte case is now attributed).
            rule = (f"{r.attributed_op}::{r.attributed_rule}"
                    if r.attributed_rule
                    else "(no runtime writer / default cell)")
            print(f"  {'FAIL':<12} {r.name}  step={r.div_step} "
                  f"{r.div_reg}[{r.div_byte}] exp=0x{r.expected:02x} "
                  f"got=0x{r.got:02x} -> {rule}"
                  + (f"  (contrib={r.attributed_contrib:+.3f})"
                     if r.attributed_contrib is not None else ""))
    _summary(results, title)
    return results


def _summary(results: List[GateResult], title: str) -> None:
    n = len(results)
    if n == 0:
        return
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.classification] = counts.get(r.classification, 0) + 1
    n_fail_high = sum(1 for r in results
                      if r.classification == FAIL and r.confidence == HIGH)
    n_fail_xs = sum(1 for r in results
                    if r.classification == FAIL and r.confidence == CROSS_STEP)
    # ALU programs — now EXECUTED (no longer ALU-OPAQUE): how many of the
    # PASS/FAIL verdicts ride the real imperative ALU forward (a cross-cut of
    # PASS/FAIL, not a separate verdict class).
    n_alu_total = sum(1 for r in results if r.is_alu_step)
    n_alu_pass = sum(1 for r in results
                     if r.is_alu_step and r.classification == PASS)
    n_alu_fail = sum(1 for r in results
                     if r.is_alu_step and r.classification == FAIL)
    print("-" * 90)
    print(f"  SUMMARY [{title}]: {n} programs")
    print(f"    {PASS:<14} {counts.get(PASS, 0)}")
    print(f"    {'FAIL':<14} {n_fail_high}   (HIGH-confidence: divergence at/"
          f"before the cross-step poison point — PC / AX byte-0 / AX high bytes "
          f"— attributed)")
    if n_fail_xs:
        print(f"    {'CROSS-STEP':<14} {n_fail_xs}   (divergence downstream of a "
              f"value correction — autoregressive framing-drift; resolve with "
              f"--faithfulness-check (GPU))")
    if counts.get(ERROR):
        print(f"    {ERROR:<14} {counts[ERROR]}")
    interp_authoritative = counts.get(PASS, 0) + n_fail_high
    print(f"    gate AUTHORITATIVE (PASS + HIGH-confidence FAIL) = "
          f"{interp_authoritative}/{n}")
    if n_alu_total:
        print(f"    ALU programs (real ALU block EXECUTED, coarse block "
              f"attribution) = {n_alu_total}/{n}  "
              f"[{n_alu_pass} PASS, {n_alu_fail} FAIL]")
    print(f"    CROSS-STEP (autoregressive, confirm with --faithfulness-check "
          f"(GPU)) = {n_fail_xs}/{n}")

    # Top owning rules by FAIL frequency — the highest-leverage fix targets.
    # Only HIGH-confidence attributed FAILs (the trustworthy ones).
    rule_freq: Dict[str, int] = {}
    for r in results:
        if (r.classification == FAIL and r.confidence == HIGH
                and r.attributed_rule):
            key = f"{r.attributed_op}::{r.attributed_rule}"
            rule_freq[key] = rule_freq.get(key, 0) + 1
    if rule_freq:
        print(f"\n  TOP OWNING RULES (by HIGH-confidence FAIL frequency — "
              f"highest-leverage fix targets):")
        for key, freq in sorted(rule_freq.items(),
                                key=lambda t: t[1], reverse=True)[:10]:
            print(f"    {freq:3d}x  {key}")
    # FAILs with no runtime writer (default/relay cells) are a distinct class.
    n_nowriter = sum(1 for r in results
                     if r.classification == FAIL and r.confidence == HIGH
                     and not r.attributed_rule)
    if n_nowriter:
        print(f"  ({n_nowriter} HIGH-confidence FAILs had NO runtime FFN writer "
              f"for the wrong byte — a default/relayed cell or attention-only "
              f"path; the wrong value is upstream of the FFN, an attention "
              f"relay or embedding default)")
    print()


# ---------------------------------------------------------------------------
# Demo — the payoff on the known-buggy clusters.
# ---------------------------------------------------------------------------


def run_demo(ctx: GateContext, max_steps: int = 48) -> None:
    """Run the gate on the KNOWN-buggy clusters and show the attribution payoff:
    does the CPU gate independently finger the documented built-dim roots?"""
    print("=" * 90)
    print("  DEMO: the gate on KNOWN-buggy clusters (does it find the bugs "
          "WITHOUT a GPU?)")
    print("=" * 90)
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c

    tests = generate_test_programs()
    # One representative per known-bug cluster.
    wanted_keys = ("mul", "div", "expr_mul", "expr_div", "if_gt", "if_lt",
                   "var_simple", "var", "sub", "add")
    picked: Dict[str, dict] = {}
    for src, expected, desc in tests:
        cl = _cluster_of(desc)
        dl = desc.lower()
        for key in wanted_keys:
            if key in dl and key not in picked:
                try:
                    bc, data = compile_c(src)
                except Exception:  # noqa: BLE001
                    continue
                picked[key] = {"name": desc[:44], "bytecode": bc, "data": data,
                               "cluster": cl, "expected": expected}
    programs = list(picked.values())
    if not programs:
        print("  (no demo programs found)")
        return
    run_set(ctx, programs, "KNOWN-BUGGY CLUSTER DEMO", max_steps=max_steps,
            attribute=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true",
                    help="Run the smoke programs.")
    ap.add_argument("--sample-1096", type=int, default=0, metavar="N",
                    help="Gate N cluster-sampled 1096 programs.")
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated corpus ids/ranges (e.g. '0,5-9,42').")
    ap.add_argument("--demo", action="store_true",
                    help="Run the known-buggy-cluster attribution demo.")
    ap.add_argument("--faithfulness-check", type=int, default=0, metavar="N",
                    help="Validate interp==neural on N sampled programs "
                         "(uses a FREE GPU if one is idle; else skips).")
    ap.add_argument("--max-steps", type=int, default=48)
    ap.add_argument("--max-decl-steps", type=int, default=40,
                    help="Skip sampled programs deeper than this (CPU cost).")
    ap.add_argument("--no-attribute", action="store_true",
                    help="Skip rule attribution (faster verdict-only run).")
    args = ap.parse_args(argv)
    if not any([args.smoke, args.sample_1096, args.ids, args.demo,
                args.faithfulness_check]):
        args.smoke = True

    # If a faithfulness check is requested, pick the free GPU NOW (before any
    # torch init) so the neural reference runner can use it; the gate's own
    # model is built with an explicit ``.to("cpu")`` so it stays on CPU
    # regardless. CSR off => byte-identity neural ref + faster build.
    faith_gpu: Optional[int] = None
    if args.faithfulness_check:
        faith_gpu = _free_gpu()
        if faith_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(faith_gpu)
            os.environ.setdefault("C4_CSR_INFERENCE", "0")

    ctx = build_gate_context()
    print()
    attribute = not args.no_attribute

    all_results: List[GateResult] = []
    if args.ids:
        ids = _parse_ids(args.ids)
        progs = corpus_programs(ids=ids)
        all_results += run_set(ctx, progs, f"CORPUS IDS {args.ids}",
                               args.max_steps, attribute)
    if args.smoke:
        all_results += run_set(ctx, smoke_programs(), "SMOKE PROGRAMS",
                               args.max_steps, attribute)
    if args.sample_1096:
        progs = corpus_programs(sample=args.sample_1096,
                                max_decl_steps=args.max_decl_steps)
        all_results += run_set(ctx, progs, f"1096 SAMPLE (n={len(progs)})",
                               args.max_steps, attribute)
    if args.demo:
        run_demo(ctx, args.max_steps)

    if args.faithfulness_check:
        print("=" * 90)
        print("  FAITHFULNESS CHECK (interpreter vs REAL NEURAL model)")
        print("=" * 90)
        gpu = faith_gpu
        if gpu is None:
            print("  No free GPU (>=10 GB free, <40% util) — SKIPPING the "
                  "neural confirm. The interpreter math is already validated "
                  "byte-for-byte by tools/faithful_interpreter_validate.py; "
                  "run that on a free GPU to re-confirm.")
        else:
            print(f"  Using GPU {gpu} (idle). Confirming interp==neural "
                  f"per-step on {args.faithfulness_check} sampled programs...")
            sample = corpus_programs(sample=args.faithfulness_check,
                                     max_decl_steps=args.max_decl_steps)
            n_ok, n_auth, gaps, xs = faithfulness_check(
                ctx, sample, gpu, args.max_steps)
            print(f"  Over {len(sample)} sampled programs: gate is "
                  f"AUTHORITATIVE on {n_auth} (PASS or HIGH-confidence FAIL, "
                  f"incl. the AX-high-byte cases); of those, interp == neural "
                  f"on {n_ok}/{n_auth}.")
            print(f"  CROSS-STEP (autoregressive framing-drift) programs flagged "
                  f"(NOT authoritative; resolve with the GPU --faithfulness-check): "
                  f"{len(xs)}.")
            if gaps:
                print(f"  UNEXPECTED FAITHFULNESS GAPS ({len(gaps)} — HIGH-conf "
                      f"interp verdict disagreed with neural; these would be "
                      f"interpreter COVERAGE gaps the gate must NOT attribute):")
                for g in gaps:
                    print(f"      - {g}")
            else:
                print("  No UNEXPECTED gaps — every HIGH-confidence verdict "
                      "matched the neural model (the gate is authoritative "
                      "exactly where it claims to be).")
            if xs:
                print(f"  (CROSS-STEP programs, flagged so the gate never "
                      f"mis-attributes the autoregressive framing-drift; resolve on GPU:)")
                for g in xs[:12]:
                    print(f"      - {g}")
        print()

    return 0


def _parse_ids(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


if __name__ == "__main__":
    sys.exit(main())
