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
  * **PASS** — the interpreter's per-step ``(PC, AX)`` matches the oracle every
    step through HALT (same criterion as ``--criterion full_trace``).
  * **FAIL@step/byte + ATTRIBUTED RULE** — the first step+register+byte where the
    interpreter diverges from the oracle, AND the single declarative rule whose
    RUNTIME SwiGLU contribution dominates the wrong value at the predicting
    position (the structural attribution: not "which rule could write" but
    "which rule's actual output produced the wrong byte"). THIS IS THE KEY
    OUTPUT — the highest-leverage fix target.
  * **ALU-OPAQUE** — the divergence is at/after a step whose opcode is one of the
    4 still-imperative composite ALU blocks (ADD/SUB -> AddSub5StageBlock,
    MUL -> FlattenedALUMul, DIV/MOD -> FlattenedDivMod, SHL/SHR ->
    ALUShiftComposite). These have no IR-executable rule form, so the
    interpreter cannot run them from the IR and CANNOT attribute a rule — the
    program is flagged with the opaque ALU op (these need the imperative-forward
    GPU path), no rule attribution.

Faithfulness / trustworthiness
------------------------------
``--faithfulness-check`` (uses a GPU only if one is FREE; otherwise skips) runs
the same per-step decode with the REAL NEURAL model (not the oracle) on a sample
and confirms interpreter==neural step-for-step. Where they disagree, that is an
interpreter COVERAGE GAP (not a model bug) and is reported separately so the
gate never mis-attributes. The summary quantifies the corpus split:
INTERPRETABLE+faithful (gate is authoritative) vs ALU-OPAQUE vs faithfulness-gap.

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
# Opcode -> composite ALU block (the imperative #230 coverage gap).
# ---------------------------------------------------------------------------

_ALU_OPAQUE_OPCODE: Dict[int, str] = {
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
    flagged ALU-OPAQUE). The 35-token slices are exactly what the model is
    teacher-forced to reproduce.
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

# Register marker offset inside a 35-token step (DraftVM.draft_tokens layout).
_REG_OFFSETS = {"PC": 0, "AX": 5, "SP": 10, "BP": 15, "STACK0": 20}


def _decode_reg_fixed(slice_tokens: Sequence[int], marker_off: int) -> int:
    """Decode a 32-bit register from a step slice at its FIXED offset.

    The 4 little-endian value bytes live at ``marker_off+1 .. marker_off+4``.
    Decoding at the FIXED offset (not by searching for the marker token)
    re-anchors the marker exactly as production re-anchors the first token of
    each step via the Python STEP_END dispatch — so a passing program decodes
    cleanly.
    """
    val = 0
    for j in range(4):
        val |= (int(slice_tokens[marker_off + 1 + j]) & 0xFF) << (j * 8)
    return val & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    cluster: str
    classification: str               # PASS / FAIL / ALU-OPAQUE / ERROR
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
    alu_op: Optional[str] = None
    # "high"  = PC or AX byte-0 divergence — the flat teacher-forced decode is
    #           faithful here (these determine control flow + the single-byte
    #           result); the attributed rule is trustworthy.
    # "low"   = AX byte>=1 (or SP/BP/STACK0) divergence — the multi-byte high
    #           bytes are subject to the cross-step re-anchoring that a flat
    #           forward cannot reproduce (a known faithfulness gap; see the
    #           module docstring). The divergence MAY be a real bug (production
    #           also fails some) OR a flat-decode artifact — confirm with
    #           --faithfulness-check before trusting the attribution.
    confidence: str = "high"
    note: str = ""


PASS = "PASS"
FAIL = "FAIL"
ALU_OPAQUE = "ALU-OPAQUE"
ERROR = "ERROR"

# Bytes the flat teacher-forced decode reproduces faithfully vs production:
# PC (all 4 bytes — control flow) and AX byte 0 (the single-byte result). AX
# bytes 1..3 and the SP/BP/STACK0 registers ride the cross-step re-anchoring
# the flat forward cannot replicate, so a divergence there is LOW confidence.
def _div_confidence(reg: str, byte_k: int) -> str:
    if reg == "PC":
        return "high"
    if reg == "AX" and byte_k == 0:
        return "high"
    return "low"


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
    if verbose:
        print(f"[interp-oracle-gate] model: d_model={model.d_model} "
              f"blocks={len(model.blocks)} ops={len(flat_ops)}",
              file=sys.stderr, flush=True)
    return GateContext(
        model=model, layout=layout, dim_positions=dim_positions,
        flat_ffn_ops=flat_ops, interp=interp,
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
        logits = faithful_full_forward(ctx.model, full_ctx)
        fa = logits.argmax(dim=-1).tolist()
    except Exception as exc:  # noqa: BLE001
        return GateResult(name=name, cluster=cluster, classification=ERROR,
                          n_steps=n_steps, note=f"faithful forward error: {exc!r}")

    # Predicted draft token at draft position t = argmax at (prefix + t - 1).
    def pred_tok(t: int) -> int:
        return int(fa[prefix + t - 1])

    # Per step, decode the model's predicted (PC, AX) from its draft slice and
    # compare to the oracle. FIRST diverging byte wins.
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
                # ALU-OPAQUE? The step's opcode is a composite ALU block the
                # interpreter cannot execute from IR — flag, do NOT attribute.
                opcode = ot.opcodes[s] if s < len(ot.opcodes) else -1
                conf = _div_confidence(reg, k)
                if opcode in _ALU_OPAQUE_OPCODE:
                    return GateResult(
                        name=name, cluster=cluster, classification=ALU_OPAQUE,
                        n_steps=n_steps, div_step=s, div_reg=reg, div_byte=k,
                        expected=exp_b, got=got_b, expected_reg=o_val,
                        got_reg=g_val, alu_op=_ALU_OPAQUE_OPCODE[opcode],
                        confidence=conf,
                        note=(f"divergence at step {s} {reg}[{k}] is at/after "
                              f"opaque ALU op {_ALU_OPAQUE_OPCODE[opcode]} "
                              f"(no IR rule form; needs imperative GPU path)"),
                    )
                # Attribute the wrong byte to its owning declarative rule.
                # Only attribute HIGH-confidence (PC / AX byte-0) divergences:
                # a LOW-confidence (AX high-byte cross-step) divergence may be a
                # flat-decode artifact, so naming a rule there would mislead.
                attr_op = attr_rule = None
                attr_c = None
                if attribute and conf == "high":
                    attr_op, attr_rule, attr_c = _attribute_byte(
                        ctx, full_ctx, prefix, s, reg, k, got_b,
                    )
                op_name = _OPCODE_NAME.get(opcode, f"op{opcode}")
                low_tag = ("  [LOW-CONFIDENCE: AX high-byte cross-step "
                           "re-anchor — confirm with --faithfulness-check]"
                           if conf == "low" else "")
                return GateResult(
                    name=name, cluster=cluster, classification=FAIL,
                    n_steps=n_steps, div_step=s, div_reg=reg, div_byte=k,
                    expected=exp_b, got=got_b, expected_reg=o_val, got_reg=g_val,
                    attributed_op=attr_op, attributed_rule=attr_rule,
                    attributed_contrib=attr_c, confidence=conf,
                    note=(f"step {s} opcode={op_name} {reg}[{k}] "
                          f"exp=0x{exp_b:02x} got=0x{got_b:02x}" + low_tag),
                )
    # No (PC, AX) divergence found. BUT if the program's path executes an
    # opaque composite-ALU opcode, the gate CANNOT certify a PASS: the efficient
    # composite-ALU block the interpreter runs (AddSub/Mul/DivMod/Shift) is NOT
    # guaranteed bit-identical to production's imperative ALU runtime, so the
    # ALU step's result is not authoritative even when it happens to match the
    # oracle here. Flag it ALU-OPAQUE (gate declines to judge) rather than
    # over-claiming PASS — e.g. a MUL the interpreter computes correctly that
    # production's imperative MUL gets wrong would be a false PASS.
    alu_steps = [(s, ot.opcodes[s]) for s in range(min(n_steps, len(ot.opcodes)))
                 if ot.opcodes[s] in _ALU_OPAQUE_OPCODE]
    if alu_steps:
        s0, op0 = alu_steps[0]
        return GateResult(
            name=name, cluster=cluster, classification=ALU_OPAQUE,
            n_steps=n_steps, div_step=s0, alu_op=_ALU_OPAQUE_OPCODE[op0],
            note=(f"path executes opaque ALU op {_ALU_OPAQUE_OPCODE[op0]} at "
                  f"step {s0} (and {len(alu_steps) - 1} more); the composite-ALU "
                  f"forward is not bit-certified vs production — needs the "
                  f"imperative GPU path"),
        )
    return GateResult(name=name, cluster=cluster, classification=PASS,
                      n_steps=n_steps, note=f"all {n_steps} steps match oracle")


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
        resid = _faithful_residual_pre_head(ctx.model, full_ctx)[pred_pos]
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
) -> Tuple[int, int, List[str]]:
    """Run interpreter-vs-NEURAL per-step decode on ``programs`` (GPU build).

    Builds the REAL neural runner on ``gpu`` and, for each program, runs the
    production fail-fast decode (the authoritative neural per-step (PC, AX)) and
    the CPU interpreter's per-step decode over the SAME oracle tape, and counts
    programs where they AGREE step-for-step. Disagreements are interpreter
    COVERAGE GAPS (reported by name) so the gate never mis-attributes them.
    Returns ``(n_faithful, n_authoritative, gap_names, lowconf_gap_names)``.
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
    gaps: List[str] = []         # interp != neural and NOT a known low-conf gap
    lowconf_gaps: List[str] = []  # the characterized AX high-byte flat-decode gap
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

        # LOW-confidence interp FAIL (AX high-byte cross-step): this is the
        # CHARACTERIZED flat-decode gap, NOT an interpreter math bug. Count it
        # as a coverage gap only when it actually disagrees with the neural
        # verdict (interp fails, neural passes).
        if ires.classification == FAIL and ires.confidence == "low":
            if n_pass:
                lowconf_gaps.append(
                    f"{p['name']} (interp FAIL? @ step{ires.div_step} "
                    f"AX[{ires.div_byte}] cross-step, neural PASS)")
            else:
                n_faithful += 1  # both fail — consistent
            continue

        # The gate is AUTHORITATIVE for this program (PASS or HIGH-conf FAIL).
        n_authoritative += 1
        if ires.classification == ALU_OPAQUE:
            if not n_pass:
                n_faithful += 1
            else:
                gaps.append(f"{p['name']} (ALU-OPAQUE but neural PASS)")
            continue
        same_verdict = (i_pass == n_pass)
        same_div = (ires.div_step == n_div) if not i_pass else True
        if same_verdict and same_div:
            n_faithful += 1
        else:
            gaps.append(
                f"{p['name']} (interp={ires.classification}@{ires.div_step} "
                f"neural={n_status}@{n_div})"
            )
    return n_faithful, n_authoritative, gaps, lowconf_gaps


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
            print(f"  PASS         {r.name}  ({r.n_steps} steps)")
        elif r.classification == ALU_OPAQUE:
            print(f"  ALU-OPAQUE   {r.name}  step={r.div_step} {r.alu_op}")
        elif r.classification == ERROR:
            print(f"  ERROR        {r.name}  {r.note}")
        else:
            tag = "FAIL" if r.confidence == "high" else "FAIL?"
            rule = (f"{r.attributed_op}::{r.attributed_rule}"
                    if r.attributed_rule
                    else ("(low-conf AX high-byte — not attributed)"
                          if r.confidence == "low"
                          else "(no runtime writer / default cell)"))
            print(f"  {tag:<12} {r.name}  step={r.div_step} "
                  f"{r.div_reg}[{r.div_byte}] exp=0x{r.expected:02x} "
                  f"got=0x{r.got:02x} -> {rule}"
                  + (f"  (contrib={r.attributed_contrib:+.3f})"
                     if r.attributed_contrib is not None else "")
                  + ("  [LOW-CONF cross-step]" if r.confidence == "low" else ""))
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
                      if r.classification == FAIL and r.confidence == "high")
    n_fail_low = sum(1 for r in results
                     if r.classification == FAIL and r.confidence == "low")
    print("-" * 90)
    print(f"  SUMMARY [{title}]: {n} programs")
    print(f"    {PASS:<14} {counts.get(PASS, 0)}")
    print(f"    {'FAIL':<14} {n_fail_high}   (HIGH-confidence: PC / AX byte-0 "
          f"divergence — attributed)")
    if n_fail_low:
        print(f"    {'FAIL?':<14} {n_fail_low}   (LOW-confidence: AX high-byte "
              f"cross-step re-anchor — possible flat-decode gap)")
    if counts.get(ALU_OPAQUE):
        print(f"    {ALU_OPAQUE:<14} {counts[ALU_OPAQUE]}")
    if counts.get(ERROR):
        print(f"    {ERROR:<14} {counts[ERROR]}")
    interp_authoritative = counts.get(PASS, 0) + n_fail_high
    print(f"    gate AUTHORITATIVE (PASS + HIGH-confidence FAIL) = "
          f"{interp_authoritative}/{n}")
    print(f"    ALU-OPAQUE (needs imperative GPU ALU path) = "
          f"{counts.get(ALU_OPAQUE, 0)}/{n}; LOW-confidence (confirm with "
          f"--faithfulness-check) = {n_fail_low}/{n}")

    # Top owning rules by FAIL frequency — the highest-leverage fix targets.
    # Only HIGH-confidence attributed FAILs (the trustworthy ones).
    rule_freq: Dict[str, int] = {}
    for r in results:
        if (r.classification == FAIL and r.confidence == "high"
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
                     if r.classification == FAIL and r.confidence == "high"
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
            n_ok, n_auth, gaps, lowconf = faithfulness_check(
                ctx, sample, gpu, args.max_steps)
            print(f"  Over {len(sample)} sampled programs: gate is "
                  f"AUTHORITATIVE on {n_auth} (PASS or HIGH-confidence FAIL); "
                  f"of those, interp == neural on {n_ok}/{n_auth}.")
            print(f"  LOW-confidence (AX high-byte cross-step) gaps where the "
                  f"flat decode disagrees with neural: {len(lowconf)} "
                  f"(CHARACTERIZED flat-decode limitation, not a model bug).")
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
            if lowconf:
                print(f"  (low-confidence cross-step gaps, reported separately "
                      f"so the gate never mis-attributes them:)")
                for g in lowconf[:12]:
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
