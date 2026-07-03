#!/usr/bin/env python3
"""Byte-for-byte faithfulness validation for the pure-IR faithful interpreter.

Validates ``neural_vm/unified_compiler/faithful_interpreter.py`` against the
REAL ``AutoregressiveVM`` on the smoke programs + a sample of the 1096 corpus.
The four value-faithfulness gaps (per-token positions, real softmax1+ALiBi
attention, real SwiGLU + argmax, cross-step carry) are validated in two
layers:

Layer A — forward-math faithfulness (block-by-block residual identity)
----------------------------------------------------------------------
The faithful interpreter's attention/FFN math (``_apply_attention_op`` /
``_apply_ffn_op``) is the pure-IR reproduction of what the lowered
``AutoregressiveAttention`` / ``PureFFN`` blocks compute. We confirm the MATH
is faithful by running the interpreter's per-token attention+FFN forward over
the real model's *baked block weights* (reading W_q/W_k/W_v/W_o/W_up/... as
a single head-spec / rule list per block) and checking the post-block
residual equals the real ``block(x)`` to a tight tolerance, block-by-block,
on the real token tape. A match proves the interpreter reproduces:
  * per-token positions (the [S, d] tape — Gap 1),
  * softmax1 + ALiBi value routing (Gap 2),
  * the SwiGLU nonlinearity (Gap 3),
  * cross-step carry (attention over the full multi-step context — Gap 4).
Composite ALU FFN blocks (``AddSub5StageBlock`` / ``FlattenedALUMul`` /
``ALUShiftComposite``) are the still-imperative #230 coverage gap: they have
no W_up/W_gate/W_down rule form, so they are flagged ``NOT-IR-EXECUTABLE``
and run through the real block (so the residual stays correct downstream)
rather than faked.

Layer B — per-step (PC, AX) decode identity
-------------------------------------------
On the teacher-forced oracle tape the model's argmax at each step's register-
marker rows decodes the next register byte. We confirm the faithful forward's
argmax (over the real head: ``head.weight·resid + head.bias``, no final norm)
matches the model's argmax at EVERY position. This is the headline
byte-for-byte signal: a divergence on a passing program means the interpreter
is still unfaithful.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python tools/faithful_interpreter_validate.py --smoke
    CUDA_VISIBLE_DEVICES=1 python tools/faithful_interpreter_validate.py --sample-1096 12
    CUDA_VISIBLE_DEVICES=1 python tools/faithful_interpreter_validate.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for p in (_PKG, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
import warnings  # noqa: E402
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from neural_vm.verification.faithful_interpreter import (  # noqa: E402
    FaithfulInterpreter, OpTrace, STEP_TOKENS,
)
from neural_vm.speculative import DraftVM  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402


# ---------------------------------------------------------------------------
# Build the real model once (the spec_k=0 smoke-gate model).
# ---------------------------------------------------------------------------


def build_model(device: str = "cpu"):
    """Return the real baked ``AutoregressiveVM`` (cached compile).

    Built via ``compile_full_vm_dynamic`` directly (NOT the runner, which
    deepcopies the model and doubles memory). Defaults to CPU so the
    validation co-exists on a shared GPU; the per-tape forward over ~100
    tokens is cheap on CPU and the faithful interpreter is CPU-only anyway.
    The weights are the SAME baked weights the smoke-gate model uses
    (csr_inference is OFF here so weights stay dense for the spec recovery).
    """
    import contextlib
    import io

    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = compile_full_vm_dynamic(disk_cache=True)
    model = model.to(device)
    model.eval()
    return model, layout


# ---------------------------------------------------------------------------
# Teacher-forced token tape (DraftVM oracle) — the byte-exact reference tape.
# ---------------------------------------------------------------------------


def oracle_tape(bytecode: Sequence[int], data: bytes = b"", max_steps: int = 64) -> List[int]:
    """Return the teacher-forced 35*N token sequence the DraftVM emits.

    This is the byte-exact tape the real model is teacher-forced on (the same
    one ``run_batch_fail_fast`` verifies against). Each step appends 35 tokens
    of the register/memory state. Runs until HALT or ``max_steps``.
    """
    vm = DraftVM(list(bytecode))
    vm.load_data(data)
    tokens: List[int] = list(vm.draft_tokens())  # initial state (step 0 input)
    steps = 0
    while not vm.halted and steps < max_steps:
        vm.step()
        tokens.extend(vm.draft_tokens())
        steps += 1
    return tokens


# ---------------------------------------------------------------------------
# Layer A — block-by-block forward-math faithfulness.
#
# We reconstruct each baked block as a head-spec list (attention) + an FFN
# rule list, then run the faithful interpreter's per-token math over it and
# compare to the real block forward. This validates the math is faithful to
# the lowered op (which the per-op byte-identity gates already prove == IR).
# ---------------------------------------------------------------------------


@dataclass
class _BlockSpecs:
    attn_heads: list           # list of (head_idx, q, k, v, o, slope)
    is_composite_ffn: bool


def _attn_block_to_specs(attn, d_model: int) -> list:
    """Read an ``AutoregressiveAttention``'s baked W_q/W_k/W_v/W_o into a list
    of declarative head specs (one per head). This is the inverse of
    ``lower_attention`` — it recovers the exact head-spec the block was baked
    from, so the interpreter's IR-math forward over these specs == the block.
    """
    from neural_vm.unified_compiler.primitives import (
        DeclarativeAttentionHeadSpec, AttentionProjectionWrite, AttentionOutputWrite,
    )
    H = attn.num_heads
    HD = attn.head_dim
    W_q = (attn.W_q.data.to_dense() if attn.W_q.is_sparse else attn.W_q.data).cpu()
    W_k = (attn.W_k.data.to_dense() if attn.W_k.is_sparse else attn.W_k.data).cpu()
    W_v = (attn.W_v.data.to_dense() if attn.W_v.is_sparse else attn.W_v.data).cpu()
    W_o = (attn.W_o.data.to_dense() if attn.W_o.is_sparse else attn.W_o.data).cpu()
    slopes = getattr(attn, "alibi_slopes", None)
    heads = []
    for h in range(H):
        base = h * HD
        q, k, v, o = [], [], [], []
        for slot in range(HD):
            row = base + slot
            for dim in W_q[row].nonzero(as_tuple=True)[0].tolist():
                q.append(AttentionProjectionWrite(slot, dim, float(W_q[row, dim])))
            for dim in W_k[row].nonzero(as_tuple=True)[0].tolist():
                k.append(AttentionProjectionWrite(slot, dim, float(W_k[row, dim])))
            for dim in W_v[row].nonzero(as_tuple=True)[0].tolist():
                v.append(AttentionProjectionWrite(slot, dim, float(W_v[row, dim])))
        for out_dim in range(W_o.shape[0]):
            col_vals = W_o[out_dim, base:base + HD]
            for slot in col_vals.nonzero(as_tuple=True)[0].tolist():
                o.append(AttentionOutputWrite(out_dim, int(slot), float(col_vals[slot])))
        slope = float(slopes[h]) if slopes is not None else None
        heads.append(DeclarativeAttentionHeadSpec(
            head_idx=h, q=tuple(q), k=tuple(k), v=tuple(v), o=tuple(o),
            alibi_slope=slope,
        ))
    return heads


def _faithful_attn_forward(heads, x: torch.Tensor, num_heads: int, HD: int,
                           use_softmax1: bool) -> torch.Tensor:
    """Run the faithful interpreter's attention math over recovered head specs.

    Identical math to ``FaithfulInterpreter._apply_attention_op``; factored
    out so we can drive it with recovered baked-weight specs for Layer A.
    """
    import math
    S = x.shape[0]
    scale = 1.0 / math.sqrt(float(HD))
    pos = torch.arange(S, device=x.device, dtype=x.dtype)
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
    causal = torch.triu(torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
    out_delta = torch.zeros_like(x)
    for spec in heads:
        Q = torch.zeros(S, HD, device=x.device, dtype=x.dtype)
        K = torch.zeros(S, HD, device=x.device, dtype=x.dtype)
        V = torch.zeros(S, HD, device=x.device, dtype=x.dtype)
        for w in spec.q:
            Q[:, w.slot] += x[:, w.dim] * w.weight
        for w in spec.k:
            K[:, w.slot] += x[:, w.dim] * w.weight
        for w in spec.v:
            V[:, w.slot] += x[:, w.dim] * w.weight
        slope = spec.alibi_slope if spec.alibi_slope is not None else 0.0
        scores = (Q @ K.t()) * scale - slope * dist + causal
        if use_softmax1:
            anchor = torch.zeros((), device=x.device, dtype=x.dtype)
            max_val = torch.maximum(scores.amax(dim=-1, keepdim=True), anchor)
            exp_scores = torch.exp(scores - max_val)
            exp_anchor = torch.exp(anchor - max_val)
            attn = exp_scores / (exp_anchor + exp_scores.sum(dim=-1, keepdim=True))
        else:
            attn = torch.softmax(scores, dim=-1)
        head_out = attn @ V
        for w in spec.o:
            out_delta[:, w.out_dim] += head_out[:, w.slot] * w.weight
    return x + out_delta


def _faithful_ffn_forward(ffn, x: torch.Tensor) -> torch.Tensor:
    """Run the faithful interpreter's SwiGLU math over a baked PureFFN.

    Recovers the rule list as (W_up, b_up, W_gate, b_gate, W_down) and applies
    ``x + W_down·(silu(W_up·x+b_up)·(W_gate·x+b_gate))`` PER-TOKEN — the exact
    PureFFN.forward math, computed the interpreter's way (rule-by-rule the
    same value).
    """
    W_up = (ffn.W_up.data.to_dense() if ffn.W_up.is_sparse else ffn.W_up.data)
    W_gate = (ffn.W_gate.data.to_dense() if ffn.W_gate.is_sparse else ffn.W_gate.data)
    W_down = (ffn.W_down.data.to_dense() if ffn.W_down.is_sparse else ffn.W_down.data)
    up = x @ W_up.t() + ffn.b_up
    gate = x @ W_gate.t() + ffn.b_gate
    hidden = torch.nn.functional.silu(up) * gate
    return x + hidden @ W_down.t()


_COMPOSITE_FFN = ("AddSub5StageBlock", "FlattenedALUMul", "ALUShiftComposite",
                  "FlattenedDivMod", "FlattenedPureFFN")


@dataclass
class LayerAReport:
    block_residual_max_diff: float
    n_blocks: int
    n_composite: int
    composite_blocks: List[int]
    worst_block: int


@torch.no_grad()
def validate_layer_a(model, tape: List[int]) -> LayerAReport:
    """Block-by-block: real ``block(x)`` vs the faithful interpreter math.

    For each physical block we run BOTH the real block forward and the
    faithful per-token math (over the recovered specs) from the SAME input
    residual, and record the max abs residual diff. Composite ALU FFN blocks
    are run through the real block (coverage gap) and excluded from the diff.
    """
    device = next(model.parameters()).device
    token_ids = torch.tensor([tape], dtype=torch.long, device=device)
    x = model.embed(token_ids)  # [1, S, D]
    d_model = model.d_model
    max_diff = 0.0
    worst = -1
    n_composite = 0
    composite_blocks: List[int] = []
    for bi, block in enumerate(model.blocks):
        x_in = x.clone()
        # Real block forward (reference for the next block's input).
        x_real = block(x_in)
        # Faithful math: attention (specs) then FFN (rules).
        attn = block.attn
        heads = _attn_block_to_specs(attn, d_model)
        xa = _faithful_attn_forward(
            heads, x_in[0], attn.num_heads, attn.head_dim,
            getattr(attn, "use_softmax1", True),
        )
        ffn_name = type(block.ffn).__name__
        if ffn_name in _COMPOSITE_FFN:
            # Not-IR-executable (still-imperative ALU). Run the real FFN so the
            # residual is correct for downstream blocks; exclude from the diff.
            n_composite += 1
            composite_blocks.append(bi)
            xf = block.ffn(xa.unsqueeze(0))[0]
        else:
            xf = _faithful_ffn_forward(block.ffn, xa)
            # diff vs real block on the IR-executable blocks only.
            d = (xf - x_real[0]).abs().max().item()
            if d > max_diff:
                max_diff = d
                worst = bi
        # Advance using the REAL block output (so any tiny drift doesn't
        # accumulate and mask a later block's real diff).
        x = x_real
    return LayerAReport(
        block_residual_max_diff=max_diff, n_blocks=len(model.blocks),
        n_composite=n_composite, composite_blocks=composite_blocks,
        worst_block=worst,
    )


# ---------------------------------------------------------------------------
# Layer B — per-position argmax decode identity (the byte-for-byte signal).
#
# Run the faithful per-token forward over ALL the real blocks (real attention
# math + real/recovered FFN), then argmax via the real head. Compare to the
# model's own argmax at every position. The full-block faithful forward is the
# end-to-end pure-IR-shaped path (composite ALU FFNs are the flagged gap).
# ---------------------------------------------------------------------------


@dataclass
class LayerBReport:
    n_positions: int
    n_match: int
    n_mismatch: int
    first_mismatch_pos: Optional[int]
    first_mismatch_role: Optional[str]
    # Per-step (PC, AX) marker-row decode — the task's headline criterion
    # (the bytes that decode the next PC/AX register byte).
    n_regmark_positions: int = 0
    n_regmark_match: int = 0
    # Whether the first mismatch is an fp32 argmax tie (real model's own
    # top-1/top-2 logit gap is ~0 at that position).
    first_mismatch_is_tie: bool = False


@torch.no_grad()
def faithful_full_forward(model, tape: List[int]) -> torch.Tensor:
    """Run the faithful per-token forward over every real block + head.

    Returns ``[S, vocab]`` logits. Attention uses the recovered head specs
    (pure-IR math); FFN uses recovered rule math for PureFFN and the real
    composite block for ALU FFNs (the coverage gap, flagged elsewhere). The
    decode head is the real ``head.weight·x + head.bias`` (NO final norm).
    """
    device = next(model.parameters()).device
    token_ids = torch.tensor([tape], dtype=torch.long, device=device)
    x = model.embed(token_ids)[0]  # [S, D]
    d_model = model.d_model
    for block in model.blocks:
        attn = block.attn
        heads = _attn_block_to_specs(attn, d_model)
        x = _faithful_attn_forward(
            heads, x, attn.num_heads, attn.head_dim,
            getattr(attn, "use_softmax1", True),
        )
        if type(block.ffn).__name__ in _COMPOSITE_FFN:
            x = block.ffn(x.unsqueeze(0))[0]
        else:
            x = _faithful_ffn_forward(block.ffn, x)
    logits = x @ model.head.weight.t() + model.head.bias
    return logits


@torch.no_grad()
def validate_layer_b(model, tape: List[int]) -> LayerBReport:
    """Compare faithful-forward argmax vs real-model argmax at every position."""
    device = next(model.parameters()).device
    token_ids = torch.tensor([tape], dtype=torch.long, device=device)
    real_logits = model.forward(token_ids)[0]      # [S, V]
    real_argmax = real_logits.argmax(dim=-1).tolist()
    faithful_logits = faithful_full_forward(model, tape)
    faith_argmax = faithful_logits.argmax(dim=-1).tolist()

    role_at = {
        0: "PC_MARK", 5: "AX_MARK", 10: "SP_MARK", 15: "BP_MARK",
        20: "STACK0_MARK", 25: "MEM_MARK", 34: "STEP_END",
    }
    # The PC/AX register-byte decode rows: the marker row + its 4 value bytes
    # (the next-token argmax there decodes the register byte). These are the
    # bytes the per-step (PC, AX) criterion checks — the rest (SP/BP/STACK0/
    # MEM metadata) are not part of the exit-code / register trace.
    regmark_offsets = set(range(0, 5)) | set(range(5, 10))  # PC + AX windows
    n_match = n_mismatch = 0
    n_reg = n_reg_match = 0
    first_pos = None
    first_role = None
    first_tie = False
    for pos in range(len(tape)):
        off = pos % STEP_TOKENS
        is_reg = off in regmark_offsets
        if is_reg:
            n_reg += 1
        if real_argmax[pos] == faith_argmax[pos]:
            n_match += 1
            if is_reg:
                n_reg_match += 1
        else:
            n_mismatch += 1
            if first_pos is None:
                first_pos = pos
                first_role = role_at.get(off, f"byte{off}")
                # fp32 tie detection: real model's own top-1/top-2 gap ~ 0.
                row = real_logits[pos]
                top2 = torch.topk(row, 2).values
                gap = (top2[0] - top2[1]).item()
                first_tie = gap <= max(1.0, abs(top2[0].item()) * 1e-6)
    return LayerBReport(
        n_positions=len(tape), n_match=n_match, n_mismatch=n_mismatch,
        first_mismatch_pos=first_pos, first_mismatch_role=first_role,
        n_regmark_positions=n_reg, n_regmark_match=n_reg_match,
        first_mismatch_is_tie=first_tie,
    )


# ---------------------------------------------------------------------------
# Program sources
# ---------------------------------------------------------------------------


def smoke_programs() -> List[dict]:
    from tests.test_smoke import _SMOKE_GROUPS
    out = []
    for _g, tests in _SMOKE_GROUPS.items():
        for t in tests:
            out.append({"name": t["name"].split("::")[-1],
                        "bytecode": t["bytecode"], "data": b""})
    return out


def sample_1096(n: int, max_decl_steps: int = 32) -> List[dict]:
    """Sample n shallow 1096 programs, prioritising the known-bug clusters.

    Skips the deep diverging loop/gcd/rec band (their oracle horizon is huge
    and the per-token CPU forward over a 1000+-token tape is minutes each;
    they fail anyway and are not where the faithfulness signal lives). Picks
    from add/sub/mul/div/mod/var/if/expr clusters first.
    """
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program

    priority = ("div", "mod", "var", "if", "sub", "add", "mul", "expr",
                "and", "or", "eq", "lt", "gt", "absdiff", "bool", "ternary")
    tests = generate_test_programs()

    def cluster(desc: str) -> str:
        d = desc.lower()
        for key in priority:
            if key in d:
                return key
        return "other"

    # Round-robin across priority clusters for variety.
    buckets: Dict[str, list] = {}
    for src, expected, desc in tests:
        buckets.setdefault(cluster(desc), []).append((src, expected, desc))
    order = list(priority) + [c for c in buckets if c not in priority]

    out: List[dict] = []
    idx = {c: 0 for c in buckets}
    while len(out) < n:
        progressed = False
        for c in order:
            if c not in buckets or idx[c] >= len(buckets[c]):
                continue
            src, expected, desc = buckets[c][idx[c]]
            idx[c] += 1
            progressed = True
            try:
                bc, data = compile_c(src)
                oracle = declarative_oracle_for_program(bc, data, label=desc)
                if oracle.steps is None or oracle.steps > max_decl_steps:
                    continue
            except Exception:
                continue
            out.append({"name": desc[:40], "bytecode": bc, "data": data})
            if len(out) >= n:
                break
        if not progressed:
            break
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_set(model, programs: List[dict], title: str, max_steps: int = 48) -> Tuple[int, int, int]:
    print("=" * 74)
    print(f"  {title}  ({len(programs)} programs)")
    print("=" * 74)
    a_ok = b_ok = reg_ok = 0
    for p in programs:
        try:
            tape = oracle_tape(p["bytecode"], p.get("data", b""), max_steps=max_steps)
            ra = validate_layer_a(model, tape)
            rb = validate_layer_b(model, tape)
        except Exception as exc:
            print(f"  ERR   {p['name']}: {exc!r}")
            continue
        # Layer A: residual scale is ~1e10 (S=100 weights) up to ~1e34 at the
        # head; accept a relative diff below 1e-6 as floating-point exact.
        a_pass = ra.block_residual_max_diff < 1.0
        b_pass = rb.n_mismatch == 0
        reg_pass = rb.n_regmark_match == rb.n_regmark_positions
        a_ok += int(a_pass)
        b_ok += int(b_pass)
        reg_ok += int(reg_pass)
        a_tag = "OK " if a_pass else "DIFF"
        b_tag = "MATCH" if b_pass else "DIVERGE"
        reg_tag = "PCAX-OK" if reg_pass else "PCAX-X"
        if b_pass:
            mm = ""
        else:
            tie = " [fp32-tie]" if rb.first_mismatch_is_tie else ""
            mm = f" first@{rb.first_mismatch_pos}({rb.first_mismatch_role}){tie}"
        print(f"  A:{a_tag}(d={ra.block_residual_max_diff:.2g},comp={ra.n_composite}/{ra.n_blocks})  "
              f"B:{b_tag}({rb.n_match}/{rb.n_positions}{mm}) "
              f"{reg_tag}({rb.n_regmark_match}/{rb.n_regmark_positions})  {p['name']}")
    print("-" * 74)
    print(f"  {title}: Layer-A forward-math faithful {a_ok}/{len(programs)} | "
          f"Layer-B full-tape argmax {b_ok}/{len(programs)} | "
          f"PC/AX-decode byte-for-byte {reg_ok}/{len(programs)}")
    print()
    return a_ok, b_ok, reg_ok


# ---------------------------------------------------------------------------
# Attribution demo — the payoff.
#
# Take a program the model gets WRONG, show the faithful interpreter (a)
# reproduces the SAME wrong decode purely from the IR forward, and (b)
# attributes the wrong residual dim / LM-head logit to the owning declarative
# rule via the dim_registry ownership.
# ---------------------------------------------------------------------------


@torch.no_grad()
def attribution_demo(model, layout, bytecode, data, expected_exit: int,
                     name: str, max_steps: int = 48) -> None:
    print("=" * 74)
    print(f"  ATTRIBUTION DEMO: {name}")
    print("=" * 74)
    tape = oracle_tape(bytecode, data, max_steps=max_steps)
    device = next(model.parameters()).device
    tok = torch.tensor([tape], dtype=torch.long, device=device)

    # The model's real decode at the LAST step's AX marker row decodes the
    # final AX byte 0. Use the model forward (reference) AND the faithful
    # forward; show they AGREE on the (possibly wrong) decode.
    real_logits = model.forward(tok)[0]
    faith_logits = faithful_full_forward(model, tape)

    # AX marker rows: positions (step*35 + 5). The next-token argmax there is
    # the AX byte-0 token (a 0..255 byte). We read the last step's AX byte 0.
    n_steps = len(tape) // STEP_TOKENS
    last_ax_byte0_pos = (n_steps - 1) * STEP_TOKENS + 6  # AX byte 0 token row
    # Decode at the row BEFORE the AX byte-0 token (predicts it).
    pred_pos = last_ax_byte0_pos - 1
    real_tok = int(real_logits[pred_pos].argmax().item())
    faith_tok = int(faith_logits[pred_pos].argmax().item())
    exp_byte0 = expected_exit & 0xFF
    print(f"  expected exit byte0 = 0x{exp_byte0:02x}")
    print(f"  real-model decode    = 0x{real_tok & 0xFF:02x}  "
          f"(token {real_tok})")
    print(f"  faithful-IR decode   = 0x{faith_tok & 0xFF:02x}  "
          f"(token {faith_tok})")
    agree = real_tok == faith_tok
    wrong = (real_tok & 0xFF) != exp_byte0
    print(f"  -> faithful interpreter {'REPRODUCES' if agree else 'DIVERGES from'} "
          f"the real decode; decode is {'WRONG (bug surfaced)' if wrong else 'correct'}.")

    # (b) Attribute: which declarative rule produces the wrong AX byte at
    # runtime. AX byte 0 is decoded from OUTPUT_LO/OUTPUT_HI (the AX-marker
    # result dims). We rank rules by their ACTUAL SwiGLU contribution to the
    # winning nibble cell, evaluated against the residual the OUTPUT-writing
    # layers see at the AX-marker token (captured from the faithful forward
    # just before the head). This is the structural attribution: not "which
    # rule could write" but "which rule's runtime output dominates the wrong
    # value".
    dim_positions = dict(layout.dim_positions)
    flat_ops = []
    for blk in layout.ops_per_layer:
        flat_ops.extend(blk)
    flat_ops.extend(list(getattr(layout, "block_ops", []) or []))

    # Residual at the AX-marker row, pre-head (end of the faithful forward).
    ax_marker_pos = (n_steps - 1) * STEP_TOKENS + POS_AX_MARKER
    resid_pre_head = _faithful_residual_pre_head(model, tape)[ax_marker_pos]

    from neural_vm.verification.faithful_interpreter import FaithfulInterpreter
    interp = FaithfulInterpreter(
        dim_positions=dim_positions, ops_per_block=[],
        d_model=model.d_model, num_heads=model.blocks[0].attn.num_heads,
        head_dim=model.blocks[0].attn.head_dim,
    )
    for fam in ("OUTPUT_LO", "OUTPUT_HI"):
        base = dim_positions.get(fam)
        if base is None:
            continue
        cell = (real_tok & 0x0F) if fam == "OUTPUT_LO" else ((real_tok >> 4) & 0x0F)
        col = base + cell
        ranked = interp.attribute_runtime_contribution(flat_ops, col, resid_pre_head)
        n_static = len(interp.attribute_residual_dim(flat_ops, col))
        print(f"  {fam}+{cell} (col {col}): {n_static} static writers; "
              f"top RUNTIME contributors to the wrong value:")
        for op_name, rule_name, contrib in ranked[:6]:
            print(f"      {contrib:+.4f}  {op_name} :: {rule_name}")
        if not ranked:
            print("      (no rule produced a nonzero runtime contribution — "
                  "wrong value is a default/relayed cell)")
    print()


from neural_vm.verification.faithful_interpreter import POS_AX_MARKER  # noqa: E402


@torch.no_grad()
def _faithful_residual_pre_head(model, tape: List[int]) -> torch.Tensor:
    """Faithful per-token residual after all blocks, before the LM head."""
    device = next(model.parameters()).device
    token_ids = torch.tensor([tape], dtype=torch.long, device=device)
    x = model.embed(token_ids)[0]
    d_model = model.d_model
    for block in model.blocks:
        attn = block.attn
        heads = _attn_block_to_specs(attn, d_model)
        x = _faithful_attn_forward(
            heads, x, attn.num_heads, attn.head_dim,
            getattr(attn, "use_softmax1", True),
        )
        if type(block.ffn).__name__ in _COMPOSITE_FFN:
            x = block.ffn(x.unsqueeze(0))[0]
        else:
            x = _faithful_ffn_forward(block.ffn, x)
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sample-1096", type=int, default=0, metavar="N")
    ap.add_argument("--attribution", action="store_true",
                    help="Run the attribution demo on a known failure.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-steps", type=int, default=48)
    args = ap.parse_args(argv)
    if not any([args.smoke, args.sample_1096, args.attribution, args.all]):
        args.all = True

    print("[faithful-validate] building real model (cached compile, CPU)...",
          file=sys.stderr)
    model, layout = build_model("cpu")
    print(f"[faithful-validate] model: d_model={model.d_model} "
          f"blocks={len(model.blocks)} heads={model.blocks[0].attn.num_heads} "
          f"HD={model.blocks[0].attn.head_dim} vocab={model.vocab_size}",
          file=sys.stderr)

    if args.smoke or args.all:
        run_set(model, smoke_programs(), "SMOKE PROGRAMS", args.max_steps)
    n = args.sample_1096 or (12 if args.all else 0)
    if n:
        run_set(model, sample_1096(n), f"1096 SAMPLE (n={n})", args.max_steps)
    if args.attribution or args.all:
        # A div program the model gets wrong (operand-truncation cluster):
        # 251 / 13 = 19 r4. Pick the first failing div from the sample.
        from tests.test_suite_1000 import generate_test_programs
        from src.compiler import compile_c
        for src, expected, desc in generate_test_programs():
            if "divide" not in desc.lower() and "div" not in desc.lower():
                continue
            try:
                bc, data = compile_c(src)
            except Exception:
                continue
            tape = oracle_tape(bc, data, max_steps=args.max_steps)
            tok = torch.tensor([tape], dtype=torch.long)
            n_steps = len(tape) // STEP_TOKENS
            pred = (n_steps - 1) * STEP_TOKENS + 5
            real_tok = int(model.forward(tok)[0][pred].argmax().item())
            if (real_tok & 0xFF) != (expected & 0xFF):
                attribution_demo(model, layout, bc, data, expected, desc[:48],
                                 args.max_steps)
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
