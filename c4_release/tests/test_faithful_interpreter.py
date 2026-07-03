"""Tests for the value-faithful pure-IR forward simulator.

These pin the four value-faithfulness gaps the faithful interpreter closes,
WITHOUT building the full ~1-2-minute model: each test lowers a small IR into
a fresh ``PureFFN`` / ``AutoregressiveAttention`` and checks the faithful
interpreter's per-token IR math reproduces the lowered forward exactly. The
per-op byte-identity gates (``compare_symbolic_to_lowered_*``) already prove
``lowered == IR``; these prove ``faithful-interpreter-math == lowered``, so by
transitivity the interpreter is faithful to the real op.

The end-to-end byte-for-byte validation against the real ``AutoregressiveVM``
on the smoke + 1096 corpus lives in
``tools/faithful_interpreter_validate.py`` (needs the GPU/CPU model build and
is run as a tool, not a unit test).
"""

from __future__ import annotations

import math

import pytest
import torch

from c4_release.neural_vm.verification.faithful_interpreter import (
    FaithfulInterpreter,
)
from c4_release.neural_vm.unified_compiler.ir import (
    CompilerIR,
    FFNRule,
)
from c4_release.neural_vm.unified_compiler.primitives import (
    AttentionOutputWrite,
    AttentionProjectionWrite,
    DeclarativeAttentionHeadSpec,
)


# ---------------------------------------------------------------------------
# Gap 3 — real SwiGLU nonlinearity at scale S, per token.
# ---------------------------------------------------------------------------


def test_ffn_math_matches_lowered_pureffn():
    """The interpreter's per-token SwiGLU == a freshly-lowered PureFFN.

    A single gated-write rule lowered into a PureFFN, then run on a few
    random per-token residuals, must produce the same delta the interpreter
    computes from the rule directly (the real silu/gate math, not the bag-of-
    dims threshold-sum the legacy DSLInterpreter uses).
    """
    from c4_release.neural_vm.base_layers import PureFFN

    d_model = 12
    dim_positions = {"A": 0, "B": 4, "OUT": 8}
    S = 100.0
    rule = FFNRule.gated_write(
        conditions=[("A+0", 1.0), ("A+1", 1.0)],
        threshold=1.5,                 # fires when A+0 + A+1 >= 1.5
        gate=None, gate_bias=1.0,
        writes=[("OUT+0", 0.5), ("OUT+1", -0.25)],
        name="demo",
    )
    ir = CompilerIR()
    ir.layer(0).ffn.append(rule)

    ffn = PureFFN(dim=d_model, hidden_dim=1)
    ir.lower_ffn(ffn, dim_positions, S=S)

    interp = FaithfulInterpreter(
        dim_positions=dim_positions, ops_per_block=[],
        d_model=d_model, num_heads=1, head_dim=d_model, S=S,
    )

    torch.manual_seed(0)
    x = torch.zeros(5, d_model)
    # Drive A+0/A+1 to fire on some rows, not others.
    x[0, 0] = 1.0; x[0, 1] = 1.0     # sum=2 >= 1.5 -> fires
    x[1, 0] = 1.0                    # sum=1 < 1.5 -> blocked
    x[2, 0] = 1.0; x[2, 1] = 1.0
    x[4, 1] = 1.0

    lowered = ffn(x.unsqueeze(0))[0]
    faithful = interp._apply_ffn_op(ir, x.clone(), 0, _trace())

    assert torch.allclose(faithful, lowered, atol=1e-4), (
        (faithful - lowered).abs().max().item()
    )
    # And the gate actually fired only on the >=threshold rows (gap 3: the
    # silu boundary, NOT the legacy hard threshold-sum).
    assert faithful[0, 8].item() != 0.0   # OUT+0 written on a firing row
    assert abs(faithful[1, 8].item()) < 1e-3  # blocked row stays ~0


# ---------------------------------------------------------------------------
# Gap 2 — real softmax1 + ALiBi attention value routing (context-dependent).
# ---------------------------------------------------------------------------


def test_attention_math_matches_lowered_attention():
    """The interpreter's softmax1+ALiBi MHA == a lowered AutoregressiveAttention.

    A single head that copies a value dim from the attended position to an
    output dim, run over a multi-token tape, must match the real attention
    block forward (softmax1, ALiBi slope, causal mask) — proving the value
    routing is context-dependent, not the legacy V->O copy.
    """
    from c4_release.neural_vm.vm_step import AutoregressiveAttention

    d_model = 8
    num_heads = 1
    HD = d_model
    # Head: Q reads dim 0, K reads dim 1, V reads dim 2, O writes dim 3.
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AttentionProjectionWrite(0, 0, 5.0),),
        k=(AttentionProjectionWrite(0, 1, 1.0),),
        v=(AttentionProjectionWrite(1, 2, 1.0),),
        o=(AttentionOutputWrite(3, 1, 1.0),),
        alibi_slope=0.5,
    )
    ir = CompilerIR()
    ir.layer(0).attention.append(spec)

    attn = AutoregressiveAttention(
        d_model, num_heads=num_heads, attention_normalization="softmax1",
        positional_encoding="alibi", use_flash_attention=False,
    )
    # Match the slope the spec declares so the two ALiBi biases agree.
    attn.alibi_slopes[0] = 0.5
    ir.lower_attention(attn, HD)

    interp = FaithfulInterpreter(
        dim_positions={}, ops_per_block=[],
        d_model=d_model, num_heads=num_heads, head_dim=HD,
        use_softmax1=True,
    )

    torch.manual_seed(1)
    x = torch.randn(4, d_model)
    lowered = attn(x.unsqueeze(0))[0]
    faithful = interp._apply_attention_op(ir, x.clone(), 0, _trace())

    assert torch.allclose(faithful, lowered, atol=1e-4), (
        (faithful - lowered).abs().max().item()
    )


# ---------------------------------------------------------------------------
# Gap 1 — per-token positions: distinct tokens get distinct residual rows
# and ALiBi distances vary by position.
# ---------------------------------------------------------------------------


def test_per_token_positions_and_alibi_distance():
    """Attention output differs per query position because ALiBi distance
    depends on the absolute token position (gap 1 + the position-aware bias).

    With a uniform value field, a head with a steep ALiBi slope weights the
    nearer key more, so the output at a later query (which has a more distant
    program-start key) differs from an earlier query — only possible with real
    per-token positions, not a single bag-of-dims state.
    """
    d_model = 6
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AttentionProjectionWrite(0, 0, 1.0),),
        k=(AttentionProjectionWrite(0, 0, 1.0),),
        v=(AttentionProjectionWrite(1, 1, 1.0),),
        o=(AttentionOutputWrite(2, 1, 1.0),),
        alibi_slope=2.0,
    )
    ir = CompilerIR()
    ir.layer(0).attention.append(spec)
    interp = FaithfulInterpreter(
        dim_positions={}, ops_per_block=[],
        d_model=d_model, num_heads=1, head_dim=d_model,
    )
    # Every token: Q/K-dim=1.0 (uniform score), V-dim ramps with position.
    x = torch.zeros(5, d_model)
    x[:, 0] = 1.0
    for p in range(5):
        x[p, 1] = float(p)
    out = interp._apply_attention_op(ir, x.clone(), 0, _trace())
    routed = out[:, 2] - x[:, 2]  # the O-write delta per position
    # Each query attends causally with ALiBi recency: later queries see a
    # different mix, so the routed value is strictly increasing and the
    # per-position values are all distinct (proves position-awareness).
    assert routed[0].item() == pytest.approx(0.0, abs=1e-6)  # only self
    assert len(set(round(v, 4) for v in routed.tolist())) == 5


# ---------------------------------------------------------------------------
# Gap 4 — cross-step carry: attention reaches across a 35-token step boundary.
# ---------------------------------------------------------------------------


def test_cross_step_carry_over_step_boundary():
    """A head with a flat slope can pull a value from a PRIOR step window.

    Place a value at a token in step 0 and a query at a token in step 1
    (>35 positions later); a zero-slope head with a matching Q/K key must
    route the prior-step value forward — the cross-step carry the bag-of-dims
    interpreter cannot represent (it has one state per step).
    """
    d_model = 6
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        q=(AttentionProjectionWrite(0, 0, 100.0),),  # large -> beats softmax1 sink
        k=(AttentionProjectionWrite(0, 1, 1.0),),
        v=(AttentionProjectionWrite(1, 2, 1.0),),
        o=(AttentionOutputWrite(3, 1, 1.0),),
        alibi_slope=0.0,  # no recency decay -> can reach across steps
    )
    ir = CompilerIR()
    ir.layer(0).attention.append(spec)
    interp = FaithfulInterpreter(
        dim_positions={}, ops_per_block=[],
        d_model=d_model, num_heads=1, head_dim=d_model,
    )
    S = 40  # > 35 tokens: spans a step boundary
    x = torch.zeros(S, d_model)
    # Step-0 source token at position 3: K-key on, V carries 7.0.
    x[3, 1] = 1.0
    x[3, 2] = 7.0
    # Step-1 query token at position 38: Q-key on.
    x[38, 0] = 1.0
    out = interp._apply_attention_op(ir, x.clone(), 0, _trace())
    # The query at 38 pulled the prior-step (pos 3) value forward.
    assert out[38, 3].item() == pytest.approx(7.0, abs=0.2)


# ---------------------------------------------------------------------------
# Coverage boundary — opaque (non-IR) ops are flagged, not faked.
# ---------------------------------------------------------------------------


def test_opaque_op_is_flagged_not_executed():
    class _OpaqueOp:
        name = "imperative_alu"
        kind = "ffn"
        layer_idx = 0
        compiler_ir = None
        compiler_ir_factory = None

    interp = FaithfulInterpreter(
        dim_positions={}, ops_per_block=[[_OpaqueOp()]],
        d_model=4, num_heads=1, head_dim=4,
    )
    seed = torch.zeros(1, 4)
    res = interp.forward(seed)
    assert res.opaque_skipped == ["imperative_alu"]
    # Opaque op leaves the residual unchanged (not faked).
    assert torch.equal(res.residual, seed)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _trace():
    from c4_release.neural_vm.verification.faithful_interpreter import OpTrace
    return OpTrace(name="t", kind="ffn", layer_idx=0)
