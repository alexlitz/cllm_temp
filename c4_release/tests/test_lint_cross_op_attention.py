"""Tests for ``tools/lint_cross_op_attention.py``.

The lint is the AUTHORING-TIME counterpart to the GPU tripwire: it catches a
non-local attention regression (a shared head's softmax output changing for
an op the fix never intended to touch) BEFORE a build is shipped to a GPU.

Two tiers:

  * Fast unit tests (always run): the probe-construction + production
    attention-math helpers in isolation, on a tiny synthetic ``PureAttention``.
    These exercise the softmax1/ALiBi numerics + the multi-row probe layout
    without compiling the full VM.

  * The discrimination contract (``slow``): build the real VM flag-OFF /
    flag-ON and assert the lint FLAGS the byte-0 fix
    (``C4_OPERAND_GATHER_PSH_ROWSELECT``) while PASSING a clean band/LM-head
    fix (``C4_AX_BYTE1_FULL_WIDTH``). This is the proof the lint discriminates
    real regressions from clean fixes — the exact blind spot that made byte-0
    look like +8 when it was -39.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_C4 = _THIS.parent.parent
if str(_C4) not in sys.path:
    sys.path.insert(0, str(_C4))

torch = pytest.importorskip("torch")

from tools import lint_cross_op_attention as lint  # noqa: E402


# ---------------------------------------------------------------------------
# Fast unit tests — probe construction + attention math (no VM build).
# ---------------------------------------------------------------------------
class _FakeLayout:
    """Minimal layout stub: just dim_positions + d_model."""

    def __init__(self, dim_positions, d_model):
        self.dim_positions = dim_positions
        self.d_model = d_model


def _toy_dim_positions():
    # Distinct positions for every name the probe references.
    names = [
        "STACK0_BYTE0", "CONST", "PSH_AT_SP", "MARK_AX",
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
        "OP_ADD", "OP_SUB", "OP_MUL",
    ]
    dp = {}
    pos = 1
    for n in names:
        dp[n] = pos
        # leave a 16-wide gap after the embed bands so band+offset writes fit
        pos += 20 if n in ("CLEAN_EMBED_LO", "CLEAN_EMBED_HI") else 1
    return dp, pos + 4


def test_probe_query_is_marked_and_opcode_tagged():
    dp, D = _toy_dim_positions()
    layout = _FakeLayout(dp, D)
    ctx = {"name": "targeted_clean_psh", "psh": True, "hi_nibble": 5, "n_rows": 2}
    x, q = lint._make_probe(layout, "OP_ADD", ctx)
    assert x.shape == (3, D)
    assert q == 2
    # Query row carries MARK_AX + the opcode flag.
    assert x[q, dp["MARK_AX"]] == 1.0
    assert x[q, dp["OP_ADD"]] == 1.0
    # Row 0 is the PSH-output candidate with a clean high nibble.
    assert x[0, dp["STACK0_BYTE0"]] == 1.0
    assert x[0, dp["PSH_AT_SP"]] == 1.0
    assert x[0, dp["CLEAN_EMBED_HI"] + 5] == 1.0


def test_probe_n_rows_controls_sequence_length():
    dp, D = _toy_dim_positions()
    layout = _FakeLayout(dp, D)
    for n_rows in (1, 2, 3):
        ctx = {"name": "c", "psh": True, "hi_nibble": 5, "n_rows": n_rows}
        x, q = lint._make_probe(layout, "OP_ADD", ctx)
        assert x.shape[0] == n_rows + 1
        assert q == n_rows


def test_probe_no_psh_context_omits_psh_flag():
    dp, D = _toy_dim_positions()
    layout = _FakeLayout(dp, D)
    ctx = {"name": "var_expr_no_psh", "psh": False, "hi_nibble": 5, "n_rows": 2}
    x, _ = lint._make_probe(layout, "OP_SUB", ctx)
    assert x[0, dp["PSH_AT_SP"]] == 0.0


def test_probe_corrupted_hi_context_leaves_hi_nibble_default():
    dp, D = _toy_dim_positions()
    layout = _FakeLayout(dp, D)
    ctx = {"name": "corrupted_hi_nib", "psh": True, "hi_nibble": 0, "n_rows": 2}
    x, _ = lint._make_probe(layout, "OP_ADD", ctx)
    # No clean high nibble cell is set on row 0 (the IMM-decode-bug case).
    hi = dp["CLEAN_EMBED_HI"]
    assert x[0, hi:hi + 16].sum() == 0.0


def _toy_attn(d_model, n_heads, *, slope=0.1):
    """A tiny ``PureAttention`` whose ``forward`` is unused — the lint reads
    W_q/W_k/W_v/W_o + scale + alibi_slopes + use_softmax1 directly."""
    from neural_vm.base_layers import PureAttention

    attn = PureAttention(dim=d_model, num_heads=n_heads, causal=True)
    attn.register_buffer("alibi_slopes", torch.full((n_heads,), float(slope)))
    attn.use_softmax1 = True
    return attn


def test_head_attn_weights_match_softmax1_alibi_math():
    """The lint's head math must reproduce softmax1 + ALiBi + causal exactly."""
    d_model, n_heads = 8, 2
    attn = _toy_attn(d_model, n_heads, slope=0.25)
    HD = attn.head_dim
    # Head 0: Q reads dim 0, K reads dim 1 (so query@key score is controllable).
    attn.W_q.data.zero_(); attn.W_k.data.zero_()
    attn.W_v.data.zero_(); attn.W_o.data.zero_()
    attn.W_q.data[0, 0] = 3.0     # head-0 slot 0 reads dim 0
    attn.W_k.data[0, 1] = 3.0     # head-0 slot 0 reads dim 1
    attn.W_v.data[0, 2] = 1.0     # head-0 slot 0 relays dim 2

    S = 3
    x = torch.zeros(S, d_model)
    x[0, 1] = 1.0   # key value on row 0
    x[1, 1] = 0.0   # weaker key on row 1
    x[2, 0] = 1.0   # query value on the last row

    w = lint._head_attn_weights(attn, x, head=0)   # [S, S]
    assert w.shape == (S, S)
    # Causality: the query (row 2) must place ~0 weight on... itself is allowed,
    # but rows strictly after it don't exist; the upper triangle (j>i) is 0.
    for i in range(S):
        for j in range(i + 1, S):
            assert abs(float(w[i, j])) < 1e-7, (i, j)

    # Hand-recompute row 2 with the production formula and compare.
    Wq = attn.W_q.data[0:HD]
    Wk = attn.W_k.data[0:HD]
    Q = x @ Wq.T
    K = x @ Wk.T
    scores = (Q @ K.T) * attn.scale
    pos = torch.arange(S, dtype=torch.float32)
    scores = scores + (-0.25 * (pos.unsqueeze(1) - pos.unsqueeze(0)).abs())
    scores = scores + torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    anchor = torch.zeros(S, 1)
    mv = torch.maximum(scores.amax(-1, keepdim=True), anchor)
    es = torch.exp(scores - mv); ea = torch.exp(anchor - mv)
    expected = es / (ea + es.sum(-1, keepdim=True))
    assert torch.allclose(w, expected, atol=1e-6)
    # softmax1 sink keeps total real-key mass strictly below 1.
    assert float(w[2].sum()) < 1.0


def test_detect_modified_shared_head_on_synthetic_pair():
    """A planted K-slot change on a PRE-EXISTING head is reported as shared."""
    d_model, n_heads = 8, 2

    off = _toy_attn(d_model, n_heads)
    on = _toy_attn(d_model, n_heads)
    # Head 0 pre-exists in BOTH (Q+K rows non-empty).
    for a in (off, on):
        a.W_q.data[0, 0] = 5.0
        a.W_k.data[0, 1] = 5.0
        a.W_o.data[2, 0] = 1.0   # head 0 writes out dim 2
    # ON adds a NEW K slot to head 0 (the byte-0-style extension).
    on.W_k.data[3, 4] = 7.0

    class _M:
        def __init__(self, attn):
            self.blocks = [type("B", (), {"attn": attn})()]

    mods = lint.detect_modified_shared_heads(_M(off), _M(on))
    assert len(mods) == 1
    m = mods[0]
    assert m.block == 0 and m.head == 0
    assert m.pre_existing is True
    assert 2 in m.out_dims


def test_is_expected_whitelist_matching():
    s = {"OP_ADD", "targeted_clean_psh", "OP_SUB:three_competing"}
    assert lint._is_expected("OP_ADD", "var_expr_no_psh", s)      # opcode match
    assert lint._is_expected("OP_MUL", "targeted_clean_psh", s)   # context match
    assert lint._is_expected("OP_SUB", "three_competing", s)      # pair match
    assert not lint._is_expected("OP_MUL", "var_expr_no_psh", s)


# ---------------------------------------------------------------------------
# The discrimination contract (slow — compiles the full VM 4x).
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_byte0_fix_is_flagged_as_non_local():
    """The byte-0 fix MUST flag a non-local change on the shared L7 head."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    ok, reports = lint.lint_flag(
        "C4_OPERAND_GATHER_PSH_ROWSELECT",
        expect=("OP_ADD:targeted_clean_psh", "OP_SUB:targeted_clean_psh"),
        verbose=False,
    )
    # Skip (don't fail) if the byte-0 flag is reverted out of this checkout —
    # the lint correctly reports "no shared head" and the contract is moot.
    shared = [r for r in reports if r.mod.pre_existing]
    if not shared:
        pytest.skip(
            "C4_OPERAND_GATHER_PSH_ROWSELECT modifies no shared head in this "
            "checkout (flag reverted) — discrimination is not exercisable."
        )
    assert ok is False, "byte-0 fix should be FLAGGED (non-local change)"
    flagged = [
        r for rep in reports for r in rep.rows
        if r.changed
        and not lint._is_expected(
            r.opcode, r.context,
            {"OP_ADD:targeted_clean_psh", "OP_SUB:targeted_clean_psh"},
        )
    ]
    assert flagged, "expected at least one non-local (opcode,context) flag"
    # The flagged contexts are the var/expr-representative ones, NOT the
    # targeted clean-PSH context.
    assert any(r.context in ("corrupted_hi_nib", "three_competing")
               for r in flagged)


@pytest.mark.slow
def test_clean_band_fix_passes():
    """A band/LM-head fix (FULL_WIDTH) modifies no shared head -> PASS."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    ok, reports = lint.lint_flag("C4_AX_BYTE1_FULL_WIDTH", verbose=False)
    assert ok is True, "clean band fix should PASS"
    assert not any(r.mod.pre_existing for r in reports), (
        "clean band fix must not modify a shared attention head"
    )
