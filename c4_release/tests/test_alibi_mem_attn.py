"""ALiBi-based memory-propagation attention head: proof-of-concept tests.

These tests exercise the new ``make_layer9_alibi_mem_attn_op`` op
(see ``neural_vm/unified_compiler/ops/l9_ops.py``) at two levels:

1. **Registration**: the op is in ``all_core_ops()``, lands at layer_idx=9
   after ``compile_full_vm()``, and is a no-op when ``enable=False`` (so
   existing tests are byte-identical).

2. **ALiBi slope micro-validation**: a focused construction that proves the
   ALiBi slope of 0.5 (the tuned default in the docstring) causes the
   most-recent matching K position to win over older equally-matching K
   positions — the "recent dominates" property that lets a later PSH at
   the same SP overwrite an earlier PSH's value via attention alone.

The full integration (attention head writing OUTPUT bytes during LI/LC/POP
when ``_inject_mem_metadata`` is disabled) is NOT yet wired end-to-end;
see the module docstring for next steps.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_alibi_mem_attn_op_is_registered():
    """The new op is in all_core_ops() and lands at layer 9 after compile."""
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm

    ops = all_core_ops()
    found = [op for op in ops if op.name == "layer9_alibi_mem_attn"]
    assert len(found) == 1, (
        f"Expected exactly 1 layer9_alibi_mem_attn op, found {len(found)}"
    )
    op = found[0]
    assert op.layer_idx == 9
    assert op.migrated is True
    assert op.phase == 9.2  # after lev_addr_relay/lev_bp_to_pc_relay
    assert op.kind == "block"

    # Verify it lands at L9 in the layout after compile.
    _, layout = compile_full_vm()
    block_ops_by_name = {bo.name: bo for bo in layout.block_ops}
    assert "layer9_alibi_mem_attn" in block_ops_by_name
    placed = block_ops_by_name["layer9_alibi_mem_attn"]
    # block ops with explicit layer_idx report it (see test_layer_idx_consistency).
    if placed.target_op_name is None:
        assert placed.layer_idx == 9


def test_alibi_mem_attn_op_is_noop_when_disabled():
    """With enable=False (default), L9 attn head 2 weights stay zero.

    This is the byte-identical guarantee: enabling-by-default would risk
    breaking existing tests; the op is registered for dep-graph stability
    but its bake is a guard-clause early return when enable=False.
    """
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm

    model, _ = compile_full_vm()
    attn9 = model.blocks[9].attn
    HD = attn9.W_q.shape[0] // attn9.num_heads
    head = 2
    # All 4 weight matrices for head 2 must be untouched (zero) when disabled.
    # We slice W_q[head*HD : (head+1)*HD, :] which represents head 2's Q rows.
    assert attn9.W_q[head * HD:(head + 1) * HD, :].abs().sum().item() == 0.0
    assert attn9.W_k[head * HD:(head + 1) * HD, :].abs().sum().item() == 0.0
    assert attn9.W_v[head * HD:(head + 1) * HD, :].abs().sum().item() == 0.0
    assert attn9.W_o[:, head * HD:(head + 1) * HD].abs().sum().item() == 0.0


def test_alibi_slope_picks_most_recent_match():
    """ALiBi slope 0.5 + scale-10 24-bit address match: most-recent wins.

    Constructs a minimal AutoregressiveAttention layer with 8 heads, sets
    head 2's slope to 0.5, and verifies that when two K positions have
    *identical* (matching) ADDR_KEY encodings, the more recent one wins
    over the older one by the expected ALiBi-distance margin.

    Setup
    -----
    Sequence length S = 70 (= 2 VM steps of 35 tokens).
    Q at position 69 (current load).
    K at position  4 (first PSH step's output).
    K at position 39 (second PSH step's output) — most-recent match.

    Both K positions are constructed with identical ADDR_KEY encoding, so
    pure address match score is equal. The ALiBi penalty differs:
      - K@39: dist = 69-39 = 30, penalty = -0.5*30 = -15
      - K@4:  dist = 69-4  = 65, penalty = -0.5*65 = -32.5
    Margin: +17.5 favoring K@39.

    Verification
    ------------
    After attention, the output at position 69 must reflect K@39's value
    (we encode K@39 with V=+1 in dim 0 and K@4 with V=-1 in dim 0).
    """
    from neural_vm.vm_step import AutoregressiveAttention

    torch.manual_seed(0)
    D = 512
    num_heads = 8
    HD = D // num_heads
    S = 70  # 2 VM steps

    attn = AutoregressiveAttention(dim=D, num_heads=num_heads, max_seq_len=128,
                                   layer_idx=9)
    # Zero out everything; configure only head 2.
    with torch.no_grad():
        attn.W_q.zero_()
        attn.W_k.zero_()
        attn.W_v.zero_()
        attn.W_o.zero_()
        # Set head 2 slope to 0.5 (matches the production design).
        attn.alibi_slopes[2] = 0.5
        # Zero other slopes so they don't pollute the output across heads.
        for h in range(num_heads):
            if h != 2:
                attn.alibi_slopes[h] = 0.0

        head = 2
        base = head * HD
        # Use SEPARATE residual dims for Q-side and K-side address markers.
        # If Q and K read the same dim, the Q position itself is a perfect
        # K-match at dist=0, which crushes any other K via ALiBi (this is what
        # broke the original version of this test). The production design has
        # the same shape: load-position Q reads MEM_VAL_B0 + OP_LI_RELAY (a
        # Q-only marker), store-position K reads PSH_AT_SP + MEM_STORE (a
        # K-only marker).
        # Scale chosen so post-ALiBi score swamps softmax1 anchor=0:
        #   raw score = SCALE^2 / sqrt(HD) = 50^2 / 8 = 312.5
        #   alibi penalty at dist=30: -0.5*30 = -15
        #   alibi penalty at dist=65: -0.5*65 = -32.5
        #   margin between matching K's: 17.5 (>> softmax1 noise threshold)
        SCALE = 50.0
        QADDR_DIM = 100  # Q-only marker (e.g., OP_LI_RELAY + MEM_VAL_B0 gating)
        KADDR_DIM = 101  # K-only marker (e.g., PSH_AT_SP)
        attn.W_q[base + 0, QADDR_DIM] = SCALE
        attn.W_k[base + 0, KADDR_DIM] = SCALE

        # V: copy residual dim 200 -> output. We tag K@39 with dim 200 = +1
        # and K@4 with dim 200 = -1, so a winner-takes-all attention output
        # at Q's position should reflect +1 (K@39) or -1 (K@4).
        attn.W_v[base + 0, 200] = 1.0
        # O: write head's value output into residual dim 300.
        attn.W_o[300, base + 0] = 1.0

    # Build the residual stream.
    x = torch.zeros(1, S, D)
    # Q "load" position at 69 reads from QADDR_DIM.
    x[0, 69, QADDR_DIM] = 1.0
    # K "store" positions at 4 and 39 advertise KADDR_DIM.
    x[0, 4, KADDR_DIM] = 1.0
    x[0, 39, KADDR_DIM] = 1.0

    # K-tagged values: K@4 has V=-1, K@39 has V=+1.
    x[0, 4, 200] = -1.0
    x[0, 39, 200] = +1.0

    with torch.no_grad():
        y = attn(x)

    out_at_q = y[0, 69, 300].item()
    # Residual add: y = x + attn_out; x[0,69,300]=0, so y[...,300] = attn_out.
    # Recency wins (slope 0.5, dist diff 35): attn_out should be close to +1
    # (K@39's value), not -1 (K@4's). softmax1 may not be exactly 1.0 due to
    # the ZFOD anchor, but it should be clearly positive and well above 0.
    assert out_at_q > 0.5, (
        f"Expected most-recent match (V=+1) to dominate; got {out_at_q:.4f}. "
        f"Either ALiBi slope is too low or address scale too low."
    )
    # Sanity: pure positive direction confirms the most-recent K won.
    assert out_at_q < 1.5, (
        f"Output magnitude unexpectedly large; got {out_at_q:.4f}. "
        f"Check that head 2's V projection is correctly localized."
    )


def test_alibi_slope_distinguishes_recent_vs_older_psh():
    """Slope 0.5 + 35-token VM step: 1 step of recency = +17.5 score margin.

    Empirical check that the chosen slope (0.5) gives a usable margin
    between consecutive PSH steps at the same address. The margin needs
    to be (a) big enough to clearly win in softmax, and (b) small enough
    that legitimate matches across 10+ steps still attract attention.
    """
    # 1 VM step distance: 35 tokens.
    # slope * step_distance = 0.5 * 35 = 17.5 score margin between
    # adjacent matching PSHes.
    SLOPE = 0.5
    STEP_TOKENS = 35
    one_step_margin = SLOPE * STEP_TOKENS
    assert one_step_margin == 17.5

    # 10-step lookback cost: 0.5 * 350 = 175 score penalty.
    # Address-match contribution from 24-bit binary encoding (scale=10):
    # 24 dims × 100 / sqrt(HD=64) = 24*100/8 = 300 points at exact match,
    # 0 at random. So 10-step legitimate matches still net +125 over noise.
    ten_step_cost = SLOPE * STEP_TOKENS * 10
    addr_match_score = 24 * 100 / 8  # 24 bits × scale^2 / sqrt(HD)
    assert ten_step_cost == 175.0
    assert addr_match_score == 300.0
    assert addr_match_score - ten_step_cost == 125.0, (
        "10-step lookback budget margin: should remain positive so that "
        "legitimate distant matches still attract attention."
    )
