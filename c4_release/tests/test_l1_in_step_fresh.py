"""B7-1: unit tests for the IN_STEP_FRESH lifecycle dim.

The L1 attention head 5 emits a positive in-step freshness signal that:
  * is ~1.0 immediately after the most recent ``MARK_SE_ONLY`` (or
    ``MARK_CS`` at program start),
  * decays toward 0.0 as more tokens accumulate within the current step,
  * resets to ~1.0 at the next ``MARK_SE_ONLY``.

These tests build a synthetic residual stream that mirrors the L1 input
layout (CONST + IS_MARK + MARK_SE_ONLY + MARK_CS at the relevant token
positions), invoke ``layer1_threshold_attn`` standalone, and assert that
the IN_STEP_FRESH output column has the expected monotonic-decay/reset
shape.

See ``investigation/l7-l9-structural-audit:REPORT.md`` Section 2.4 and
``investigation/bd-dim-usage-map:REPORT.md`` Section 5 for the design
rationale.
"""

import torch

from c4_release.neural_vm.vm_step import AutoregressiveAttention, _SetDim
from c4_release.neural_vm.unified_compiler.ops.l1_ops import (
    make_layer1_threshold_attn_op,
)
from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic


# Step layout: PC(5) + AX(5) + SP(5) + BP(5) + STACK0(5) + MEM(9) + SE(1) = 35.
STEP_TOKENS = 35
SE_POS_IN_STEP = STEP_TOKENS - 1  # SE is the last token of each step.


def _build_attn_at_l1_layout():
    """Construct a bare AutoregressiveAttention sized like L1 (d=512, 8 heads),
    bake just the head-5 IN_STEP_FRESH spec into it, and return the attention
    module + the dim positions used.
    """
    d_model = 512
    num_heads = 8
    attn = AutoregressiveAttention(
        dim=d_model, num_heads=num_heads, max_seq_len=512, layer_idx=1,
    )
    # The bake_fn expects ``dim_positions`` (a name -> int dict). Using the
    # legacy _SetDim positions keeps the test self-contained without invoking
    # the compiler's pin_io_only allocator.
    dim_positions = {
        name: getattr(_SetDim, name)
        for name in dir(_SetDim)
        if not name.startswith("_") and isinstance(getattr(_SetDim, name), int)
    }
    op = make_layer1_threshold_attn_op()
    with torch.no_grad():
        attn.W_q.zero_()
        attn.W_k.zero_()
        attn.W_v.zero_()
        attn.W_o.zero_()
        op.bake_fn(attn, dim_positions, 100.0)
    return attn, dim_positions


def _build_residual_with_steps(n_steps: int, d_model: int, dim_positions):
    """Construct a residual stream of ``n_steps`` synthetic 35-token steps.

    Head 5 only reads ``CONST`` (Q side) and ``MARK_SE_ONLY`` / ``MARK_CS``
    (K and V side), so the test residual sets just those three dims:
      * CONST=1.0 at every position.
      * MARK_CS=1.0 only at position 0 (program start).
      * MARK_SE_ONLY=1.0 at the SE position of each step.
    """
    S = n_steps * STEP_TOKENS
    x = torch.zeros(1, S, d_model)
    x[0, :, dim_positions["CONST"]] = 1.0
    # MARK_CS at program start only.
    x[0, 0, dim_positions["MARK_CS"]] = 1.0
    # STEP_END at the SE position of every step.
    for step in range(n_steps):
        se_pos = step * STEP_TOKENS + SE_POS_IN_STEP
        x[0, se_pos, dim_positions["MARK_SE_ONLY"]] = 1.0
    return x


def _read_in_step_fresh(attn, x, dim_positions):
    """Run the attention forward and return the IN_STEP_FRESH column.

    Subtract the residual since the attention class returns ``x + attn_out``.
    """
    with torch.no_grad():
        y = attn(x)
    return (y[0, :, dim_positions["IN_STEP_FRESH"]]
            - x[0, :, dim_positions["IN_STEP_FRESH"]])


def test_in_step_fresh_high_at_step_start():
    """At the token immediately after a STEP_END the signal should be ~1.0."""
    attn, dim_positions = _build_attn_at_l1_layout()
    # Need at least 2 steps so we have a STEP_END followed by a fresh step.
    x = _build_residual_with_steps(n_steps=3, d_model=attn.W_q.shape[1],
                                   dim_positions=dim_positions)
    fresh = _read_in_step_fresh(attn, x, dim_positions)

    # Position right after the first STEP_END (step 0's SE at pos 34) is
    # pos 35 — the first token of step 1.
    pos_after_se = STEP_TOKENS
    # Position right after the second STEP_END is 2*STEP_TOKENS.
    pos_after_se_2 = 2 * STEP_TOKENS

    assert fresh[pos_after_se].item() > 0.7, (
        f"Expected IN_STEP_FRESH > 0.7 immediately after STEP_END at pos "
        f"{pos_after_se}; got {fresh[pos_after_se].item():.4f}."
    )
    assert fresh[pos_after_se_2].item() > 0.7, (
        f"Expected IN_STEP_FRESH > 0.7 at second step start (pos "
        f"{pos_after_se_2}); got {fresh[pos_after_se_2].item():.4f}."
    )


def test_in_step_fresh_high_at_program_start():
    """At position 0 (MARK_CS) the signal should be ~1.0 even though no SE
    has fired yet — MARK_CS serves as the implicit step-zero boundary."""
    attn, dim_positions = _build_attn_at_l1_layout()
    x = _build_residual_with_steps(n_steps=1, d_model=attn.W_q.shape[1],
                                   dim_positions=dim_positions)
    fresh = _read_in_step_fresh(attn, x, dim_positions)
    # At pos 0 the only key with K signal is MARK_CS at pos 0, dist=0.
    assert fresh[0].item() > 0.7, (
        f"Expected IN_STEP_FRESH > 0.7 at program start (MARK_CS); "
        f"got {fresh[0].item():.4f}."
    )


def test_in_step_fresh_decays_within_step():
    """Within a step (e.g. step 1), IN_STEP_FRESH should be monotonically
    non-increasing as distance from the prior STEP_END grows."""
    attn, dim_positions = _build_attn_at_l1_layout()
    x = _build_residual_with_steps(n_steps=3, d_model=attn.W_q.shape[1],
                                   dim_positions=dim_positions)
    fresh = _read_in_step_fresh(attn, x, dim_positions)

    # Sample positions across step 1 (positions 35..69 inclusive of SE@69).
    step_start = STEP_TOKENS                 # pos 35: just after SE
    step_mid = STEP_TOKENS + 15              # pos 50: middle of step
    step_end = 2 * STEP_TOKENS - 2           # pos 68: just before next SE

    v_start = fresh[step_start].item()
    v_mid = fresh[step_mid].item()
    v_end = fresh[step_end].item()

    assert v_start > v_mid + 0.01, (
        f"Expected decay between step start and middle: "
        f"start={v_start:.4f}, mid={v_mid:.4f}."
    )
    assert v_mid > v_end + 0.01, (
        f"Expected decay between step middle and end: "
        f"mid={v_mid:.4f}, end={v_end:.4f}."
    )
    # Sanity: end-of-step value should be measurably below the start.
    assert v_start - v_end > 0.05, (
        f"Expected IN_STEP_FRESH to drop by > 0.05 across one step; "
        f"start={v_start:.4f}, end={v_end:.4f}, drop={v_start - v_end:.4f}."
    )


def test_in_step_fresh_resets_at_next_step_end():
    """At the next STEP_END, the IN_STEP_FRESH signal should rebound back
    to ~1.0 (dist=0 to the most-recent SE)."""
    attn, dim_positions = _build_attn_at_l1_layout()
    x = _build_residual_with_steps(n_steps=3, d_model=attn.W_q.shape[1],
                                   dim_positions=dim_positions)
    fresh = _read_in_step_fresh(attn, x, dim_positions)

    # Step 1's SE at pos 69 (= 2*STEP_TOKENS - 1).
    se_2 = 2 * STEP_TOKENS - 1

    pre_reset = fresh[se_2 - 1].item()
    at_reset = fresh[se_2].item()

    assert at_reset > pre_reset, (
        f"Expected IN_STEP_FRESH to rebound at STEP_END (pos {se_2}); "
        f"pre={pre_reset:.4f}, at_SE={at_reset:.4f}."
    )
    assert at_reset > 0.7, (
        f"Expected reset value > 0.7 at STEP_END pos {se_2}; "
        f"got {at_reset:.4f}."
    )


def test_in_step_fresh_dim_allocated_by_compiler():
    """The full-VM compiler must allocate the IN_STEP_FRESH dim without
    triggering any STALENESS warnings."""
    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        _model, layout = compile_full_vm_dynamic()
        staleness = [w for w in wlist if "STALENESS" in str(w.message)]

    assert "IN_STEP_FRESH" in layout.dim_positions, (
        "IN_STEP_FRESH dim not allocated by compile_full_vm_dynamic."
    )
    pos = layout.dim_positions["IN_STEP_FRESH"]
    assert pos >= 0, f"IN_STEP_FRESH allocated at invalid position {pos}."
    assert len(staleness) == 0, (
        f"compile_full_vm_dynamic emitted {len(staleness)} STALENESS warnings: "
        f"{[str(w.message) for w in staleness[:3]]}"
    )


def test_in_step_fresh_setdim_slot_is_96():
    """Smoke test: _SetDim.IN_STEP_FRESH should match the B6-K recommendation
    (slot 96, reclaimed from dead L0 head 5 region)."""
    assert _SetDim.IN_STEP_FRESH == 96, (
        f"Expected IN_STEP_FRESH at slot 96 per B6-K; got {_SetDim.IN_STEP_FRESH}."
    )
