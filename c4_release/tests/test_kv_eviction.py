"""Phase 7.F.2 — runtime KV eviction byte-identity gate.

Acceptance criteria for the ``compile_full_vm(..., kv_eviction_policy=...)``
flag wiring:

1. Default policy=OFF produces byte-identical logits to no flag passed at
   all (the historical-baseline guarantee).
2. policy=STATIC_LIVENESS produces byte-identical logits to OFF on a small
   smoke input. This is the safety property: a conservative analyzer
   should never mark a still-live entry dead, so eviction never changes
   model output.
3. When the analyzer marks a position evictable at step S, the K/V cache
   rows at that position are zeroed after the step boundary hook fires.
4. Eviction decisions are deterministic: two analyses of the same op
   corpus produce the same evictable sets and the runtime apply_eviction
   zeros the same rows on each call.
5. Spec-decode and main-decode reach identical eviction decisions when
   given the same ``KVEvictionState`` (decisions are a pure function of
   the precomputed report + step index, not of any runtime tensor).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.kv_eviction import (
    KVEvictionPolicy,
    KVEvictionState,
    apply_eviction,
    build_state_from_report,
)
from neural_vm.kv_liveness_analyzer import (
    KVEntry,
    LivenessReport,
    analyze_kv_liveness,
)
from neural_vm.base_layers import PureAttention
from neural_vm.embedding import E


# ---------------------------------------------------------------------------
# Shared smoke inputs
# ---------------------------------------------------------------------------


# Five small token-id inputs that exercise the byte-identity gate. Each is
# a [1, seq_len] sequence of legal vocab IDs (vocab=276). They're picked to
# stay short (<=10 tokens) so a single ``compile_full_vm`` is enough to
# evaluate all of them quickly.
_SMOKE_INPUTS = [
    torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
    torch.tensor([[10, 11, 12]], dtype=torch.long),
    torch.tensor([[100, 50, 25, 12]], dtype=torch.long),
    torch.tensor([[5, 5, 5, 5, 5, 5]], dtype=torch.long),
    torch.tensor([[200, 201, 202, 203, 204, 205, 206]], dtype=torch.long),
]


def _logits_for_model(model, token_ids: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return the post-head logits tensor."""
    with torch.no_grad():
        return model(token_ids)


# ---------------------------------------------------------------------------
# Session fixtures: compile each model once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_no_flag():
    """Baseline: ``compile_full_vm()`` with no eviction flag passed."""
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm
    model, layout = compile_full_vm(disk_cache=False)
    return model, layout


@pytest.fixture(scope="module")
def model_off():
    """``policy=OFF`` model — must match the baseline byte-for-byte."""
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm
    model, layout = compile_full_vm(
        disk_cache=False, kv_eviction_policy=KVEvictionPolicy.OFF,
    )
    return model, layout


@pytest.fixture(scope="module")
def model_static_liveness():
    """``policy=STATIC_LIVENESS`` model with the analyzer attached."""
    from neural_vm.unified_compiler.full_vm_compiler import compile_full_vm
    model, layout = compile_full_vm(
        disk_cache=False,
        kv_eviction_policy=KVEvictionPolicy.STATIC_LIVENESS,
        kv_eviction_n_steps=8,
    )
    return model, layout


# ---------------------------------------------------------------------------
# Test 1: default OFF == no flag.
# ---------------------------------------------------------------------------


def test_default_off_matches_no_flag(model_no_flag, model_off):
    """``compile_full_vm()`` (no flag) and ``compile_full_vm(policy=OFF)``
    must produce byte-identical logits for every smoke input."""
    base_model, base_layout = model_no_flag
    off_model, off_layout = model_off

    assert base_layout.d_model == off_layout.d_model
    assert base_layout.n_layers == off_layout.n_layers

    for idx, ids in enumerate(_SMOKE_INPUTS):
        base_logits = _logits_for_model(base_model, ids)
        off_logits = _logits_for_model(off_model, ids)
        assert base_logits.shape == off_logits.shape, (
            f"input {idx} shape mismatch: base={base_logits.shape} "
            f"off={off_logits.shape}"
        )
        # Strict bytewise equality.
        assert torch.equal(base_logits, off_logits), (
            f"input {idx}: default OFF diverged from no-flag baseline "
            f"(max abs diff = {(base_logits - off_logits).abs().max().item()})"
        )


# ---------------------------------------------------------------------------
# Test 2: STATIC_LIVENESS == OFF on the smoke corpus.
# ---------------------------------------------------------------------------


def test_static_liveness_byte_identical_to_off(model_off, model_static_liveness):
    """The analyzer is conservative-by-construction, so policy=STATIC_LIVENESS
    must produce logits byte-identical to policy=OFF on the smoke corpus."""

    off_model, _ = model_off
    sl_model, _ = model_static_liveness

    for idx, ids in enumerate(_SMOKE_INPUTS):
        off_logits = _logits_for_model(off_model, ids)
        sl_logits = _logits_for_model(sl_model, ids)
        assert off_logits.shape == sl_logits.shape, (
            f"input {idx} shape mismatch: off={off_logits.shape} "
            f"sl={sl_logits.shape}"
        )
        assert torch.equal(off_logits, sl_logits), (
            f"input {idx}: STATIC_LIVENESS diverged from OFF baseline "
            f"(max abs diff = {(off_logits - sl_logits).abs().max().item()})"
        )


# ---------------------------------------------------------------------------
# Test 3: Evicted entries are zero'd in K/V cache after the hook fires.
# ---------------------------------------------------------------------------


def test_evicted_rows_are_zeroed_after_step_boundary():
    """An evictable position at step S must end up as a zero K/V row after
    :func:`apply_eviction` is called for that step."""

    # Build a state directly: at step 2 we mark positions {1, 3} evictable.
    state = KVEvictionState(
        policy=KVEvictionPolicy.STATIC_LIVENESS,
        layer_idx=0,
        evictable_positions_at_step={2: {1, 3}},
    )

    # Construct a stand-in attention with explicit K_cache / V_cache tensors
    # of shape [B, H, S_kv, HD] (the canonical layout the eviction pass
    # operates on).
    class _CacheBox:
        pass

    attn = _CacheBox()
    H = 2
    HD = 4
    S_kv = 5
    attn.K_cache = torch.arange(1, 1 + S_kv * H * HD, dtype=torch.float).view(1, H, S_kv, HD)
    attn.V_cache = torch.arange(101, 101 + S_kv * H * HD, dtype=torch.float).view(1, H, S_kv, HD)

    # Snapshot the rows we expect to stay live.
    untouched_positions = {0, 2, 4}
    pre_K_snap = {p: attn.K_cache[0, :, p, :].clone() for p in untouched_positions}
    pre_V_snap = {p: attn.V_cache[0, :, p, :].clone() for p in untouched_positions}

    zeroed = apply_eviction(attn, state, step_idx=2)

    # Two positions, both heads → 2 rows total per K and V tensor (rows are
    # indexed by position; both heads share the position dim).
    assert zeroed == 2, f"expected 2 rows zeroed, got {zeroed}"

    # Evicted rows are all-zero across every head.
    for pos in (1, 3):
        assert torch.all(attn.K_cache[0, :, pos, :] == 0), (
            f"K_cache position {pos} not zeroed: {attn.K_cache[0, :, pos, :]}"
        )
        assert torch.all(attn.V_cache[0, :, pos, :] == 0), (
            f"V_cache position {pos} not zeroed: {attn.V_cache[0, :, pos, :]}"
        )

    # Untouched rows are bitwise unchanged.
    for pos in untouched_positions:
        assert torch.equal(attn.K_cache[0, :, pos, :], pre_K_snap[pos]), (
            f"K_cache position {pos} was modified unexpectedly"
        )
        assert torch.equal(attn.V_cache[0, :, pos, :], pre_V_snap[pos]), (
            f"V_cache position {pos} was modified unexpectedly"
        )

    # Bookkeeping: state reflects the eviction.
    assert state.total_evictions == 2
    assert {1, 3} <= state.evicted_positions

    # Hook is a no-op (zero rows zeroed) at a different step.
    pre_total = state.total_evictions
    zeroed_at_other = apply_eviction(attn, state, step_idx=7)
    assert zeroed_at_other == 0
    assert state.total_evictions == pre_total


# ---------------------------------------------------------------------------
# Test 4: Determinism — same decisions across 2 runs.
# ---------------------------------------------------------------------------


def test_eviction_decisions_are_deterministic_across_runs():
    """Two independent analyses of the same op corpus -> same evictable sets,
    and two ``apply_eviction`` calls -> same rows zeroed (both at the
    ``state.evicted_positions`` level and on the actual K/V tensors)."""

    # Minimal fake op corpus: an FFN-only op writes TEMP_SCRATCH at step 0,
    # plus an attention head reading some other dim. This is enough to
    # exercise both the per-step scratch path and the (layer, head) universe
    # inference.
    from neural_vm.unified_compiler.ir import CompilerIR, FFNRule

    def _build_op(name: str):
        # Build a deliberately new op each call so we don't share state
        # through the IR.
        class _Op:
            pass

        op = _Op()
        op.name = name
        op.reads = {"OUTPUT_LO"}
        op.writes = {"TEMP_SCRATCH"}
        op.step_idx = 0
        op.layer_idx = 1
        ir = CompilerIR()
        ir.layer(0).ffn.append(
            FFNRule.constant_write(
                conditions=(("OUTPUT_LO", 1.0),),
                threshold=0.5,
                writes=(("TEMP_SCRATCH+0", 1.0),),
                name=f"r_{name}",
            )
        )
        op.compiler_ir = ir
        return op

    # Run the analyzer twice -- same op corpus, same n_steps.
    report_a = analyze_kv_liveness([_build_op("a")], n_steps=4)
    report_b = analyze_kv_liveness([_build_op("a")], n_steps=4)

    # Reports should agree per step.
    assert report_a.evictable_at_step.keys() == report_b.evictable_at_step.keys()
    for step in report_a.evictable_at_step:
        assert report_a.evictable_at_step[step] == report_b.evictable_at_step[step], (
            f"step {step} evictable set differs between runs:\n"
            f"  run A: {report_a.evictable_at_step[step]}\n"
            f"  run B: {report_b.evictable_at_step[step]}"
        )

    state_a = build_state_from_report(report_a, layer_idx=1)
    state_b = build_state_from_report(report_b, layer_idx=1)

    # Two states from the two reports agree per step.
    assert (
        state_a.evictable_positions_at_step
        == state_b.evictable_positions_at_step
    )

    # apply_eviction on identical caches must zero the same rows across runs.
    def _make_attn():
        class _Box:
            pass

        attn = _Box()
        attn.K_cache = torch.arange(1, 1 + 8, dtype=torch.float).view(1, 1, 4, 2).clone()
        attn.V_cache = torch.arange(101, 101 + 8, dtype=torch.float).view(1, 1, 4, 2).clone()
        return attn

    attn_a = _make_attn()
    attn_b = _make_attn()
    # Use a step that the analyzer actually populated (if any). Cover all
    # plausible steps -- determinism must hold step-by-step.
    for step in range(4):
        # Reset bookkeeping to compare per-step row-zeroing exactly. Cloning
        # the cache so a previous step's zeroing doesn't leak forward.
        attn_a_step = _make_attn()
        attn_b_step = _make_attn()
        zeroed_a = apply_eviction(attn_a_step, state_a, step_idx=step)
        zeroed_b = apply_eviction(attn_b_step, state_b, step_idx=step)
        assert zeroed_a == zeroed_b, (
            f"step {step}: zeroed-row count differs (a={zeroed_a} b={zeroed_b})"
        )
        assert torch.equal(attn_a_step.K_cache, attn_b_step.K_cache), (
            f"step {step}: K_cache diverges across deterministic runs"
        )
        assert torch.equal(attn_a_step.V_cache, attn_b_step.V_cache), (
            f"step {step}: V_cache diverges across deterministic runs"
        )


# ---------------------------------------------------------------------------
# Test 5: Spec-decode and main-decode share the eviction state.
# ---------------------------------------------------------------------------


def test_spec_decode_and_main_decode_share_eviction_state():
    """The eviction state is a pure function of (precomputed report, step
    index). The spec-decode path and the main-decode path therefore reach
    identical decisions when handed the same state — we simulate this by
    running ``apply_eviction`` from both call sites and asserting the
    K/V tensors end up bitwise equal."""

    # Single state shared by both paths.
    state = KVEvictionState(
        policy=KVEvictionPolicy.STATIC_LIVENESS,
        layer_idx=3,
        evictable_positions_at_step={0: {2}, 1: {0, 1}, 2: {0, 4}},
    )

    def _fresh_attn():
        class _Box:
            pass

        attn = _Box()
        # Independent tensor for each "decode path" so we can compare them
        # after each step's apply_eviction.
        attn.K_cache = torch.arange(1, 1 + 6 * 3, dtype=torch.float).view(1, 1, 6, 3).clone()
        attn.V_cache = torch.arange(101, 101 + 6 * 3, dtype=torch.float).view(1, 1, 6, 3).clone()
        return attn

    main_attn = _fresh_attn()
    spec_attn = _fresh_attn()

    for step in (0, 1, 2):
        main_zeroed = apply_eviction(main_attn, state, step_idx=step)
        spec_zeroed = apply_eviction(spec_attn, state, step_idx=step)
        # Same decision count (rows zeroed).
        assert main_zeroed == spec_zeroed, (
            f"step {step}: main and spec decoders disagree on row count "
            f"(main={main_zeroed} spec={spec_zeroed})"
        )
        # Same K/V tensor state after the call.
        assert torch.equal(main_attn.K_cache, spec_attn.K_cache), (
            f"step {step}: K_cache mismatch between main and spec decode paths"
        )
        assert torch.equal(main_attn.V_cache, spec_attn.V_cache), (
            f"step {step}: V_cache mismatch between main and spec decode paths"
        )

    # The shared state's bookkeeping is also identical (it was incremented
    # twice — once per call — but the *decisions* recorded are the same).
    assert state.evicted_positions == {0, 1, 2, 4}
    # Each call site zeroed the same 5 distinct (step, position) pairs:
    #   step 0 -> {2} (1)
    #   step 1 -> {0, 1} (2)
    #   step 2 -> {0, 4} (2)
    assert state.total_evictions == 2 * (1 + 2 + 2)
