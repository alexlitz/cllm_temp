"""Phase 7.F.1 — KV liveness analyzer unit tests.

These tests exercise the high-confidence categories handled by the
analyzer's first implementation:

  * FFN-only op with no cross-step reader -> entries evictable.
  * Per-step scratch (TEMP_*) is evictable immediately after its step.
  * PREV_STEP dims become evictable one step after their write.
  * Attention K/V read keeps the entry live across all steps.
  * Cycle-member dims stay live by default; opt-out flag flips that.
  * Register-overwrite scenario produces evictable entries (no future
    read, but a later op writes the same dim).
  * Coverage is reported as a fraction in [0, 1].
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

import pytest

from neural_vm.kv_liveness_analyzer import (
    KVEntry,
    LivenessReport,
    analyze_kv_liveness,
)
from neural_vm.unified_compiler.ir import (
    AttentionHeadIR,
    AttentionOp,
    CompilerIR,
    FFNOp,
    FFNRule,
)
from neural_vm.unified_compiler.primitives import (
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
)


class _FakeOp:
    """Minimal Operation stand-in compatible with the analyzer's interface."""

    def __init__(
        self,
        name: str,
        *,
        reads: Optional[Iterable[str]] = None,
        writes: Optional[Iterable[str]] = None,
        compiler_ir=None,
        step_idx: Optional[int] = None,
        layer_idx: Optional[int] = None,
    ):
        self.name = name
        self.reads: Set[str] = set(reads or ())
        self.writes: Set[str] = set(writes or ())
        self.compiler_ir = compiler_ir
        self.step_idx = step_idx
        self.layer_idx = layer_idx


def _ffn_op_with_rule(
    *,
    conditions,
    threshold,
    writes,
    name="r",
    scope=None,
) -> CompilerIR:
    """Wrap a single FFNRule into a CompilerIR for use as compiler_ir."""

    ir = CompilerIR()
    ir.layer(0).ffn.append(
        FFNRule.constant_write(
            conditions=conditions,
            threshold=threshold,
            writes=writes,
            name=name,
            scope=scope,
        )
    )
    return ir


def _attn_op_reading_dim(
    *,
    head_idx: int,
    k_dim: int,
    v_dim: int,
    o_dim: int,
) -> CompilerIR:
    """Build a CompilerIR with one attention head reading dim k_dim / v_dim."""

    ir = CompilerIR()
    spec = DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=(AP(0, 0, 1.0),),
        k=(AP(0, k_dim, 1.0),),
        v=(AP(0, v_dim, 1.0),),
        o=(AO(o_dim, 0, 1.0),),
    )
    ir.layer(0).attention.append(spec)
    return ir


# ---------------------------------------------------------------------------
# Basic plumbing
# ---------------------------------------------------------------------------


def test_empty_ops_produces_empty_report():
    report = analyze_kv_liveness([], n_steps=3)
    assert isinstance(report, LivenessReport)
    assert report.evictable_at_step == {0: set(), 1: set(), 2: set()}
    assert report.cycle_conservative == set()
    assert report.coverage == 0.0


def test_report_coverage_in_unit_interval():
    op = _FakeOp(
        "scratch_write",
        writes={"TEMP_SCRATCH"},
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("TEMP_SCRATCH+0", 1.0),),
        ),
    )
    report = analyze_kv_liveness([op], n_steps=3)
    assert 0.0 <= report.coverage <= 1.0


# ---------------------------------------------------------------------------
# Category 1: FFN-only op with no cross-step reader -> evictable.
# ---------------------------------------------------------------------------


def test_ffn_only_temp_write_is_evictable_after_step():
    """A TEMP dim written by an FFN-only op with no later reader becomes
    evictable at the same step (per-step scratch category)."""

    op = _FakeOp(
        "ffn_writes_temp",
        reads={"OUTPUT_LO"},
        writes={"TEMP"},
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("TEMP+0", 1.0),),
        ),
    )
    report = analyze_kv_liveness([op], n_steps=2)
    # At least one KVEntry whose dim_name is "TEMP" should be evictable.
    temp_evictables = {
        e for s in report.evictable_at_step.values() for e in s
        if e.dim_name == "TEMP"
    }
    assert temp_evictables, (
        "Expected TEMP entries to be evictable; got none. "
        f"Report: {report}"
    )


# ---------------------------------------------------------------------------
# Category 2: AttentionHeadIR reads a dim => entry stays live everywhere.
# ---------------------------------------------------------------------------


def test_attention_kv_read_keeps_entry_live():
    """An attention head reading a dim at step T keeps every KV entry
    for that dim at positions <= T live (since causal attention can
    reach back to them). The analyzer is allowed to evict positions
    strictly greater than T because no op at a later step could attend
    to them — this matches the static guarantee."""

    # Op 1: writes TEMP at step 0.
    writer = _FakeOp(
        "temp_writer",
        writes={"TEMP"},
        step_idx=0,
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("TEMP+0", 1.0),),
        ),
    )
    # Op 2: attention head reads TEMP via its K projection at step 5.
    reader = _FakeOp(
        "attn_reads_temp",
        reads={"TEMP"},
        layer_idx=1,
        step_idx=5,
        compiler_ir=_attn_op_reading_dim(
            head_idx=0, k_dim=10, v_dim=11, o_dim=20,
        ),
    )
    report = analyze_kv_liveness([writer, reader], n_steps=10)
    # KV entries at positions 0..5 for dim TEMP must stay live (the
    # step-5 reader can attend back to them).
    evicted_temp_positions = {
        e.position
        for s in report.evictable_at_step.values()
        for e in s
        if e.dim_name == "TEMP"
    }
    illegal = evicted_temp_positions & set(range(6))
    assert not illegal, (
        f"Positions {illegal} were evicted even though a step-5 attention "
        f"head reads TEMP and could attend back to them."
    )


# ---------------------------------------------------------------------------
# Category 3: PREV_STEP dim is evictable one step after the write.
# ---------------------------------------------------------------------------


def test_prev_step_dim_evictable_after_consuming_step():
    writer = _FakeOp(
        "writes_prev",
        writes={"OUTPUT_LO_PREV_STEP"},
        step_idx=0,
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("OUTPUT_LO_PREV_STEP+0", 1.0),),
        ),
    )
    report = analyze_kv_liveness([writer], n_steps=5)
    # The entry written at step 0 should become evictable at step 1, not
    # at step 0 itself.
    step0_evictable_prev = {
        e for e in report.evictable_at_step[0]
        if e.dim_name == "OUTPUT_LO_PREV_STEP" and e.position == 0
    }
    step1_evictable_prev = {
        e for e in report.evictable_at_step[1]
        if e.dim_name == "OUTPUT_LO_PREV_STEP" and e.position == 0
    }
    # PREV_STEP entries appear in *later* steps (position == step of
    # write). Confirm they're evicted at step+1 specifically.
    assert step1_evictable_prev, (
        f"Expected PREV_STEP entry written at step 0 to be evicted at "
        f"step 1; evictables at step 1: {report.evictable_at_step[1]}"
    )
    # Strict: step 0 should NOT contain a PREV_STEP entry whose position
    # is 0 (because step 0 itself might still be consuming it).
    assert not step0_evictable_prev, (
        "PREV_STEP entry should not be evicted at the same step it was "
        "written; it might still be consumed by step+1's attention."
    )


# ---------------------------------------------------------------------------
# Category 4: Cycle-member dim stays live under the conservative flag.
# ---------------------------------------------------------------------------


def test_cycle_member_dim_is_conservative_kept_by_default():
    """When a dim D is both read and written by the same op, it forms a
    self-loop in the use/def graph; under the conservative flag the
    analyzer must keep the entry live, not evict it."""

    cycle_op = _FakeOp(
        "cycle_op",
        reads={"AX_CARRY"},
        writes={"AX_CARRY"},
        compiler_ir=_ffn_op_with_rule(
            conditions=(("AX_CARRY", 1.0),),
            threshold=0.5,
            writes=(("AX_CARRY+0", 1.0),),
        ),
    )
    report = analyze_kv_liveness([cycle_op], n_steps=3)
    cycle_ax_entries = {
        e for e in report.cycle_conservative if e.dim_name == "AX_CARRY"
    }
    assert cycle_ax_entries, (
        "Expected AX_CARRY entries to be conservatively kept due to "
        f"self-loop; got cycle_conservative={report.cycle_conservative}"
    )
    # And NOT evictable.
    for entries in report.evictable_at_step.values():
        for e in entries:
            assert e.dim_name != "AX_CARRY", (
                f"AX_CARRY should not be evictable while in a cycle; "
                f"entry={e}"
            )


def test_cycle_conservative_can_be_opted_out():
    """With treat_cycle_members_conservative=False the analyzer is allowed
    to evict cycle-member dims based on the per-step category. For a
    TEMP-prefixed cycle dim this should yield at least one evictable
    entry."""

    cycle_op = _FakeOp(
        "cycle_op",
        reads={"TEMP_CYCLE"},
        writes={"TEMP_CYCLE"},
        compiler_ir=_ffn_op_with_rule(
            conditions=(("TEMP_CYCLE", 1.0),),
            threshold=0.5,
            writes=(("TEMP_CYCLE+0", 1.0),),
        ),
    )
    report = analyze_kv_liveness(
        [cycle_op],
        n_steps=3,
        treat_cycle_members_conservative=False,
    )
    # Cycle conservative should be empty when the flag is False.
    assert report.cycle_conservative == set()
    evictable_temp = {
        e for s in report.evictable_at_step.values() for e in s
        if e.dim_name == "TEMP_CYCLE"
    }
    # With the per-step scratch category active, evictable should be
    # non-empty even though the dim is in a self-loop.
    assert evictable_temp, (
        f"With cycle conservatism disabled, TEMP_CYCLE entries should "
        f"flow into the scratch eviction path; got {report}"
    )


# ---------------------------------------------------------------------------
# Category 5: Register-overwrite scenario.
# ---------------------------------------------------------------------------


def test_register_overwrite_makes_prior_entries_evictable():
    """Two ops write the same dim; no future K-read. The first op's
    entries should be evictable thanks to the semantic-overwrite rule."""

    # Op 1 writes REG_BANK at step 0.
    writer_a = _FakeOp(
        "writer_a",
        writes={"REG_BANK"},
        step_idx=0,
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("REG_BANK+0", 1.0),),
            name="ra",
        ),
    )
    # Op 2 writes REG_BANK at step 2 (later).
    writer_b = _FakeOp(
        "writer_b",
        writes={"REG_BANK"},
        step_idx=2,
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("REG_BANK+0", 1.0),),
            name="rb",
        ),
    )
    report = analyze_kv_liveness([writer_a, writer_b], n_steps=4)

    # No future attention read for REG_BANK, future_writes covers it.
    # Entries at step 0 should be evictable.
    step0 = {e for e in report.evictable_at_step[0] if e.dim_name == "REG_BANK"}
    assert step0, (
        f"Expected REG_BANK entries to be evictable at step 0 (later "
        f"overwrite, no future read); got {report.evictable_at_step[0]}"
    )


# ---------------------------------------------------------------------------
# Category 6: Unknown dim with no scratch/prev-step/overwrite => kept.
# ---------------------------------------------------------------------------


def test_unknown_dim_is_kept_conservatively():
    """A dim that doesn't fit any high-confidence category should be
    kept (not evicted). This guarantees the analyzer never over-evicts."""

    op = _FakeOp(
        "writes_unknown",
        writes={"SOME_RANDOM_DIM"},
        step_idx=0,
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("SOME_RANDOM_DIM+0", 1.0),),
        ),
    )
    report = analyze_kv_liveness([op], n_steps=3)
    for entries in report.evictable_at_step.values():
        for e in entries:
            assert e.dim_name != "SOME_RANDOM_DIM", (
                f"Expected SOME_RANDOM_DIM to stay live (no category "
                f"matched); got evictable entry {e}"
            )


# ---------------------------------------------------------------------------
# Category 7: Smoke / API contract.
# ---------------------------------------------------------------------------


def test_layers_and_heads_kwarg_restricts_universe():
    op = _FakeOp(
        "writes_temp",
        writes={"TEMP"},
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("TEMP+0", 1.0),),
        ),
    )
    report = analyze_kv_liveness(
        [op], n_steps=2, layers=[3, 4], heads=[7],
    )
    for entries in report.evictable_at_step.values():
        for e in entries:
            assert e.layer in {3, 4}
            assert e.head == 7


def test_kv_entry_is_frozen_and_hashable():
    e = KVEntry(layer=0, position=1, head=2, dim_name="TEMP")
    assert hash(e) == hash(KVEntry(layer=0, position=1, head=2, dim_name="TEMP"))
    with pytest.raises(Exception):
        e.layer = 9  # type: ignore[misc]


def test_attention_head_seen_drives_layer_head_universe():
    """When ops include an AttentionHeadIR, the analyzer derives
    (layer, head) coverage from it instead of defaulting to (0, 0)."""

    op = _FakeOp(
        "attn_head_op",
        reads={"TEMP_OTHER"},
        layer_idx=5,
        compiler_ir=_attn_op_reading_dim(
            head_idx=3, k_dim=8, v_dim=9, o_dim=12,
        ),
    )
    # A second op contributing a TEMP write so the analyzer has something
    # to potentially evict.
    writer = _FakeOp(
        "ffn_writer",
        writes={"TEMP_X"},
        compiler_ir=_ffn_op_with_rule(
            conditions=(("OUTPUT_LO", 1.0),),
            threshold=0.5,
            writes=(("TEMP_X+0", 1.0),),
        ),
    )
    report = analyze_kv_liveness([op, writer], n_steps=2)
    layers_seen = {e.layer for s in report.evictable_at_step.values() for e in s}
    heads_seen = {e.head for s in report.evictable_at_step.values() for e in s}
    # The attention-bearing op declared layer_idx=5, head_idx=3.
    assert 5 in layers_seen, f"Expected layer 5 in evictable layers; got {layers_seen}"
    assert 3 in heads_seen, f"Expected head 3 in evictable heads; got {heads_seen}"
