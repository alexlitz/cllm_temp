"""Phase 8.E.2 — overwrite-map unit tests.

These tests exercise the per-step overwrite map produced by
``neural_vm.kv_overwrite_map.build_overwrite_map``:

  * categorize_dim assigns dims to the right §8.E.1 bucket.
  * REGISTER / MEM_CELL / OUTPUT_SLOT dims overwrite at the next
    later writer's step.
  * TRANSIENT_SCRATCH dims become dead at end-of-step.
  * PREV_STEP dims become dead at step+1.
  * UNKNOWN dims produce no overwrite entry (conservative keep).
  * Every-step ops contribute writes at every step.
  * The earliest later writer wins for REGISTER overwrites.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

import pytest

from neural_vm.kv_overwrite_map import (
    OverwriteCategory,
    OverwriteEntry,
    OverwriteMap,
    build_overwrite_map,
    categorize_dim,
)
from neural_vm.unified_compiler.ir import (
    CompilerIR,
    FFNRule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeOp:
    """Minimal Operation stand-in compatible with the builder's interface."""

    def __init__(
        self,
        name: str,
        *,
        reads: Optional[Iterable[str]] = None,
        writes: Optional[Iterable[str]] = None,
        compiler_ir=None,
        step_idx: Optional[object] = None,
        layer_idx: Optional[int] = None,
    ):
        self.name = name
        self.reads: Set[str] = set(reads or ())
        self.writes: Set[str] = set(writes or ())
        self.compiler_ir = compiler_ir
        self.step_idx = step_idx
        self.layer_idx = layer_idx


def _ir_writing(dim_with_offset: str) -> CompilerIR:
    """Build a one-rule CompilerIR that writes the given dim."""

    ir = CompilerIR()
    ir.layer(0).ffn.append(
        FFNRule.constant_write(
            conditions=(("ANY", 1.0),),
            threshold=0.5,
            writes=((dim_with_offset, 1.0),),
        )
    )
    return ir


# ---------------------------------------------------------------------------
# categorize_dim
# ---------------------------------------------------------------------------


class TestCategorizeDim:
    def test_register_prefixes(self):
        # ALU_TEMP* / MUL_TEMP* etc. fall under TRANSIENT_SCRATCH because the
        # scratch prefix is checked first — that's intentional. ALU_OP /
        # ADDR_KEY etc. remain in REGISTER.
        for name in ("REG_PC", "REG_AX", "PC_byte0", "AX_BYTE0", "SP_byte0",
                     "BP_byte0", "MARK_STACK0", "OP_PSH", "ADDR_KEY",
                     "ALU_OP", "FETCH_LO", "BYTE_INDEX_0", "EMBED_LO"):
            assert categorize_dim(name) is OverwriteCategory.REGISTER, name

    def test_register_exact(self):
        for name in ("PC", "AX", "SP", "BP"):
            assert categorize_dim(name) is OverwriteCategory.REGISTER, name

    def test_mem_cell(self):
        assert categorize_dim("MEM_addr0") is OverwriteCategory.MEM_CELL
        assert categorize_dim("MEM_BYTE0") is OverwriteCategory.MEM_CELL

    def test_output_slot(self):
        assert categorize_dim("OUTPUT_LO") is OverwriteCategory.OUTPUT_SLOT
        assert categorize_dim("OUTPUT_HI") is OverwriteCategory.OUTPUT_SLOT

    def test_transient_scratch_prefixes(self):
        for name in ("TEMP", "TEMP_FOO", "ALU_TEMP_X", "MUL_TEMP",
                     "DIV_TEMP", "MUL_ACCUM_LO", "DIV_STAGING",
                     "MEM_STAGING", "AX_FULL_LO"):
            assert categorize_dim(name) is OverwriteCategory.TRANSIENT_SCRATCH, name

    def test_transient_scratch_suffixes(self):
        # _THIS_STEP / _SCRATCH win even when the prefix would otherwise
        # match output-slot — they're per-step transients.
        assert categorize_dim("OUTPUT_HI_THIS_STEP") is (
            OverwriteCategory.TRANSIENT_SCRATCH
        )
        assert categorize_dim("X_SCRATCH") is OverwriteCategory.TRANSIENT_SCRATCH

    def test_prev_step_wins_over_other_categories(self):
        # PREV_STEP suffix wins over output-slot prefix etc.
        assert categorize_dim("OUTPUT_LO_PREV_STEP") is OverwriteCategory.PREV_STEP
        assert categorize_dim("OUTPUT_HI_PREV_STEP") is OverwriteCategory.PREV_STEP
        assert categorize_dim("FOO_PREV") is OverwriteCategory.PREV_STEP
        assert categorize_dim("BAR_LAST_STEP") is OverwriteCategory.PREV_STEP

    def test_unknown_falls_through(self):
        assert categorize_dim("RANDOM_UNRELATED") is OverwriteCategory.UNKNOWN
        assert categorize_dim("MY_CUSTOM_DIM") is OverwriteCategory.UNKNOWN


# ---------------------------------------------------------------------------
# build_overwrite_map basic plumbing
# ---------------------------------------------------------------------------


def test_empty_ops_produces_empty_map():
    m = build_overwrite_map([], n_steps=5)
    assert isinstance(m, OverwriteMap)
    assert m.total_entries() == 0
    assert m.n_steps == 5


def test_unknown_dim_produces_no_entry():
    op = _FakeOp(
        "writes_random",
        writes={"RANDOM_DIM"},
        compiler_ir=_ir_writing("RANDOM_DIM+0"),
        step_idx=0,
    )
    m = build_overwrite_map([op], n_steps=3)
    assert m.total_entries() == 0


# ---------------------------------------------------------------------------
# REGISTER / MEM_CELL / OUTPUT_SLOT — overwrite at next writer's step
# ---------------------------------------------------------------------------


def test_register_overwrite_at_next_writer():
    """PC written at step 0, then again at step 2 — entry for position 0
    overwrites at step 2."""

    op0 = _FakeOp(
        "write_pc_step0",
        writes={"REG_PC"},
        compiler_ir=_ir_writing("REG_PC+0"),
        step_idx=0,
    )
    op2 = _FakeOp(
        "write_pc_step2",
        writes={"REG_PC"},
        compiler_ir=_ir_writing("REG_PC+0"),
        step_idx=2,
    )
    m = build_overwrite_map([op0, op2], n_steps=4)
    entries = m.entries_at_step(2)
    assert any(
        e.position == 0 and e.dim_name == "REG_PC"
        and e.category is OverwriteCategory.REGISTER
        for e in entries
    )


def test_register_earliest_later_writer_wins():
    op0 = _FakeOp("a", writes={"REG_PC"},
                  compiler_ir=_ir_writing("REG_PC+0"), step_idx=0)
    op1 = _FakeOp("b", writes={"REG_PC"},
                  compiler_ir=_ir_writing("REG_PC+0"), step_idx=1)
    op3 = _FakeOp("c", writes={"REG_PC"},
                  compiler_ir=_ir_writing("REG_PC+0"), step_idx=3)
    m = build_overwrite_map([op0, op1, op3], n_steps=4)
    pos0_entries = [
        e for e in m.entries_at_step(1)
        if e.position == 0 and e.dim_name == "REG_PC"
    ]
    assert len(pos0_entries) == 1
    assert pos0_entries[0].overwrite_step == 1


def test_register_no_later_writer_lands_in_none_bucket():
    op0 = _FakeOp("only", writes={"REG_PC"},
                  compiler_ir=_ir_writing("REG_PC+0"), step_idx=0)
    m = build_overwrite_map([op0], n_steps=3)
    # No future writer at step 1+ — overwrite_step is None.
    unread = m.entries_at_step(None)
    assert any(
        e.position == 0 and e.dim_name == "REG_PC" for e in unread
    )


def test_mem_cell_overwrite_at_next_writer():
    op0 = _FakeOp("a", writes={"MEM_addr0"},
                  compiler_ir=_ir_writing("MEM_addr0+0"), step_idx=0)
    op2 = _FakeOp("b", writes={"MEM_addr0"},
                  compiler_ir=_ir_writing("MEM_addr0+0"), step_idx=2)
    m = build_overwrite_map([op0, op2], n_steps=4)
    entries = m.entries_at_step(2)
    assert any(
        e.position == 0 and e.dim_name == "MEM_addr0"
        and e.category is OverwriteCategory.MEM_CELL
        for e in entries
    )


def test_output_slot_overwrite_at_next_writer():
    op0 = _FakeOp("a", writes={"OUTPUT_LO"},
                  compiler_ir=_ir_writing("OUTPUT_LO+0"), step_idx=0)
    op3 = _FakeOp("b", writes={"OUTPUT_LO"},
                  compiler_ir=_ir_writing("OUTPUT_LO+0"), step_idx=3)
    m = build_overwrite_map([op0, op3], n_steps=5)
    entries = m.entries_at_step(3)
    assert any(
        e.position == 0 and e.dim_name == "OUTPUT_LO"
        and e.category is OverwriteCategory.OUTPUT_SLOT
        for e in entries
    )


# ---------------------------------------------------------------------------
# TRANSIENT_SCRATCH — dead at end-of-step regardless of any later writer
# ---------------------------------------------------------------------------


def test_transient_scratch_dead_at_position():
    op = _FakeOp(
        "scratch",
        writes={"TEMP"},
        compiler_ir=_ir_writing("TEMP+0"),
        step_idx=0,
    )
    m = build_overwrite_map([op], n_steps=3)
    entries = m.entries_at_step(0)
    assert any(
        e.dim_name == "TEMP" and e.position == 0
        and e.overwrite_step == 0
        and e.category is OverwriteCategory.TRANSIENT_SCRATCH
        for e in entries
    )


def test_transient_scratch_even_with_no_later_writer():
    op = _FakeOp(
        "scratch_only",
        writes={"AX_FULL_LO"},
        compiler_ir=_ir_writing("AX_FULL_LO+0"),
        step_idx=2,
    )
    m = build_overwrite_map([op], n_steps=5)
    entries = m.entries_at_step(2)
    assert any(
        e.dim_name == "AX_FULL_LO" and e.position == 2
        and e.category is OverwriteCategory.TRANSIENT_SCRATCH
        for e in entries
    )


# ---------------------------------------------------------------------------
# PREV_STEP — dead at step + 1
# ---------------------------------------------------------------------------


def test_prev_step_dead_at_next_step():
    op = _FakeOp(
        "prev_relay",
        writes={"OUTPUT_LO_PREV_STEP"},
        compiler_ir=_ir_writing("OUTPUT_LO_PREV_STEP+0"),
        step_idx=2,
    )
    m = build_overwrite_map([op], n_steps=5)
    entries = m.entries_at_step(3)
    assert any(
        e.dim_name == "OUTPUT_LO_PREV_STEP" and e.position == 2
        and e.overwrite_step == 3
        and e.category is OverwriteCategory.PREV_STEP
        for e in entries
    )


def test_prev_step_at_final_position_has_no_overwrite():
    """When PREV_STEP is written at the last step, no S+1 exists, so
    overwrite_step lands in the None bucket."""

    op = _FakeOp(
        "prev_relay_last",
        writes={"OUTPUT_LO_PREV_STEP"},
        compiler_ir=_ir_writing("OUTPUT_LO_PREV_STEP+0"),
        step_idx=4,
    )
    m = build_overwrite_map([op], n_steps=5)
    unread = m.entries_at_step(None)
    assert any(
        e.position == 4 and e.dim_name == "OUTPUT_LO_PREV_STEP"
        for e in unread
    )


# ---------------------------------------------------------------------------
# Every-step ops
# ---------------------------------------------------------------------------


def test_every_step_op_contributes_at_every_step():
    """An op with step_idx=None fires at every step — so a register
    written by it at step 0 is overwritten at step 1, 1 -> 2, etc."""

    op = _FakeOp(
        "decode_every_step",
        writes={"REG_PC"},
        compiler_ir=_ir_writing("REG_PC+0"),
        step_idx=None,
    )
    m = build_overwrite_map([op], n_steps=4)
    # position 0 -> overwrite at step 1; position 2 -> overwrite at 3.
    s1 = m.entries_at_step(1)
    assert any(e.position == 0 and e.dim_name == "REG_PC" for e in s1)
    s3 = m.entries_at_step(3)
    assert any(e.position == 2 and e.dim_name == "REG_PC" for e in s3)
    # Last position (3) has no later writer in 0..n_steps-1 -> None.
    none_bucket = m.entries_at_step(None)
    assert any(
        e.position == 3 and e.dim_name == "REG_PC" for e in none_bucket
    )


# ---------------------------------------------------------------------------
# AttentionHeadIR writes count via op.writes
# ---------------------------------------------------------------------------


def test_attention_o_writes_via_op_writes():
    """AttentionHeadIR W_o writes land on numeric residual dims; the IR
    can't carry dim *names*. Op.writes is the canonical declaration —
    so the builder must honour it even when there's no FFNRule write."""

    op = _FakeOp(
        "attn_write_pc",
        writes={"REG_PC"},
        step_idx=0,
    )
    op2 = _FakeOp(
        "later_write_pc",
        writes={"REG_PC"},
        step_idx=1,
    )
    m = build_overwrite_map([op, op2], n_steps=3)
    s1 = m.entries_at_step(1)
    assert any(
        e.position == 0 and e.dim_name == "REG_PC"
        and e.category is OverwriteCategory.REGISTER
        for e in s1
    )


# ---------------------------------------------------------------------------
# OverwriteMap accessor API
# ---------------------------------------------------------------------------


def test_entries_by_category_aggregates_across_steps():
    op0 = _FakeOp("a", writes={"REG_PC"},
                  compiler_ir=_ir_writing("REG_PC+0"), step_idx=0)
    op1 = _FakeOp("b", writes={"REG_PC"},
                  compiler_ir=_ir_writing("REG_PC+0"), step_idx=1)
    op2 = _FakeOp("c", writes={"REG_PC"},
                  compiler_ir=_ir_writing("REG_PC+0"), step_idx=2)
    m = build_overwrite_map([op0, op1, op2], n_steps=3)
    reg_entries = m.entries_by_category(OverwriteCategory.REGISTER)
    # Positions 0 and 1 should each produce an entry (positions 2 has no
    # later writer, lands in None bucket but still REGISTER category).
    positions = sorted(e.position for e in reg_entries)
    assert 0 in positions
    assert 1 in positions
    assert 2 in positions


def test_category_counts_partitions_total():
    op_reg = _FakeOp("a", writes={"REG_PC"},
                     compiler_ir=_ir_writing("REG_PC+0"), step_idx=0)
    op_reg2 = _FakeOp("b", writes={"REG_PC"},
                      compiler_ir=_ir_writing("REG_PC+0"), step_idx=1)
    op_scratch = _FakeOp("c", writes={"TEMP"},
                         compiler_ir=_ir_writing("TEMP+0"), step_idx=0)
    op_prev = _FakeOp("d", writes={"OUTPUT_LO_PREV_STEP"},
                      compiler_ir=_ir_writing("OUTPUT_LO_PREV_STEP+0"),
                      step_idx=0)
    m = build_overwrite_map([op_reg, op_reg2, op_scratch, op_prev], n_steps=3)
    counts = m.category_counts()
    assert sum(counts.values()) == m.total_entries()
    # We should see all three categories present.
    assert counts.get(OverwriteCategory.REGISTER, 0) >= 1
    assert counts.get(OverwriteCategory.TRANSIENT_SCRATCH, 0) >= 1
    assert counts.get(OverwriteCategory.PREV_STEP, 0) >= 1
