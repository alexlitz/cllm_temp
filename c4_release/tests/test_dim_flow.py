"""Tests for the dim-flow analyzer.

Covers:

* Synthetic FFN op: writers / readers / gate condition surfacing.
* Synthetic attention head: O write detection, Q/K/V read role tagging.
* Full-VM layout: STACK0_BYTE_VAL_1_LO returns L11 broadcast head_8
  as writer and L18 mem_generation as reader (matches A3.6 attribution).
* Unknown dim: returns empty list, does not crash.
"""

from __future__ import annotations

import warnings

import pytest

from c4_release.neural_vm.unified_compiler.dim_flow import (
    DimFlow,
    DimReader,
    DimWriter,
    enumerate_dim_readers,
    enumerate_dim_writers,
    find_zeroing_writers,
    format_report,
    trace_dim_flow,
)
from c4_release.neural_vm.unified_compiler.ir import FFNOp, FFNRule
from c4_release.neural_vm.unified_compiler.primitives import (
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
)


# ---------------------------------------------------------------------------
# Synthetic layout helpers
# ---------------------------------------------------------------------------


class _StubOp:
    """Minimal Operation-shaped stub for dim_flow tests."""

    def __init__(self, name, kind, compiler_ir, *, factory=None):
        self.name = name
        self.kind = kind
        self.compiler_ir = compiler_ir
        self.compiler_ir_factory = factory
        self.layer_idx = None


class _StubLayout:
    """Minimal ModelLayout-shaped stub.

    ``ops_per_layer`` is the per-layer op list; ``block_ops`` / ``model_ops``
    are flat. ``dim_positions`` / ``dim_sizes`` mirror ``ModelLayout``.
    """

    def __init__(self, ops_per_layer, dim_positions, dim_sizes,
                 block_ops=(), model_ops=()):
        self.ops_per_layer = ops_per_layer
        self.dim_positions = dim_positions
        self.dim_sizes = dim_sizes
        self.block_ops = list(block_ops)
        self.model_ops = list(model_ops)

    def resolve_block_op_layer(self, op):
        if op.layer_idx is not None:
            return op.layer_idx
        return -1


# ---------------------------------------------------------------------------
# Synthetic-FFN tests
# ---------------------------------------------------------------------------


def test_ffn_writer_and_reader_are_surfaced():
    """A single FFN op writing OUT_LO+2 from MARK_SP must show up as a
    writer at OUT_LO and as a reader (condition) at MARK_SP."""

    rule = FFNRule.constant_write(
        conditions=(("MARK_SP", 10.0),),
        threshold=5.0,
        writes=(("OUT_LO+2", 7.0),),
        name="rule_a",
    )
    op = _StubOp("op_a", "ffn", FFNOp(rules=[rule]))
    layout = _StubLayout(
        ops_per_layer=[[], [op]],  # op placed at layer 1
        dim_positions={"MARK_SP": 0, "OUT_LO": 10},
        dim_sizes={"MARK_SP": 1, "OUT_LO": 16},
    )

    writers = enumerate_dim_writers(layout, "OUT_LO")
    assert len(writers) == 1
    w = writers[0]
    assert w.layer_idx == 1
    assert w.op_name == "op_a"
    assert w.op_kind == "ffn"
    assert w.rule_name == "rule_a"
    assert w.offset == 2
    assert "+7" in w.write_value_expr

    # MARK_SP appears as a condition input to the rule.
    readers = enumerate_dim_readers(layout, "MARK_SP")
    assert len(readers) == 1
    r = readers[0]
    assert r.layer_idx == 1
    assert r.op_name == "op_a"
    assert r.read_role == "ffn_condition"
    assert r.weight == 10.0


def test_ffn_gate_and_gate_terms_classified():
    """Gate and gate_terms reads must surface with role ``ffn_gate``."""

    rule = FFNRule.gated_write(
        conditions=(("MARK_SP", 1.0),),
        threshold=0.5,
        gate="GATE_DIM",
        gate_terms=(("EXTRA_GATE", 2.0),),
        writes=(("OUT_LO+0", 1.0),),
        name="rule_g",
    )
    op = _StubOp("op_g", "ffn", FFNOp(rules=[rule]))
    layout = _StubLayout(
        ops_per_layer=[[op]],
        dim_positions={
            "MARK_SP": 0, "GATE_DIM": 1, "EXTRA_GATE": 2, "OUT_LO": 10,
        },
        dim_sizes={
            "MARK_SP": 1, "GATE_DIM": 1, "EXTRA_GATE": 1, "OUT_LO": 16,
        },
    )

    gate_readers = enumerate_dim_readers(layout, "GATE_DIM")
    assert len(gate_readers) == 1
    assert gate_readers[0].read_role == "ffn_gate"

    extra_readers = enumerate_dim_readers(layout, "EXTRA_GATE")
    assert len(extra_readers) == 1
    assert extra_readers[0].read_role == "ffn_gate"


# ---------------------------------------------------------------------------
# Synthetic-attention tests
# ---------------------------------------------------------------------------


def test_attention_o_write_and_qkv_roles():
    """An attention head that O-writes into OUT_LO and reads MARK_SP via
    Q must show up as a writer at OUT_LO and a reader (role=attn_Q) at
    MARK_SP."""

    dim_positions = {
        "MARK_SP": 0,    # 1 cell, dim 0
        "SOURCE": 10,    # 4 cells, dims 10..13
        "OUT_LO": 20,    # 4 cells, dims 20..23
    }
    dim_sizes = {"MARK_SP": 1, "SOURCE": 4, "OUT_LO": 4}

    spec = DeclarativeAttentionHeadSpec(
        head_idx=3,
        q=(AP(0, dim_positions["MARK_SP"], 5.0),),    # Q reads MARK_SP
        k=(AP(0, dim_positions["MARK_SP"], 5.0),),    # K reads MARK_SP
        v=(AP(0, dim_positions["SOURCE"], 1.0),),     # V reads SOURCE
        o=(AO(dim_positions["OUT_LO"] + 1, 0, 2.0),), # O writes OUT_LO+1
    )

    # Wrap in an FFNOp-like-but-attention container: dim_flow's IR walker
    # recognises an object with .layers[].attention.rules, so build the
    # CompilerIR directly.
    from c4_release.neural_vm.unified_compiler.ir import CompilerIR
    ir = CompilerIR()
    ir.layer(0).attention.append(spec, name="head_3")

    op = _StubOp("attn_op", "attn", ir)
    layout = _StubLayout(
        ops_per_layer=[[op]],
        dim_positions=dim_positions,
        dim_sizes=dim_sizes,
    )

    writers = enumerate_dim_writers(layout, "OUT_LO")
    assert len(writers) == 1
    assert writers[0].op_kind == "attn"
    assert writers[0].offset == 1
    # Q/K-side gate summary surfaces the head's effective scopes.
    assert any("Q:MARK_SP" in c for c in writers[0].gate_conditions)
    assert any("K:MARK_SP" in c for c in writers[0].gate_conditions)

    # MARK_SP read shows up in both Q and K roles.
    readers = enumerate_dim_readers(layout, "MARK_SP")
    roles = {r.read_role for r in readers}
    assert "attn_Q" in roles
    assert "attn_K" in roles

    # SOURCE read shows up as attn_V.
    source_readers = enumerate_dim_readers(layout, "SOURCE")
    assert any(r.read_role == "attn_V" for r in source_readers)


def test_zeroing_writer_attention_v_mag_zero():
    """An attention head whose O write targets a slot with no V writes
    is a zeroing writer."""

    dim_positions = {"OUT_LO": 0}
    dim_sizes = {"OUT_LO": 4}

    spec = DeclarativeAttentionHeadSpec(
        head_idx=1,
        q=(),
        k=(),
        v=(),  # No V writes => V mag 0 at every slot.
        o=(AO(2, 0, 1.0),),  # O writes OUT_LO+2 from slot 0
    )

    from c4_release.neural_vm.unified_compiler.ir import CompilerIR
    ir = CompilerIR()
    ir.layer(0).attention.append(spec, name="dead_head")

    op = _StubOp("dead_attn", "attn", ir)
    layout = _StubLayout(
        ops_per_layer=[[op]],
        dim_positions=dim_positions,
        dim_sizes=dim_sizes,
    )

    zw = find_zeroing_writers(layout, "OUT_LO")
    assert len(zw) == 1
    assert zw[0].op_kind == "attn"
    assert "V mag 0" in zw[0].write_value_expr


# ---------------------------------------------------------------------------
# Full-VM layout: A3.6 regression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _full_vm_layout():
    """Compile the full VM layout once for the production-shape tests."""

    from c4_release.neural_vm.unified_compiler.decl_verifier import (
        _build_layout_only,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _build_layout_only(
            alu_mode="efficient",
            enable_conversational_io=False,
            enable_tool_calling=False,
            n_heads=8,
        )


def test_full_vm_stack0_byte_val_1_lo_surfaces_a3_chain(_full_vm_layout):
    """The A3.6 attribution: STACK0_BYTE_VAL_1_LO is written by
    ``layer10_psh_ax_broadcast_bake`` (head_8) and read by
    ``layer14_mem_generation`` (head_1 V slots). The dim-flow analyzer
    must surface both ends of that chain.
    """

    flow = trace_dim_flow(_full_vm_layout, "STACK0_BYTE_VAL_1_LO")

    # Writer: the L10 broadcast bake (head 8).
    writer_op_names = {w.op_name for w in flow.writers}
    assert "layer10_psh_ax_broadcast_bake" in writer_op_names, (
        f"Expected the L10 broadcast bake as a writer of "
        f"STACK0_BYTE_VAL_1_LO; got: {sorted(writer_op_names)}"
    )

    # Reader: L14 mem_generation.
    reader_op_names = {r.op_name for r in flow.readers}
    assert "layer14_mem_generation" in reader_op_names, (
        f"Expected layer14_mem_generation as a reader of "
        f"STACK0_BYTE_VAL_1_LO; got: {sorted(reader_op_names)}"
    )

    # Reader role must be attn_V (the head reads STACK0_BYTE_VAL into V).
    mem_gen_readers = [
        r for r in flow.readers
        if r.op_name == "layer14_mem_generation"
    ]
    roles = {r.read_role for r in mem_gen_readers}
    assert "attn_V" in roles, (
        f"Expected attn_V read role; got: {sorted(roles)}"
    )

    # Write must precede read in layer order.
    first_write = min(w.layer_idx for w in flow.writers)
    first_read = min(r.layer_idx for r in flow.readers)
    assert first_write < first_read, (
        f"Producer-consumer order violated: writes at L{first_write}, "
        f"reads at L{first_read}"
    )


def test_format_report_renders_without_crash(_full_vm_layout):
    """End-to-end smoke: the human-readable report formats cleanly for
    the production-shape STACK0_BYTE_VAL_1_LO trace."""

    flow = trace_dim_flow(_full_vm_layout, "STACK0_BYTE_VAL_1_LO")
    zw = find_zeroing_writers(_full_vm_layout, "STACK0_BYTE_VAL_1_LO")
    report = format_report(flow, zeroing_writers=zw)
    assert "STACK0_BYTE_VAL_1_LO" in report
    assert "WRITERS" in report
    assert "READERS" in report
    assert "ZEROING WRITERS" in report


def test_unknown_dim_returns_empty(_full_vm_layout):
    """Querying a dim name that isn't declared must return empty lists,
    not raise."""

    writers = enumerate_dim_writers(_full_vm_layout, "__NONEXISTENT_DIM__")
    readers = enumerate_dim_readers(_full_vm_layout, "__NONEXISTENT_DIM__")
    assert writers == []
    assert readers == []
