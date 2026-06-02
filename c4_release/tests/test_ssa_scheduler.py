"""Phase 9 SSA scheduler-hook tests.

Verifies that SSA-form cross-step reads:
  * are auto-declared as aliases of their base dim by ``add_op``,
  * resolve to the same numeric slot as the base dim at allocation,
  * are skipped by the topo sorter so no cyclic back-edge is created.
"""

import pytest

from neural_vm.unified_compiler.layer_compiler import LayerCompiler, Operation


def _noop_bake(target, dim_positions, S):
    return None


def test_ssa_read_auto_declares_alias():
    c = LayerCompiler()
    c.declare_dim("OUTPUT_LO", 16, pinned=174)
    c.declare_dim("MARK_AX", 1, pinned=0)
    c.add_op(
        Operation(
            name="reader",
            kind="ffn",
            reads={"MARK_AX", "OUTPUT_LO.writer_x.-1"},
            writes=set(),
            bake_fn=_noop_bake,
        )
    )
    # SSA alias was auto-declared as a 16-wide alias of OUTPUT_LO.
    assert "OUTPUT_LO.writer_x.-1" in c.dims
    assert c.dims["OUTPUT_LO.writer_x.-1"] == 16
    aliases = getattr(c, "_aliases", {})
    assert aliases["OUTPUT_LO.writer_x.-1"] == "OUTPUT_LO"


def test_ssa_alias_resolves_to_base_slot():
    c = LayerCompiler()
    c.declare_dim("OUTPUT_LO", 16, pinned=174)
    c.declare_dim("MARK_AX", 1, pinned=0)
    c.add_op(
        Operation(
            name="reader",
            kind="ffn",
            reads={"MARK_AX", "OUTPUT_LO.writer_x.-1"},
            writes=set(),
            bake_fn=_noop_bake,
        )
    )
    positions = c._allocate_dims()
    assert positions["OUTPUT_LO"] == 174
    assert positions["OUTPUT_LO.writer_x.-1"] == 174  # aliased


def test_ssa_cross_step_breaks_cycle():
    """Two ops form a same-step cycle. SSA tag on one read breaks it."""
    c = LayerCompiler()
    c.declare_dim("A", 1, pinned=0)
    c.declare_dim("B", 1, pinned=1)
    # producer_a: reads B (cross-step!), writes A.
    # producer_b: reads A (same-step), writes B.
    # Without SSA: cycle (a depends on b depends on a).
    # With SSA tagging b's read of A as cross-step: no cycle.
    c.add_op(
        Operation(
            name="producer_a",
            kind="attn",
            reads={"B.producer_b.-1"},  # SSA cross-step read
            writes={"A"},
            bake_fn=_noop_bake,
        )
    )
    c.add_op(
        Operation(
            name="producer_b",
            kind="ffn",
            reads={"A"},
            writes={"B"},
            bake_fn=_noop_bake,
        )
    )
    # Should NOT raise a cycle error.
    topo = c._topological_sort()
    names = [op.name for op in topo]
    assert names == ["producer_b", "producer_a"] or names == ["producer_a", "producer_b"]
    # producer_b reads A and producer_a writes A, so producer_a must
    # come before producer_b in the topological order.
    assert names.index("producer_a") < names.index("producer_b")


def test_ssa_unversioned_read_still_creates_edge():
    """Sanity: removing the SSA suffix re-creates the cycle."""
    c = LayerCompiler()
    c.declare_dim("A", 1, pinned=0)
    c.declare_dim("B", 1, pinned=1)
    c.add_op(
        Operation(
            name="producer_a",
            kind="attn",
            reads={"B"},  # NOT SSA — same-step read, creates cycle
            writes={"A"},
            bake_fn=_noop_bake,
        )
    )
    c.add_op(
        Operation(
            name="producer_b",
            kind="ffn",
            reads={"A"},
            writes={"B"},
            bake_fn=_noop_bake,
        )
    )
    with pytest.raises(ValueError, match="cycle"):
        c._topological_sort()


def test_ssa_undeclared_base_raises():
    c = LayerCompiler()
    c.declare_dim("X", 1, pinned=0)
    with pytest.raises(ValueError, match="undeclared base dim"):
        c.add_op(
            Operation(
                name="op",
                kind="ffn",
                reads={"NOT_A_DIM.writer.-1"},
                writes=set(),
                bake_fn=_noop_bake,
            )
        )


def test_ssa_wildcard_writer_accepted():
    c = LayerCompiler()
    c.declare_dim("OUTPUT_LO", 16, pinned=174)
    c.add_op(
        Operation(
            name="reader",
            kind="ffn",
            reads={"OUTPUT_LO.*.-1"},
            writes=set(),
            bake_fn=_noop_bake,
        )
    )
    assert "OUTPUT_LO.*.-1" in c.dims
