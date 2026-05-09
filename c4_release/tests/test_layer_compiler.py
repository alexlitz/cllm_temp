"""Tests for the layer-allocation compiler MVP.

These verify that LayerCompiler:
- topologically sorts ops by dim dependencies
- assigns each op to the earliest legal layer
- respects one-attn-one-ffn-per-layer
- detects cycles
- auto-computes d_model and n_layers
"""

import pytest

import torch

from neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
    ModelLayout,
    build_model_from_layout,
)


def _noop(module, dims, S):
    return None


def _op(name, reads, writes, kind):
    return Operation(
        name=name,
        reads=set(reads),
        writes=set(writes),
        kind=kind,
        bake_fn=_noop,
    )


class TestBasic:
    def test_empty_compile(self):
        compiler = LayerCompiler()
        layout = compiler.compile()
        assert layout.d_model == 0
        assert layout.n_layers == 0
        assert layout.ops_per_layer == []

    def test_single_ffn_op(self):
        c = LayerCompiler()
        c.declare_dim("MARK_PC", 1)
        c.declare_dim("OUTPUT_LO", 16)
        c.add_op(_op("default_pc", reads=["MARK_PC"], writes=["OUTPUT_LO"], kind="ffn"))
        layout = c.compile()
        assert layout.d_model == 17
        assert layout.n_layers == 1
        assert layout.dim_positions["MARK_PC"] == 0
        assert layout.dim_positions["OUTPUT_LO"] == 1
        assert layout.ops_at(0)[0].name == "default_pc"

    def test_attn_then_ffn_same_layer(self):
        # attn writes EMBED, ffn reads EMBED writes OUTPUT — both can fit at L0
        # because attn and ffn slots are independent within a layer.
        # But the dep chain forces ffn to a later layer than attn.
        # So they end up at L0 (attn) and L1 (ffn).
        c = LayerCompiler()
        c.declare_dim("MARK_PC", 1)
        c.declare_dim("EMBED", 16)
        c.declare_dim("OUTPUT", 16)
        c.add_op(_op("relay", reads=["MARK_PC"], writes=["EMBED"], kind="attn"))
        c.add_op(_op("incr", reads=["EMBED"], writes=["OUTPUT"], kind="ffn"))
        layout = c.compile()
        assert layout.n_layers == 2
        assert layout.ops_at(0)[0].name == "relay"
        assert layout.ops_at(1)[0].name == "incr"


class TestLayerSharing:
    def test_two_independent_ops_can_share_layer_with_diff_kinds(self):
        # ffn reads dim_a, writes dim_b
        # attn reads dim_c, writes dim_d
        # No deps between them. Both can land at L0 (attn slot + ffn slot).
        c = LayerCompiler()
        c.declare_dim("a", 1)
        c.declare_dim("b", 1)
        c.declare_dim("c", 1)
        c.declare_dim("d", 1)
        c.add_op(_op("op_ffn", reads=["a"], writes=["b"], kind="ffn"))
        c.add_op(_op("op_attn", reads=["c"], writes=["d"], kind="attn"))
        layout = c.compile()
        assert layout.n_layers == 1
        names_at_0 = {o.name for o in layout.ops_at(0)}
        assert names_at_0 == {"op_ffn", "op_attn"}

    def test_two_ffn_ops_must_use_separate_layers(self):
        # Two ffn ops with no dependency still need separate layers
        # (one ffn slot per layer).
        c = LayerCompiler()
        c.declare_dim("a", 1)
        c.declare_dim("b", 1)
        c.declare_dim("c", 1)
        c.declare_dim("d", 1)
        c.add_op(_op("op1", reads=["a"], writes=["b"], kind="ffn"))
        c.add_op(_op("op2", reads=["c"], writes=["d"], kind="ffn"))
        layout = c.compile()
        assert layout.n_layers == 2


class TestDeps:
    def test_long_chain(self):
        # 5 ops in a chain, each depending on the previous
        c = LayerCompiler()
        for i in range(6):
            c.declare_dim(f"d{i}", 1)
        for i in range(5):
            c.add_op(_op(
                f"op{i}",
                reads=[f"d{i}"],
                writes=[f"d{i+1}"],
                kind="ffn",
            ))
        layout = c.compile()
        # 5 sequential ffn ops -> 5 layers
        assert layout.n_layers == 5
        for i in range(5):
            assert layout.ops_at(i)[0].name == f"op{i}"

    def test_cycle_detected(self):
        # op_a writes X reads Y; op_b writes Y reads X. Cycle.
        c = LayerCompiler()
        c.declare_dim("X", 1)
        c.declare_dim("Y", 1)
        c.add_op(_op("op_a", reads=["Y"], writes=["X"], kind="ffn"))
        c.add_op(_op("op_b", reads=["X"], writes=["Y"], kind="attn"))
        with pytest.raises(ValueError, match="cycle"):
            c.compile()


class TestValidation:
    def test_undeclared_read_raises(self):
        c = LayerCompiler()
        c.declare_dim("written_dim", 1)
        with pytest.raises(ValueError, match="undeclared"):
            c.add_op(_op("op", reads=["nope"], writes=["written_dim"], kind="ffn"))

    def test_undeclared_write_raises(self):
        c = LayerCompiler()
        c.declare_dim("read_dim", 1)
        with pytest.raises(ValueError, match="undeclared"):
            c.add_op(_op("op", reads=["read_dim"], writes=["nope"], kind="ffn"))

    def test_bad_kind_raises(self):
        c = LayerCompiler()
        c.declare_dim("x", 1)
        with pytest.raises(ValueError, match="kind"):
            c.add_op(_op("op", reads=["x"], writes=["x"], kind="bogus"))

    def test_duplicate_op_raises(self):
        c = LayerCompiler()
        c.declare_dim("x", 1)
        c.add_op(_op("op", reads=[], writes=["x"], kind="ffn"))
        with pytest.raises(ValueError, match="already added"):
            c.add_op(_op("op", reads=[], writes=["x"], kind="attn"))


class TestAutoSizing:
    def test_d_model_sums_dim_sizes(self):
        c = LayerCompiler()
        c.declare_dim("a", 5)
        c.declare_dim("b", 12)
        c.declare_dim("c", 8)
        layout = c.compile()
        assert layout.d_model == 25

    def test_n_layers_zero_when_no_ops(self):
        c = LayerCompiler()
        c.declare_dim("a", 1)
        layout = c.compile()
        assert layout.n_layers == 0

    def test_dim_positions_are_contiguous_bump_pointer(self):
        c = LayerCompiler()
        c.declare_dim("a", 5)
        c.declare_dim("b", 12)
        c.declare_dim("c", 8)
        layout = c.compile()
        assert layout.dim_positions["a"] == 0
        assert layout.dim_positions["b"] == 5
        assert layout.dim_positions["c"] == 17
        assert layout.dim_range("a") == range(0, 5)
        assert layout.dim_range("b") == range(5, 17)
        assert layout.dim_range("c") == range(17, 25)


class TestRealisticExample:
    def test_pc_increment_pipeline(self):
        """Mini example resembling the PC carry-forward + increment chain.

        Layer 0 (attn): pc_carry_forward writes EMBED_PC
        Layer 1 (ffn):  pc_increment reads EMBED_PC, writes OUTPUT_PC
        Layer 1 (attn): pc_relay_to_ax reads MARK_AX, writes AX_PC

        Compiler should pack pc_increment and pc_relay_to_ax at the same layer
        (one ffn slot + one attn slot) since they don't depend on each other.
        """
        c = LayerCompiler()
        c.declare_dim("MARK_PC", 1)
        c.declare_dim("MARK_AX", 1)
        c.declare_dim("EMBED_PC", 16)
        c.declare_dim("OUTPUT_PC", 16)
        c.declare_dim("AX_PC", 16)

        c.add_op(_op("pc_carry_forward",
                     reads=["MARK_PC"], writes=["EMBED_PC"], kind="attn"))
        c.add_op(_op("pc_increment",
                     reads=["EMBED_PC"], writes=["OUTPUT_PC"], kind="ffn"))
        c.add_op(_op("pc_relay_to_ax",
                     reads=["MARK_AX"], writes=["AX_PC"], kind="attn"))

        layout = c.compile()
        # pc_carry_forward at L0 (attn).
        # pc_increment must wait for EMBED_PC -> earliest L1.
        # pc_relay_to_ax has no deps -> earliest L0, but L0's attn slot is taken,
        #                              so it goes to L1's attn slot.
        assert layout.n_layers == 2
        l0_names = {o.name for o in layout.ops_at(0)}
        l1_names = {o.name for o in layout.ops_at(1)}
        assert l0_names == {"pc_carry_forward"}
        assert l1_names == {"pc_increment", "pc_relay_to_ax"}

    def test_per_opcode_dim_ranges_no_overlap(self):
        """Demonstrate per-opcode dim allocation: each opcode has its own RESULT slot."""
        c = LayerCompiler()
        c.declare_dim("OP_ADD_FLAG", 1)
        c.declare_dim("OP_OR_FLAG", 1)
        c.declare_dim("ADD_RESULT", 16)
        c.declare_dim("OR_RESULT", 16)
        c.declare_dim("AX_OUTPUT", 16)

        c.add_op(_op("alu_add",
                     reads=["OP_ADD_FLAG"], writes=["ADD_RESULT"], kind="ffn"))
        c.add_op(_op("alu_or",
                     reads=["OP_OR_FLAG"], writes=["OR_RESULT"], kind="ffn"))
        c.add_op(_op("route_ax",
                     reads=["ADD_RESULT", "OR_RESULT", "OP_ADD_FLAG", "OP_OR_FLAG"],
                     writes=["AX_OUTPUT"], kind="ffn"))

        layout = c.compile()
        # ADD and OR can share a layer (they're independent ffn ops? no — only
        # one ffn slot per layer). They go to L0 and L1.
        # route_ax depends on both -> L2.
        assert layout.n_layers == 3
        # ADD and OR ranges don't overlap
        add_range = layout.dim_range("ADD_RESULT")
        or_range = layout.dim_range("OR_RESULT")
        assert set(add_range).isdisjoint(set(or_range))


class TestBuildModel:
    """Verify that build_model_from_layout actually constructs a working model."""

    def test_builds_model_with_auto_d_model_and_n_layers(self):
        c = LayerCompiler()
        # Declare enough dims to make a non-trivial d_model.
        c.declare_dim("MARK_PC", 1)
        c.declare_dim("EMBED", 16)
        c.declare_dim("OUTPUT", 16)

        bake_calls = []

        def attn_bake(module, dims, S):
            bake_calls.append(("attn", dims["MARK_PC"], dims["EMBED"]))

        def ffn_bake(module, dims, S):
            bake_calls.append(("ffn", dims["EMBED"], dims["OUTPUT"]))

        c.add_op(Operation(
            name="relay", reads={"MARK_PC"}, writes={"EMBED"},
            kind="attn", bake_fn=attn_bake,
        ))
        c.add_op(Operation(
            name="incr", reads={"EMBED"}, writes={"OUTPUT"},
            kind="ffn", bake_fn=ffn_bake,
        ))

        layout = c.compile()
        # d_model = 1 + 16 + 16 = 33; n_layers = 2 (relay at L0, incr at L1)
        # AutoregressiveVM rounds d_model to be divisible by n_heads. Since
        # build_model_from_layout passes layout.d_model directly, we use a
        # multiple of 8 (default num_heads) by declaring enough dims.

        # Enforce d_model divisibility for AutoregressiveAttention (multi-head)
        # by padding via an extra unused dim if needed:
        if layout.d_model % 8 != 0:
            pad = 8 - (layout.d_model % 8)
            c.declare_dim("_pad", pad)
            layout = c.compile()

        model = build_model_from_layout(layout, S=100.0)
        # Verify model uses the auto-computed d_model and n_layers
        assert model.d_model == layout.d_model
        assert len(model.blocks) == max(layout.n_layers, 1)
        # Verify each bake_fn was called with the right dim positions
        kinds = {call[0] for call in bake_calls}
        assert kinds == {"attn", "ffn"}
        # MARK_PC is at position 0 (declared first)
        attn_call = next(c for c in bake_calls if c[0] == "attn")
        assert attn_call[1] == 0  # MARK_PC starts at 0
        assert attn_call[2] == 1  # EMBED starts after MARK_PC

    def test_no_ops_yields_minimal_model(self):
        c = LayerCompiler()
        c.declare_dim("a", 8)
        c.declare_dim("b", 8)
        layout = c.compile()
        # 0 ops -> 0 layers in layout. But AutoregressiveVM needs ≥1 layer.
        # build_model_from_layout uses 1 in this case.
        model = build_model_from_layout(layout, S=100.0)
        assert model.d_model == 16
        assert len(model.blocks) == 1
