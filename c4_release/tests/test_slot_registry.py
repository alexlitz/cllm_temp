"""Tests for the slot-conflict registry (Phase 1 of memory cluster fix).

Three layers of test:

1. Unit tests for :class:`SlotRegistry` (claim / conflict detection /
   slot-share opt-out / range overlap semantics).
2. Integration test: compile a fresh model and assert the production op
   set produces zero unauthorized conflicts (the post-audit state).
3. "Bug class is caught" regression test: register two unwhitelisted
   bakes claiming the same ``(13, ("ffn",))`` slot and assert the
   registry raises ``SlotConflictError`` with a clear message naming
   both ops.
"""

from __future__ import annotations

import pytest

from neural_vm.unified_compiler.layer_compiler import (
    LayerCompiler,
    Operation,
)
from neural_vm.unified_compiler.slot_registry import (
    SlotConflictError,
    SlotRegistry,
    derive_slot_ids_for_op,
)


def _noop(module, dims, S):
    return None


# ---------------------------------------------------------------------------
# Unit tests for SlotRegistry
# ---------------------------------------------------------------------------


class TestSlotRegistryBasic:
    def test_empty_registry_has_no_conflicts(self):
        reg = SlotRegistry()
        assert reg.all_conflicts() == []
        # raise_on_conflict on an empty registry is a no-op.
        reg.raise_on_conflict()

    def test_single_claim_has_no_conflict(self):
        reg = SlotRegistry()
        reg.claim("op_a", 5, ("ffn",), "ffn")
        assert reg.all_conflicts() == []
        reg.raise_on_conflict()

    def test_two_ops_same_slot_conflicts(self):
        reg = SlotRegistry()
        reg.claim("op_a", 13, ("ffn",), "block")
        reg.claim("op_b", 13, ("ffn",), "block")
        conflicts = reg.all_conflicts()
        assert len(conflicts) == 1
        with pytest.raises(SlotConflictError) as exc_info:
            reg.raise_on_conflict()
        msg = str(exc_info.value)
        assert "op_a" in msg
        assert "op_b" in msg
        assert "layer=13" in msg

    def test_two_ops_different_layers_no_conflict(self):
        reg = SlotRegistry()
        reg.claim("op_a", 13, ("ffn",), "block")
        reg.claim("op_b", 14, ("ffn",), "block")
        assert reg.all_conflicts() == []

    def test_two_ops_different_heads_no_conflict(self):
        reg = SlotRegistry()
        reg.claim("op_a", 13, ("attn", "head", 0), "block")
        reg.claim("op_b", 13, ("attn", "head", 1), "block")
        assert reg.all_conflicts() == []

    def test_two_ops_same_head_at_same_layer_conflicts(self):
        # The V4 head_0 contest: this is the specific bug class the
        # registry must catch.
        reg = SlotRegistry()
        reg.claim("layer10_carry_relay_bake", 13, ("attn", "head", 0), "block")
        reg.claim("layer13_mem_addr_gather", 13, ("attn", "head", 0), "block")
        with pytest.raises(SlotConflictError) as exc_info:
            reg.raise_on_conflict()
        msg = str(exc_info.value)
        assert "layer10_carry_relay_bake" in msg
        assert "layer13_mem_addr_gather" in msg
        assert "head" in msg

    def test_same_op_reclaim_is_benign(self):
        # Defensive: an op shouldn't conflict with itself if (somehow)
        # the dispatcher records the same claim twice.
        reg = SlotRegistry()
        reg.claim("op_a", 13, ("ffn",), "block")
        reg.claim("op_a", 13, ("ffn",), "block")
        assert reg.all_conflicts() == []


class TestSlotShareOptOut:
    def test_shared_kind_suppresses_conflict(self):
        reg = SlotRegistry()
        # Both ops opt into ffn_units sharing — overlap should be allowed.
        reg.claim(
            "tail_op_1", 10, ("ffn_units", 0, 64), "ffn",
            slot_share=("ffn_units",),
        )
        reg.claim(
            "tail_op_2", 10, ("ffn_units", 32, 128), "ffn",
            slot_share=("ffn_units",),
        )
        assert reg.all_conflicts() == []

    def test_one_sided_share_still_suppresses(self):
        # The slot_share rule is symmetric: once either party declares
        # share for a kind, conflict is suppressed.
        reg = SlotRegistry()
        reg.claim(
            "tail_op_1", 10, ("ffn_units", 0, 64), "ffn",
            slot_share=("ffn_units",),
        )
        reg.claim("tail_op_2", 10, ("ffn_units", 32, 128), "ffn")
        assert reg.all_conflicts() == []

    def test_share_for_different_kind_does_not_suppress(self):
        reg = SlotRegistry()
        reg.claim(
            "op_a", 13, ("ffn",), "block",
            slot_share=("ffn_units",),  # wrong kind
        )
        reg.claim("op_b", 13, ("ffn",), "block")
        with pytest.raises(SlotConflictError):
            reg.raise_on_conflict()

    def test_invalid_slot_share_kind_raises(self):
        reg = SlotRegistry()
        with pytest.raises(ValueError, match="not a valid slot kind"):
            reg.claim("op_a", 13, ("ffn",), "block", slot_share=("fnn_units",))


class TestFFNUnitsRange:
    def test_overlapping_ranges_conflict(self):
        reg = SlotRegistry()
        reg.claim("op_a", 10, ("ffn_units", 0, 64), "ffn")
        reg.claim("op_b", 10, ("ffn_units", 32, 128), "ffn")
        with pytest.raises(SlotConflictError):
            reg.raise_on_conflict()

    def test_adjacent_disjoint_ranges_no_conflict(self):
        # end is EXCLUSIVE, so [0, 64) and [64, 128) are adjacent but disjoint.
        reg = SlotRegistry()
        reg.claim("op_a", 10, ("ffn_units", 0, 64), "ffn")
        reg.claim("op_b", 10, ("ffn_units", 64, 128), "ffn")
        assert reg.all_conflicts() == []

    def test_nested_range_conflicts(self):
        reg = SlotRegistry()
        reg.claim("outer", 10, ("ffn_units", 0, 256), "ffn")
        reg.claim("inner", 10, ("ffn_units", 64, 128), "ffn")
        with pytest.raises(SlotConflictError):
            reg.raise_on_conflict()


class TestDeriveSlotIds:
    """Verify the dispatcher-side slot-id derivation from Operation fields."""

    def _op(self, **kwargs):
        # The harness expects every Operation to carry a bake; we don't
        # actually invoke it here.
        defaults = dict(
            name="test_op",
            reads=set(),
            writes=set(),
            kind="ffn",
            bake_fn=_noop,
        )
        defaults.update(kwargs)
        return Operation(**defaults)

    def test_topology_anchor_yields_no_claim(self):
        op = self._op(
            name="some_anchor",
            kind="attn",
            bake_fn=None,
            declarative_authority="topology_anchor",
        )
        assert derive_slot_ids_for_op(op) == []

    def test_module_replacement_sentinel_ffn(self):
        op = self._op(
            name="alu_install",
            kind="block",
            layer_idx=13,
            produces={"__module_replacement": "L13.ffn[ALUShiftComposite]"},
        )
        assert ("ffn",) in derive_slot_ids_for_op(op)

    def test_module_replacement_sentinel_post_ops(self):
        op = self._op(
            name="divmod_install",
            kind="block",
            layer_idx=10,
            produces={"__module_replacement": "L10.post_ops[FlattenedDivMod]"},
        )
        assert ("post_ops_append",) in derive_slot_ids_for_op(op)

    def test_ffn_units_used_emits_range_claim(self):
        op = self._op(
            name="layer10_alu",
            kind="ffn",
            ffn_units_used=1846,
        )
        slots = derive_slot_ids_for_op(op)
        assert ("ffn_units", 0, 1846) in slots

    def test_pure_ffn_no_signal_falls_back_to_kind(self):
        op = self._op(name="bare_ffn", kind="ffn")
        assert derive_slot_ids_for_op(op) == [("ffn",)]

    def test_pure_attn_no_signal_falls_back_to_kind(self):
        op = self._op(name="bare_attn", kind="attn")
        assert derive_slot_ids_for_op(op) == [("attn",)]


# ---------------------------------------------------------------------------
# LayerCompiler integration tests (the dispatcher actually runs the
# registry scan and raises on conflict)
# ---------------------------------------------------------------------------


class TestLayerCompilerIntegration:
    def test_clean_compile_passes(self):
        c = LayerCompiler()
        c.declare_dim("MARK_PC", 1)
        c.declare_dim("OUTPUT_LO", 16)
        c.add_op(Operation(
            name="default_pc",
            reads={"MARK_PC"},
            writes={"OUTPUT_LO"},
            kind="ffn",
            bake_fn=_noop,
        ))
        # No claims, no conflicts, no error.
        layout = c.compile()
        assert layout.n_layers == 1

    def test_two_module_replacement_ops_at_same_layer_raises(self):
        c = LayerCompiler()
        c.declare_dim("X", 4)
        c.declare_dim("Y", 4)
        # Two block ops both claiming L13 block.ffn.
        c.add_op(Operation(
            name="alu_shift_install",
            reads={"X"},
            writes={"Y"},
            kind="block",
            layer_idx=13,
            bake_fn=_noop,
            produces={"__module_replacement": "L13.ffn[ALUShiftComposite]"},
        ))
        c.add_op(Operation(
            name="andorxor_install",
            reads={"X"},
            writes={"Y"},
            kind="block",
            layer_idx=13,
            bake_fn=_noop,
            produces={"__module_replacement": "L13.ffn[PureFFN/bitwise_rules]"},
        ))
        with pytest.raises(SlotConflictError) as exc_info:
            c.compile()
        msg = str(exc_info.value)
        assert "alu_shift_install" in msg
        assert "andorxor_install" in msg
        assert "layer=13" in msg
        assert "('ffn',)" in msg

    def test_slot_share_unblocks_compile(self):
        c = LayerCompiler()
        c.declare_dim("X", 4)
        c.declare_dim("Y", 4)
        # Two ops co-baking block.ffn at disjoint unit ranges — the
        # legitimate case the slot_share opt-out is for.
        c.add_op(Operation(
            name="tail_op_lo",
            reads={"X"},
            writes={"Y"},
            kind="ffn",
            layer_idx=10,
            bake_fn=_noop,
            ffn_units_used=64,
            slot_share=("ffn_units",),
        ))
        c.add_op(Operation(
            name="tail_op_hi",
            reads={"X"},
            writes={"Y"},
            kind="ffn",
            layer_idx=10,
            bake_fn=_noop,
            ffn_units_used=128,
            slot_share=("ffn_units",),
        ))
        # Both opt into ffn_units sharing; compile should pass even
        # though [0, 64) and [0, 128) overlap.
        c.compile()

    def test_invalid_slot_share_kind_rejected_at_add_op(self):
        c = LayerCompiler()
        c.declare_dim("X", 4)
        with pytest.raises(ValueError, match="not a recognized slot kind"):
            c.add_op(Operation(
                name="bad_op",
                reads=set(),
                writes={"X"},
                kind="ffn",
                bake_fn=_noop,
                slot_share=("fnn_units",),  # typo
            ))

    def test_disable_env_flag_skips_registry(self, monkeypatch):
        monkeypatch.setenv("C4_DISABLE_SLOT_REGISTRY", "1")
        c = LayerCompiler()
        c.declare_dim("X", 4)
        # Build a conflict that would normally raise.
        c.add_op(Operation(
            name="op_a",
            reads=set(),
            writes={"X"},
            kind="block",
            layer_idx=5,
            bake_fn=_noop,
            produces={"__module_replacement": "L5.ffn[A]"},
        ))
        c.add_op(Operation(
            name="op_b",
            reads=set(),
            writes={"X"},
            kind="block",
            layer_idx=5,
            bake_fn=_noop,
            produces={"__module_replacement": "L5.ffn[B]"},
        ))
        # With the env flag set, compile should not raise.
        c.compile()


# ---------------------------------------------------------------------------
# Full-VM integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFullVMRegistryClean:
    """The full production op set must produce zero unauthorized conflicts.

    This is the smoke-test analogue for the registry: if a future op
    introduces a silent contest, this test will catch it before the
    dynamic verifier even has to run.
    """

    def test_efficient_mode_no_conflicts(self):
        from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
            compile_full_vm_dynamic,
        )

        # Just calling compile is enough; the registry scan runs inside
        # ``LayerCompiler.compile`` and raises on conflict.
        compile_full_vm_dynamic(alu_mode="efficient")

    def test_lookup_mode_no_conflicts(self):
        from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
            compile_full_vm_dynamic,
        )

        compile_full_vm_dynamic(alu_mode="lookup")
