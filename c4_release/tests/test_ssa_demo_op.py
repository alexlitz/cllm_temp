"""Phase 9 SSA demo-op byte-identity test.

The demo op is ``layer8_head6_ax_carry_refresh`` (l8_ops.py:1872). Its
``reads`` were originally:

    {"OUTPUT_LO_PREV_STEP", "OUTPUT_HI_PREV_STEP", ...}

and are now spelled in SSA form:

    {"OUTPUT_LO.*.-1", "OUTPUT_HI.*.-1", ...}

This test verifies:

  1. The op factory returns a valid Operation whose reads contain the
     SSA spellings.
  2. The SSA names parse as cross-step reads of the unversioned bases.
  3. When the op is added to a LayerCompiler with OUTPUT_LO/HI already
     declared, the SSA names are auto-declared as aliases and resolve
     to the same numeric slot as their base dims.
  4. The op's bake remains a no-op at ``enable=False`` (the default), so
     model-level byte-identity is preserved.
"""

from __future__ import annotations

import pytest

from neural_vm.unified_compiler.layer_compiler import LayerCompiler
from neural_vm.unified_compiler.ops.l8_ops import (
    make_layer8_head6_ax_carry_refresh_op,
)
from neural_vm.unified_compiler.ssa_dim import is_ssa_form, parse_ssa_name


# Dim names this op reads (besides the SSA-form OUTPUT_*) — we declare
# them as 1-wide stubs so the LayerCompiler accepts the op.
_SCALAR_READS = {
    "MARK_AX", "HAS_SE", "CONST",
    "OP_IMM", "OP_EXIT", "OP_NOP", "OP_JMP", "OP_JSR", "OP_LEV",
    "OP_BZ", "OP_BNZ", "OP_PSH", "OP_ADJ", "OP_ENT",
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
    "OP_AND", "OP_OR", "OP_XOR",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_SHL", "OP_SHR",
    "OP_LI", "OP_LC", "OP_LEA",
}


def test_demo_op_reads_contain_ssa_names():
    op = make_layer8_head6_ax_carry_refresh_op(enable=False)
    assert "OUTPUT_LO.*.-1" in op.reads
    assert "OUTPUT_HI.*.-1" in op.reads
    # The legacy PREV_STEP names are gone.
    assert "OUTPUT_LO_PREV_STEP" not in op.reads
    assert "OUTPUT_HI_PREV_STEP" not in op.reads


def test_demo_op_ssa_names_parse_as_cross_step_wildcard():
    op = make_layer8_head6_ax_carry_refresh_op(enable=False)
    for name in ("OUTPUT_LO.*.-1", "OUTPUT_HI.*.-1"):
        assert is_ssa_form(name)
        parsed = parse_ssa_name(name)
        assert parsed.base_dim in {"OUTPUT_LO", "OUTPUT_HI"}
        assert parsed.is_any_writer
        assert parsed.is_cross_step
        assert parsed.step_offset == -1


def test_demo_op_compiles_with_ssa_alias_byte_identical():
    """SSA reads resolve to the same numeric slot as the unversioned base."""
    op = make_layer8_head6_ax_carry_refresh_op(enable=False)

    c = LayerCompiler()
    # Pin OUTPUT_LO/HI at their production slots so the alias check is
    # against real numbers.
    c.declare_dim("OUTPUT_LO", 16, pinned=174)
    c.declare_dim("OUTPUT_HI", 16, pinned=190)
    c.declare_dim("AX_CARRY_LO", 16, pinned=328)
    c.declare_dim("AX_CARRY_HI", 16, pinned=344)
    for scalar in _SCALAR_READS:
        c.declare_dim(scalar, 1)

    c.add_op(op)

    # Both SSA forms were auto-aliased onto the base.
    aliases = getattr(c, "_aliases", {})
    assert aliases["OUTPUT_LO.*.-1"] == "OUTPUT_LO"
    assert aliases["OUTPUT_HI.*.-1"] == "OUTPUT_HI"

    positions = c._allocate_dims()
    # Byte-identical numeric resolution: SSA name == base name slot.
    assert positions["OUTPUT_LO.*.-1"] == positions["OUTPUT_LO"] == 174
    assert positions["OUTPUT_HI.*.-1"] == positions["OUTPUT_HI"] == 190


def test_demo_op_bake_remains_noop_at_default_enable_false():
    """At enable=False (the production default), the bake writes nothing."""
    op = make_layer8_head6_ax_carry_refresh_op(enable=False)

    class _StubAttn:
        # Minimal stub: any attribute access during bake should be skipped.
        def __getattr__(self, item):  # pragma: no cover - should not fire
            raise AssertionError(
                f"bake should be no-op at enable=False; touched {item!r}"
            )

    class _StubBlock:
        attn = _StubAttn()

    # Should not raise.
    op.bake_fn(_StubBlock(), {}, 100.0)
