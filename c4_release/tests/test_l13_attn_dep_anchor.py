"""Regression test: L13 attn dep anchor must pin the ALU shift install op.

History — smoke_bisect_20260602_1339 ``layer_idx=13`` drop regression:
    Commit ae64239b switched ``l13_alu_shift_install`` from
    ``layer_idx=13`` to ``target_op_name="l13_alu_shift_getobd"``. The
    target stage is ``kind="ffn"`` and registered ONLY in
    ``alu_mode="efficient"``; even when registered, the dep graph did
    not anchor it at L13. Block fusion then collapsed L12.ffn/L13.ffn
    into a single dead block at L13, breaking 29 AX-write smoke tests
    (TestSmokeBasic::test_imm_exit and friends).

What this test locks in:
    1. ``_layer13_attn_dep_anchor`` is registered in both ``lookup``
       and ``efficient`` alu modes (sanity — it's unconditional).
    2. ``_layer13_attn_dep_anchor`` lands at layer 13 in both modes.
    3. ``l13_alu_shift_install`` lands at layer 13 in ``efficient``
       mode (i.e. the same layer as the attn anchor it now targets).
    4. ``layer13_mem_addr_gather`` (the other L13 attn op) lands at
       layer 13 in both modes.
    5. The compiled model's total baked-block count is identical
       between ``lookup`` and ``efficient`` modes (no L13/L12 fusion).
"""

from __future__ import annotations

import pytest

from c4_release.neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
from c4_release.neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic


_L13_ATTN_ANCHOR = "_layer13_attn_dep_anchor"
_L13_ALU_SHIFT_INSTALL = "l13_alu_shift_install"
_L13_MEM_ADDR_GATHER = "layer13_mem_addr_gather"


def _build_layout(alu_mode: str):
    """Compile a tiny VM and return ``(model, layout)``."""
    return compile_full_vm_dynamic(alu_mode=alu_mode)


def _find_layer_for_op(layout, op_name: str):
    """Return the layer index of ``op_name`` in ``layout.ops_per_layer``.

    Returns ``None`` when the op is not present at any layer.
    """
    for layer_idx, ops_in_layer in enumerate(layout.ops_per_layer):
        for op in ops_in_layer:
            if op.name == op_name:
                return layer_idx
    return None


def _block_op_layer(layout, op_name: str):
    """Return the resolved layer for a block op named ``op_name``.

    Block ops do not live in ``ops_per_layer`` (which holds attn/ffn
    placements); their target is resolved via
    ``layout.resolve_block_op_layer``.
    """
    for op in getattr(layout, "block_ops", ()):
        if op.name == op_name:
            return layout.resolve_block_op_layer(op)
    return None


@pytest.mark.parametrize("alu_mode", ["lookup", "efficient"])
def test_layer13_attn_dep_anchor_registered(alu_mode):
    """``_layer13_attn_dep_anchor`` must be unconditional across alu modes."""
    ops = all_core_ops(alu_mode=alu_mode)
    names = {op.name for op in ops}
    assert _L13_ATTN_ANCHOR in names, (
        f"{_L13_ATTN_ANCHOR} must be registered in alu_mode={alu_mode!r}; "
        "it is the L13 anchor that ``l13_alu_shift_install`` and other L13 "
        "block ops bind to via ``target_op_name``."
    )


@pytest.mark.parametrize("alu_mode", ["lookup", "efficient"])
def test_layer13_attn_dep_anchor_lands_at_l13(alu_mode):
    """``_layer13_attn_dep_anchor`` must be assigned to layer 13.

    The anchor is what pins L13 block ops (e.g. ``l13_alu_shift_install``,
    ``layer13_mem_addr_gather``) at the correct layer.
    """
    _model, layout = _build_layout(alu_mode)
    placement = _find_layer_for_op(layout, _L13_ATTN_ANCHOR)
    assert placement == 13, (
        f"{_L13_ATTN_ANCHOR} placed at layer {placement} (expected 13) in "
        f"alu_mode={alu_mode!r}. L13 block ops resolve via this anchor; if "
        "it drifts, L12/L13 fuse and AX-write smoke tests break."
    )


def test_l13_alu_shift_install_lands_at_l13_efficient_mode():
    """In efficient mode, ``l13_alu_shift_install`` must resolve to L13.

    smoke_bisect_20260602_1339 regression: with ``target_op_name=
    l13_alu_shift_getobd`` the install op did not pin L13, block
    fusion ate L13.ffn / L12.ffn into a single dead block, and 29
    smoke tests failed at the AX read. ``target_op_name=
    _layer13_attn_dep_anchor`` (committed in this branch) restores
    the L13 pin.
    """
    _model, layout = _build_layout("efficient")
    layer = _block_op_layer(layout, _L13_ALU_SHIFT_INSTALL)
    assert layer == 13, (
        f"{_L13_ALU_SHIFT_INSTALL} resolved to layer {layer} (expected 13) "
        "in efficient mode. If this drifts, the SHL/SHR composite is "
        "installed at the wrong block and 29 AX-write smoke tests fail."
    )


@pytest.mark.parametrize("alu_mode", ["lookup", "efficient"])
def test_layer13_mem_addr_gather_lands_at_l13(alu_mode):
    """``layer13_mem_addr_gather`` is the L13 attn block op; must pin L13.

    It already binds to ``_layer13_attn_dep_anchor`` via ``target_op_name``;
    this assertion guards against any future regression that would shift
    the anchor away from layer 13.
    """
    _model, layout = _build_layout(alu_mode)
    layer = _block_op_layer(layout, _L13_MEM_ADDR_GATHER)
    assert layer == 13, (
        f"{_L13_MEM_ADDR_GATHER} resolved to layer {layer} (expected 13) in "
        f"alu_mode={alu_mode!r}."
    )


def test_block_count_stable_across_alu_modes():
    """Total baked-block count must be alu-mode-invariant.

    Block fusion (the smoke_bisect_20260602_1339 failure mode) shows up
    as a 1-block count drop. Lookup and efficient mode share the same
    L0..L21 layer layout up to (efficient) wrapper expansion; the block
    count after expansion must match exactly so a future bisect agent
    can detect collapse with a single assertion.
    """
    lookup_model, _ = _build_layout("lookup")
    efficient_model, _ = _build_layout("efficient")
    lookup_blocks = len(lookup_model.blocks)
    efficient_blocks = len(efficient_model.blocks)
    assert lookup_blocks == efficient_blocks, (
        f"Block count drift between alu modes: "
        f"lookup={lookup_blocks}, efficient={efficient_blocks}. "
        "A drop of 1 typically indicates L12.ffn / L13.ffn fusion (see "
        "smoke_bisect_20260602_1339)."
    )
