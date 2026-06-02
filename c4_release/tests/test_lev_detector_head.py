"""Regression suite for the L8 LEV detector attention head.

The LEV detector head -- ``make_lev_detector_head_op`` in
``neural_vm/unified_compiler/ops/control_flow_heads.py`` -- is the V2/G7
"form 2" architectural alternative to the ``*_PREV_STEP`` dim-alias trick
for the 7-op LEV sealed loop. See ``docs/CONTROL_FLOW_DETECTOR_HEADS.md``
for the design.

This test suite pins:

1. The factory returns an :class:`Operation` with the expected metadata
   (name, phase, reads, writes, produces).
2. The four ``*_VIA_LEV_DETECTOR_*`` residual dims are declared
   unconditionally (independent of the ``enable`` flag) so the dim
   registry stays stable across enable/disable.
3. The default ``enable=False`` bake is a no-op (byte-identity at the
   weight level vs the pre-spike baseline).
4. The ``enable=True`` bake lowers without raising when the per-block
   ``num_heads`` accommodates the extra head, and writes non-zero values
   into the expected attention weight slots.
5. The op is registered in ``all_core_ops()`` so the production compile
   path picks it up.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 1. Factory metadata
# ---------------------------------------------------------------------------


def _factory():
    from neural_vm.unified_compiler.ops.control_flow_heads import (
        make_lev_detector_head_op,
    )
    return make_lev_detector_head_op


def test_factory_returns_operation_with_expected_name():
    op = _factory()(enable=False)
    assert op.name == "lev_detector_head"


def test_op_writes_four_detector_dims():
    op = _factory()(enable=False)
    assert "PC_VIA_LEV_DETECTOR_LO" in op.writes
    assert "PC_VIA_LEV_DETECTOR_HI" in op.writes
    assert "BP_VIA_LEV_DETECTOR" in op.writes
    assert "SP_VIA_LEV_DETECTOR" in op.writes


def test_op_reads_lev_marker_and_prev_step_dims():
    op = _factory()(enable=False)
    # K side fires on OP_LEV (prev-step opcode marker).
    assert "OP_LEV" in op.reads
    # Q side fires at current step's PC marker on non-first steps.
    assert "MARK_PC" in op.reads
    assert "HAS_SE" in op.reads
    # V side projects saved PC / BP / SP nibbles through KV cache via
    # cross-step PREV_STEP aliases.
    assert "TEMP_PREV_STEP" in op.reads
    assert "ADDR_B0_LO_PREV_STEP" in op.reads
    assert "ADDR_B0_HI_PREV_STEP" in op.reads


def test_op_produces_detector_dims_for_staleness_analyzer():
    op = _factory()(enable=False)
    assert op.produces is not None
    assert op.produces.get("PC_VIA_LEV_DETECTOR_LO") == "PC_byte0"
    assert op.produces.get("PC_VIA_LEV_DETECTOR_HI") == "PC_byte0"
    assert op.produces.get("BP_VIA_LEV_DETECTOR") == "BP_byte0"
    assert op.produces.get("SP_VIA_LEV_DETECTOR") == "SP_byte0"


def test_op_phase_between_ax_carry_refresh_and_op_imm_relay():
    op = _factory()(enable=False)
    # Detector must fire AFTER layer8_head6_ax_carry_refresh (phase=8.05)
    # so any future L9 alu consumer reading PC_VIA_LEV_DETECTOR_LO sees
    # an in-step producer, and BEFORE the L8 multibyte / OP_IMM relay
    # heads (phase>=8.1).
    assert 8.05 < op.phase < 8.5


def test_op_is_block_kind():
    op = _factory()(enable=False)
    # The detector lowers via the block-op pipeline so it can claim a
    # fresh head slot via AttentionHeadAllocator at bake time.
    assert op.kind == "block"


def test_disabled_bake_body_is_noop():
    """With ``enable=False`` the bake function must not touch the block.

    This is the byte-identity contract: the registered op has a populated
    ``produces`` map (so the staleness analyser sees a producer) but the
    actual weight bake is a no-op until the production wiring is flipped.
    """
    op = _factory()(enable=False)
    # Build a minimal fake block and verify the bake leaves it untouched.
    block = _make_fake_l8_block()
    snapshot_q = block.attn.W_q.data.clone()
    snapshot_k = block.attn.W_k.data.clone()
    snapshot_v = block.attn.W_v.data.clone()
    snapshot_o = block.attn.W_o.data.clone()
    dim_positions = _dummy_dim_positions()
    op.bake_fn(block, dim_positions, 50.0)
    assert torch.equal(block.attn.W_q.data, snapshot_q)
    assert torch.equal(block.attn.W_k.data, snapshot_k)
    assert torch.equal(block.attn.W_v.data, snapshot_v)
    assert torch.equal(block.attn.W_o.data, snapshot_o)


# ---------------------------------------------------------------------------
# 2. Residual dim declaration is unconditional
# ---------------------------------------------------------------------------


def test_detector_dims_declared_in_shared():
    """The four *_VIA_LEV_DETECTOR_* dims must be unconditionally declared.

    The shared post-allocator hook in ``ops/shared.py`` calls
    ``compiler.declare_dim`` for each detector dim irrespective of the
    head's ``enable`` flag, so the residual layout stays stable when the
    head is toggled.
    """
    from neural_vm.unified_compiler.ops import shared
    # Read the source file and assert the four detector dims appear in
    # the declare_dim loop. (Direct source-grep is robust against any
    # future reorganisation of the declare_dim helper.)
    src = open(shared.__file__).read()
    for name in (
        "PC_VIA_LEV_DETECTOR_LO",
        "PC_VIA_LEV_DETECTOR_HI",
        "BP_VIA_LEV_DETECTOR",
        "SP_VIA_LEV_DETECTOR",
    ):
        assert name in src, f"detector dim {name} missing from shared.py"


# ---------------------------------------------------------------------------
# 3. Enabled bake fires and writes into the expected slots
# ---------------------------------------------------------------------------


def test_enabled_bake_writes_attention_weights():
    """Flipping ``enable=True`` must produce non-zero weights in W_q/W_k/W_v/W_o.

    The fake block has ``num_heads=9`` (one above the legacy 8) so the
    dynamic-first-fit allocator places the detector at head_idx=8 and the
    bake has room to write its Q/K/V/O slots.
    """
    op = _factory()(enable=True)
    dim_positions = _dummy_dim_positions()
    # d_model must cover the highest dim position referenced (TEMP+15,
    # ADDR_B0_HI+15, etc.). Each name reserves 32 slots in our cursor
    # walk; the dim list has ~40 entries -> d_model >= 40 * 32 = 1280.
    max_pos = max(dim_positions.values()) + 32
    block = _make_fake_l8_block(num_heads=9, d_model=max(1440, max_pos))
    op.bake_fn(block, dim_positions, 50.0)
    # Detector lands at head_idx=8 (the lowest free slot after 0..7 are
    # claimed by the legacy L8 layout).
    HD = block.attn.W_q.shape[0] // block.attn.num_heads
    base = 8 * HD
    # The Q slot 0 has 3 writes (MARK_PC, HAS_SE, CONST), so the row is
    # not all-zero.
    assert torch.any(block.attn.W_q.data[base, :] != 0)
    # K slot 0 has 2 writes (OP_LEV, MARK_PC).
    assert torch.any(block.attn.W_k.data[base, :] != 0)
    # V slots 1..64 are all populated (16 each from TEMP_LO/HI/B0_LO/B0_HI).
    for k in range(16):
        assert torch.any(block.attn.W_v.data[base + 1 + k, :] != 0)
        assert torch.any(block.attn.W_v.data[base + 17 + k, :] != 0)
        assert torch.any(block.attn.W_v.data[base + 33 + k, :] != 0)
        assert torch.any(block.attn.W_v.data[base + 49 + k, :] != 0)


def test_enabled_bake_rejects_insufficient_head_dim():
    """``head_dim < 65`` must raise so the spike forces the dynamic path."""
    op = _factory()(enable=True)
    # num_heads=8 with d_model=512 -> head_dim=64, which is one slot
    # short of the detector's 1 Q + 64 V slots requirement.
    block = _make_fake_l8_block(num_heads=8, d_model=512)
    dim_positions = _dummy_dim_positions()
    with pytest.raises(ValueError, match="head_dim>=65"):
        op.bake_fn(block, dim_positions, 50.0)


# ---------------------------------------------------------------------------
# 4. Registration in all_core_ops
# ---------------------------------------------------------------------------


def test_op_registered_in_all_core_ops():
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
    ops = all_core_ops()
    names = [o.name for o in ops]
    assert "lev_detector_head" in names, (
        "lev_detector_head not registered in all_core_ops; the detector "
        "produces annotation will not participate in the staleness scan."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeParam:
    """torch.nn.Parameter-like stub: exposes ``.data`` and ``.shape``."""

    def __init__(self, t: torch.Tensor):
        self.data = t

    @property
    def shape(self):
        return self.data.shape


def _make_fake_l8_block(num_heads: int = 9, d_model: int | None = None):
    """Build a minimal block stub with the attention weights the bake touches.

    The default ``num_heads=9`` matches the post-Phase-8.O.2 dynamic head
    count path (8 legacy + 1 detector). When ``d_model`` is None we pick
    ``num_heads * 80`` so head_dim=80 (>=65 needed by the detector). The
    head_dim=64 / d_model=512 boundary is exercised by the dedicated
    ``test_enabled_bake_rejects_insufficient_head_dim`` test.
    """
    if d_model is None:
        head_dim = 80
        d_model = num_heads * head_dim
    else:
        head_dim = d_model // num_heads
    block = type("Block", (), {})()
    block.attn = type("Attn", (), {})()
    block.attn.num_heads = num_heads
    block.attn.W_q = _FakeParam(torch.zeros(num_heads * head_dim, d_model))
    block.attn.W_k = _FakeParam(torch.zeros(num_heads * head_dim, d_model))
    block.attn.W_v = _FakeParam(torch.zeros(num_heads * head_dim, d_model))
    block.attn.W_o = _FakeParam(torch.zeros(d_model, num_heads * head_dim))
    return block


def _dummy_dim_positions() -> dict:
    """Return a dict that maps every dim name the detector reads/writes to a
    distinct position, used by ``_as_setdim_proxy`` inside the bake.
    """
    names = [
        # Reads
        "MARK_PC", "HAS_SE", "CONST",
        "TEMP", "TEMP_PREV_STEP",
        "ADDR_B0_LO", "ADDR_B0_LO_PREV_STEP",
        "ADDR_B0_HI", "ADDR_B0_HI_PREV_STEP",
        "OP_LEV", "OP_IMM", "OP_EXIT", "OP_NOP", "OP_JMP", "OP_JSR",
        "OP_BZ", "OP_BNZ", "OP_PSH", "OP_ADJ", "OP_ENT",
        "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
        "OP_AND", "OP_OR", "OP_XOR",
        "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
        "OP_SHL", "OP_SHR",
        "OP_LI", "OP_LC", "OP_LEA",
        # Writes
        "PC_VIA_LEV_DETECTOR_LO",
        "PC_VIA_LEV_DETECTOR_HI",
        "BP_VIA_LEV_DETECTOR",
        "SP_VIA_LEV_DETECTOR",
    ]
    # Each named base reserves 16 slots so TEMP+k / ADDR_B0_LO+k indexing
    # below the multi-wide bands stays in-range.
    out: dict[str, int] = {}
    cursor = 0
    for n in names:
        out[n] = cursor
        cursor += 32  # generous gap for the +k indexing
    return out
