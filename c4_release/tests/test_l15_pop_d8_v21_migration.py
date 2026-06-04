"""Byte-identity gate for the V2.1 L15 pop_d8 head 9 migration.

Sample migration for ``docs/RUNTIME_ATTN_GAPS_2026_06_04.md`` Wave W8:
``_suppress_l15_lookup_pop_d8_head_9`` was rewritten on top of the V2.1
attention primitives (``binary_address_lookup_attention`` +
``attention_head_extension``). This test pins the migration to the
legacy bake's exact ``W_q``/``W_k``/``W_v``/``W_o``/``alibi_slopes``
output so a future regression on either the primitive or the migration
fires loud, not silent.
"""
from __future__ import annotations

import torch

from c4_release.neural_vm.base_layers import PureAttention
from c4_release.neural_vm.vm_step import _SetDim as BD
from c4_release.neural_vm.unified_compiler.ops.l15_ops import (
    _suppress_l15_lookup_pop_d8_head_9,
)


def _legacy_pop_d8_writer(attn, BD, HD) -> None:
    """Verbatim copy of the pre-V2.1 imperative body.

    Preserved here as the byte-identity reference for the migration.
    The production helper (``_suppress_l15_lookup_pop_d8_head_9``) now
    builds the same writes via V2.1 spec composition.
    """
    head = 9
    base = head * HD
    attn.W_q.data[base:base + HD, :] = 0.0
    attn.W_k.data[base:base + HD, :] = 0.0
    attn.W_v.data[base:base + HD, :] = 0.0
    attn.W_o.data[:, base:base + HD] = 0.0

    attn.W_q.data[base + 0, BD.CONST] = 1.0
    attn.W_k.data[base + 0, BD.CONST] = -1000.0

    pop_d8_to_e0_row = min(HD - 1, 63)
    pop_d8_to_e0_s = 50000.0
    row = base + pop_d8_to_e0_row
    attn.W_q.data[row, BD.CONST] = -4.0 * pop_d8_to_e0_s
    attn.W_q.data[row, BD.MARK_STACK0] = pop_d8_to_e0_s
    attn.W_q.data[row, BD.HAS_SE] = pop_d8_to_e0_s
    attn.W_q.data[row, BD.CMP + 3] = pop_d8_to_e0_s
    attn.W_q.data[row, BD.ADDR_B0_LO + 8] = pop_d8_to_e0_s
    attn.W_q.data[row, BD.ADDR_B0_HI + 13] = pop_d8_to_e0_s
    attn.W_q.data[row, BD.IS_BYTE] = -5.0 * pop_d8_to_e0_s
    attn.W_q.data[row, BD.MEM_STORE] = -8.0 * pop_d8_to_e0_s
    for marker_dim in (
        BD.MARK_AX, BD.MARK_PC, BD.MARK_SP, BD.MARK_BP, BD.MARK_MEM,
    ):
        attn.W_q.data[row, marker_dim] = -5.0 * pop_d8_to_e0_s
    attn.W_k.data[row, BD.MEM_VAL_B1] = 1.0
    attn.W_k.data[row, BD.ADDR_B0_LO + 0] = 1.0
    attn.W_k.data[row, BD.ADDR_B0_HI + 14] = 1.0
    for idx in range(16):
        attn.W_v.data[base + 1 + idx, BD.CLEAN_EMBED_LO + idx] = 1.0
        attn.W_v.data[base + 17 + idx, BD.CLEAN_EMBED_HI + idx] = 1.0
        attn.W_o.data[BD.OUTPUT_LO + idx, base + 1 + idx] = 40.0
        attn.W_o.data[BD.OUTPUT_HI + idx, base + 17 + idx] = 40.0
    if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
        attn.alibi_slopes[head] = 1.0


def _make_attention(num_heads: int, head_dim: int):
    """Construct a clean ``PureAttention`` sized for ``num_heads`` heads
    of width ``head_dim`` plus an ``alibi_slopes`` buffer so the
    pop_d8 writer's slope poke is observable.
    """
    dim = num_heads * head_dim
    attn = PureAttention(dim=dim, num_heads=num_heads, causal=False)
    attn.alibi_slopes = torch.zeros(num_heads, dtype=torch.float32)
    return attn


def test_pop_d8_head_9_migration_byte_identical_to_legacy():
    HD = 64
    num_heads = 12  # widest live config — covers the LEV head 9 case.
    attn_v21 = _make_attention(num_heads=num_heads, head_dim=HD)
    attn_legacy = _make_attention(num_heads=num_heads, head_dim=HD)

    _suppress_l15_lookup_pop_d8_head_9(attn_v21, BD, HD)
    _legacy_pop_d8_writer(attn_legacy, BD, HD)

    assert torch.equal(attn_v21.W_q.data, attn_legacy.W_q.data), (
        "pop_d8 W_q drift between V2.1 spec writer and legacy writer"
    )
    assert torch.equal(attn_v21.W_k.data, attn_legacy.W_k.data), (
        "pop_d8 W_k drift between V2.1 spec writer and legacy writer"
    )
    assert torch.equal(attn_v21.W_v.data, attn_legacy.W_v.data), (
        "pop_d8 W_v drift between V2.1 spec writer and legacy writer"
    )
    assert torch.equal(attn_v21.W_o.data, attn_legacy.W_o.data), (
        "pop_d8 W_o drift between V2.1 spec writer and legacy writer"
    )
    assert torch.equal(attn_v21.alibi_slopes, attn_legacy.alibi_slopes), (
        "pop_d8 alibi_slopes drift between V2.1 spec writer and legacy"
    )


def test_pop_d8_head_9_migration_wipes_prior_head_writes():
    """The wipe-then-write semantics must survive the migration:
    a prior writer that touches head 9 (e.g. LEV's
    ``_set_layer15_memory_lookup_lev_heads_4_11`` at num_heads >= 12)
    must have its head-9 writes cleared before the V2.1 spec writes
    land.
    """
    HD = 64
    num_heads = 12
    attn = _make_attention(num_heads=num_heads, head_dim=HD)

    # Plant arbitrary "junk" writes on head 9 to simulate a prior
    # writer's contributions.
    base = 9 * HD
    attn.W_q.data[base + 5, 100] = 999.0
    attn.W_k.data[base + 5, 100] = 999.0
    attn.W_v.data[base + 5, 100] = 999.0
    attn.W_o.data[100, base + 5] = 999.0

    _suppress_l15_lookup_pop_d8_head_9(attn, BD, HD)

    # The junk writes at slot 5 should be zero after the wipe.
    assert attn.W_q.data[base + 5, 100].item() == 0.0
    assert attn.W_k.data[base + 5, 100].item() == 0.0
    assert attn.W_v.data[base + 5, 100].item() == 0.0
    assert attn.W_o.data[100, base + 5].item() == 0.0


def test_pop_d8_head_9_migration_byte_identical_at_num_heads_10():
    """The pop_d8 fragment fires at ``num_heads > 9`` — verify
    byte-identity at the narrowest live config (``num_heads = 10``)
    too, where head 9 has no LEV writer touching it before the wipe.
    """
    HD = 64
    num_heads = 10
    attn_v21 = _make_attention(num_heads=num_heads, head_dim=HD)
    attn_legacy = _make_attention(num_heads=num_heads, head_dim=HD)

    _suppress_l15_lookup_pop_d8_head_9(attn_v21, BD, HD)
    _legacy_pop_d8_writer(attn_legacy, BD, HD)

    assert torch.equal(attn_v21.W_q.data, attn_legacy.W_q.data)
    assert torch.equal(attn_v21.W_k.data, attn_legacy.W_k.data)
    assert torch.equal(attn_v21.W_v.data, attn_legacy.W_v.data)
    assert torch.equal(attn_v21.W_o.data, attn_legacy.W_o.data)
    assert torch.equal(attn_v21.alibi_slopes, attn_legacy.alibi_slopes)
