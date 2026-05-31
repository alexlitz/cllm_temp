"""Per-op audit harness for L13 MEM address gather (heads 0-2).

The L13 attention layer carries 3 dedicated heads (0, 1, 2) that gather
MEM address bytes from upstream MEM-addr-byte positions (populated by
L9-L12 from STACK0 or AX_CARRY depending on opcode) into the MEM-val-byte
positions where downstream layers consume them for SI/SC/LI/LC. This
module mirrors ``test_l7_operand_gather.py`` -- it gives the previously
untested op a focused regression net combining:

  1. Op registration  -- ``layer13_mem_addr_gather`` lands at L13 with
     the expected reads/writes/claim shape.
  2. Static weight structure -- the per-head Q/K/V/O matrix slots match
     ``_set_layer13_mem_addr_gather``'s documented routing for each
     address-byte index j in {0, 1, 2}.
  3. Per-opcode behaviour -- ``LI``, ``LC``, ``SI``, ``SC`` rows all
     receive the same MEM-addr gather (the opcode-gating is upstream).
  4. Symbolic forward -- driving a synthetic residual stream where the
     MEM addr byte K positions carry CLEAN_EMBED nibbles for arbitrary
     STACK0 / AX_CARRY states, the L13 head writes the expected
     ``ADDR_B{j}_LO/HI`` one-hots at the MEM val byte positions.
  5. Declarative claim verifier -- ``verify_claims_static`` reports no
     declaration drift for this op.
  6. End-to-end smoke -- a tiny ``int x; x = 42;`` style program through
     the pure-neural runner, gated as ``xfail`` because Phase-7 SI/LI
     roundtrip is known broken upstream (see
     ``TestSmokeMemory::test_si_li_roundtrip``).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared imports / constants
# ---------------------------------------------------------------------------

from neural_vm.base_layers import PureAttention  # noqa: E402
from neural_vm.setup_helpers import _set_layer13_mem_addr_gather  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402


# Marker indices used by L13 attention -- mirrored from the bake helper.
_MEM_I = 4
# Address-byte index j (0, 1, 2) → output dim for that addr byte.
_ADDR_LO_BY_J = (_SetDim.ADDR_B0_LO, _SetDim.ADDR_B1_LO, _SetDim.ADDR_B2_LO)
_ADDR_HI_BY_J = (_SetDim.ADDR_B0_HI, _SetDim.ADDR_B1_HI, _SetDim.ADDR_B2_HI)
# K-side threshold pairs the bake uses to pick the addr byte j position.
# Each entry: (positive_threshold_dim, negative_threshold_dim).
_K_THRESHOLDS_BY_J = (
    (_SetDim.L1H1, _SetDim.L1H0),  # j=0: addr byte 0 at d=1 from MEM
    (_SetDim.L1H2, _SetDim.L1H1),  # j=1: addr byte 1 at d=2
    (_SetDim.H0,   _SetDim.L1H2),  # j=2: addr byte 2 at d=3
)
_MEM_VAL_QUERY_DIMS = (
    _SetDim.MEM_VAL_B0,
    _SetDim.MEM_VAL_B1,
    _SetDim.MEM_VAL_B2,
    _SetDim.MEM_VAL_B3,
)


def _build_baked_attn(num_heads: int = 8, dim: int = 512) -> PureAttention:
    """Return a fresh PureAttention with only L13 mem-addr-gather weights set."""
    attn = PureAttention(dim=dim, num_heads=num_heads)
    HD = dim // num_heads
    S = 100.0
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        _set_layer13_mem_addr_gather(attn, S, _SetDim, HD)
    return attn


# ---------------------------------------------------------------------------
# 1. Op registration
# ---------------------------------------------------------------------------


def test_layer13_mem_addr_gather_op_is_registered():
    """The op is in all_core_ops, lands at layer 13, kind=block, migrated."""
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops

    ops = all_core_ops()
    found = [op for op in ops if op.name == "layer13_mem_addr_gather"]
    assert len(found) == 1, (
        f"Expected exactly 1 layer13_mem_addr_gather op, found {len(found)}"
    )
    op = found[0]
    assert op.layer_idx == 13
    assert op.kind == "block"
    assert op.migrated is True
    assert op.phase == 13
    assert op.declarative_authority == "spec_generated"


def test_layer13_mem_addr_gather_op_reads_writes_contract():
    """Reads include the opcode + address-source flags; writes are ADDR_B[012]_LO/HI."""
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops

    op = next(o for o in all_core_ops() if o.name == "layer13_mem_addr_gather")
    # The reads contract advertises the opcode + source flags that compose
    # MEM addressing for SI/SC/LI/LC even though the matrix-level bake
    # only consumes structural marker dims (Q gates on MEM_VAL_B*, K on
    # threshold dims around MEM_I). Upstream layers feed STACK0/AX_CARRY
    # into the CLEAN_EMBED stream at the addr byte K positions.
    for required_read in (
        "MARK_MEM", "MARK_STACK0", "AX_CARRY_LO", "AX_CARRY_HI",
        "OP_LI", "OP_LC", "OP_SI", "OP_SC", "MEM_ADDR_SRC",
    ):
        assert required_read in op.reads, f"reads missing {required_read}"
    assert op.writes == {
        "ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
        "ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI",
        # B7-4: ADDR_B0 lifecycle bit (head 0, slot 34).
        "ADDR_B0_VALID",
    }


def test_layer13_mem_addr_gather_claims_cover_three_heads():
    """The declared claim set covers all 3 heads x 32 V slots x CLEAN_EMBED.

    Head 0 additionally claims slot 34's V row (the B7-4 ADDR_B0_VALID
    lifecycle bit), so the expected count is 3*32 + 1 = 97.
    """
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops

    op = next(o for o in all_core_ops() if o.name == "layer13_mem_addr_gather")
    # Each head h in {0,1,2}: 16 LO + 16 HI V-row claims = 32 cells. Plus
    # head 0 owns one additional V slot (34) for ADDR_B0_VALID. Total 97.
    assert len(op.claims) == 3 * 32 + 1, (
        f"Expected 97 claims (96 gather + 1 VALID); got {len(op.claims)}"
    )
    # All claims live on layer 13 attn_W_v.
    layers_scopes = {(layer, scope) for layer, scope, _, _ in op.claims}
    assert layers_scopes == {(13, "attn_W_v")}
    # Heads referenced exactly cover 0, 1, 2.
    heads_seen = {ident.split("_")[0] for _, _, ident, _ in op.claims}
    assert heads_seen == {"0", "1", "2"}
    # The ADDR_B0_VALID lifecycle slot lives on head 0 only.
    valid_claims = [c for c in op.claims if c[3] == "L1H1+4"]
    assert valid_claims == [(13, "attn_W_v", "0_34", "L1H1+4")]


# ---------------------------------------------------------------------------
# 2. Static weight structure (per-head Q/K/V/O routing)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baked_attn() -> PureAttention:
    """One baked PureAttention shared by all structural tests."""
    return _build_baked_attn()


@pytest.mark.parametrize("j", [0, 1, 2])
def test_l13_head_q_fires_on_all_mem_val_byte_positions(baked_attn, j):
    """Q row 0 on head j weights MEM_VAL_B0..B3 (queries at all val byte positions)."""
    HD = baked_attn.W_q.shape[0] // baked_attn.num_heads
    base = j * HD
    L = 15.0  # matches helper constant
    for val_dim in _MEM_VAL_QUERY_DIMS:
        assert baked_attn.W_q[base, val_dim].item() == pytest.approx(L), (
            f"head {j}: Q row 0 missing MEM_VAL weight at dim {val_dim}"
        )


@pytest.mark.parametrize("j", [0, 1, 2])
def test_l13_head_k_picks_correct_addr_byte_position(baked_attn, j):
    """K[base, +threshold] = +L and K[base, -threshold] = -L per head."""
    HD = baked_attn.W_q.shape[0] // baked_attn.num_heads
    base = j * HD
    pos_dim, neg_dim = _K_THRESHOLDS_BY_J[j]
    L = 15.0
    assert baked_attn.W_k[base, pos_dim + _MEM_I].item() == pytest.approx(L)
    assert baked_attn.W_k[base, neg_dim + _MEM_I].item() == pytest.approx(-L)


@pytest.mark.parametrize("j", [0, 1, 2])
def test_l13_head_v_copies_clean_embed_nibbles(baked_attn, j):
    """V rows base+1..16 copy CLEAN_EMBED_LO; base+17..32 copy CLEAN_EMBED_HI."""
    HD = baked_attn.W_q.shape[0] // baked_attn.num_heads
    base = j * HD
    for k in range(16):
        assert baked_attn.W_v[base + 1 + k, _SetDim.CLEAN_EMBED_LO + k].item() == 1.0
        assert baked_attn.W_v[base + 17 + k, _SetDim.CLEAN_EMBED_HI + k].item() == 1.0


@pytest.mark.parametrize("j", [0, 1, 2])
def test_l13_head_o_writes_to_addr_b_j(baked_attn, j):
    """O routes head j's gathered values to ADDR_B{j}_LO/HI output dims."""
    HD = baked_attn.W_q.shape[0] // baked_attn.num_heads
    base = j * HD
    addr_lo = _ADDR_LO_BY_J[j]
    addr_hi = _ADDR_HI_BY_J[j]
    for k in range(16):
        assert baked_attn.W_o[addr_lo + k, base + 1 + k].item() == 1.0
        assert baked_attn.W_o[addr_hi + k, base + 17 + k].item() == 1.0


@pytest.mark.parametrize("j", [0, 1, 2])
def test_l13_head_anti_leakage_gate(baked_attn, j):
    """Slot 33 anti-leakage gate: Q on MEM_VAL_B0/CONST, K on CONST."""
    HD = baked_attn.W_q.shape[0] // baked_attn.num_heads
    base = j * HD
    L = 15.0
    assert baked_attn.W_q[base + 33, _SetDim.MEM_VAL_B0].item() == pytest.approx(L)
    assert baked_attn.W_q[base + 33, _SetDim.CONST].item() == pytest.approx(-L / 2)
    assert baked_attn.W_k[base + 33, _SetDim.CONST].item() == pytest.approx(L)


def test_l13_other_heads_remain_zero(baked_attn):
    """Heads 3..7 must stay zero -- bake only touches heads 0, 1, 2.

    Non-zero touches on a non-target head would indicate L13 spilled past
    its 3-head allocation -- a regression of the dim-ownership claim.
    """
    HD = baked_attn.W_q.shape[0] // baked_attn.num_heads
    for head in range(3, baked_attn.num_heads):
        base = head * HD
        # Q/K/V are indexed by head rows; O by head columns.
        for matrix_name in ("W_q", "W_k", "W_v"):
            row_slice = getattr(baked_attn, matrix_name)[base:base + HD]
            assert row_slice.abs().sum().item() == 0.0, (
                f"head {head}: {matrix_name} rows nonzero (L13 bake spilled)"
            )
        col_slice = baked_attn.W_o[:, base:base + HD]
        assert col_slice.abs().sum().item() == 0.0, (
            f"head {head}: W_o columns nonzero (L13 bake spilled)"
        )


# ---------------------------------------------------------------------------
# 2b. ADDR_B0_VALID lifecycle bit (B7-4 slot 97)
# ---------------------------------------------------------------------------


_VALID_SLOT = 34
_ADDR_B0_VALID = _SetDim.ADDR_B0_VALID


def test_l13_addr_b0_valid_slot_q_mirrors_head0_main_q(baked_attn):
    """Slot 34's Q row weights MEM_VAL_B0..B3 (mirrors head 0 slot 0).

    Same Q row as the primary gather so the VALID bit fires at the same
    MEM val byte positions where the gathered ADDR_B0 nibbles land.
    """
    L = 15.0
    for val_dim in _MEM_VAL_QUERY_DIMS:
        assert baked_attn.W_q[_VALID_SLOT, val_dim].item() == pytest.approx(L), (
            f"slot {_VALID_SLOT}: Q row missing MEM_VAL weight at dim {val_dim}"
        )


def test_l13_addr_b0_valid_slot_k_targets_mem_addr_byte0(baked_attn):
    """Slot 34's K row picks MEM addr byte 0 (L1H1[MEM]=+L, L1H0[MEM]=-L).

    Mirrors head 0 slot 0 so the softmax aligns the VALID lookup with the
    same MEM addr byte 0 row that the ADDR_B0_LO/HI gather uses.
    """
    L = 15.0
    assert baked_attn.W_k[_VALID_SLOT, _SetDim.L1H1 + _MEM_I].item() == pytest.approx(L)
    assert baked_attn.W_k[_VALID_SLOT, _SetDim.L1H0 + _MEM_I].item() == pytest.approx(-L)


def test_l13_addr_b0_valid_slot_v_reads_l1h1_mem(baked_attn):
    """Slot 34's V row reads ``L1H1+MEM_I``.

    Reading L1H1+MEM rather than CONST means the gathered value is 1.0 only
    when the attended row really is a MEM addr byte 0 row (L1H1+MEM=1
    there). Unrelated rows have L1H1+MEM=0 so leak through softmax-mass
    spread doesn't fake-fire ADDR_B0_VALID.
    """
    assert baked_attn.W_v[_VALID_SLOT, _SetDim.L1H1 + _MEM_I].item() == pytest.approx(1.0)
    # And explicitly NOT CONST (would defeat the lifecycle bit).
    assert baked_attn.W_v[_VALID_SLOT, _SetDim.CONST].item() == 0.0


def test_l13_addr_b0_valid_slot_o_routes_to_addr_b0_valid(baked_attn):
    """W_o[ADDR_B0_VALID, slot 34] = 1.0 routes gathered constant to the dim."""
    assert baked_attn.W_o[_ADDR_B0_VALID, _VALID_SLOT].item() == pytest.approx(1.0)


def test_l13_addr_b0_valid_slot_only_on_head_0(baked_attn):
    """Heads 1 and 2 must leave slot 34 untouched (ADDR_B1/B2_VALID NOT produced).

    Per B7-4 we only allocated ADDR_B0_VALID; B1/B2 VALID bits are reserved
    for later batches. If a future change accidentally turns on the VALID
    write on head 1 or 2, this test surfaces it so the new dim allocation
    gets reviewed.
    """
    HD = baked_attn.W_q.shape[0] // baked_attn.num_heads
    for head in (1, 2):
        base = head * HD
        # The slot 34 V row on heads 1/2 must remain entirely zero.
        v_row = baked_attn.W_v[base + _VALID_SLOT]
        assert v_row.abs().sum().item() == 0.0, (
            f"head {head}: slot {_VALID_SLOT} V row nonzero (ADDR_B{head}_VALID "
            f"NOT yet allocated; bake spilled)"
        )
        # And the W_o column for slot 34 on heads 1/2 must remain entirely zero.
        o_col = baked_attn.W_o[:, base + _VALID_SLOT]
        assert o_col.abs().sum().item() == 0.0, (
            f"head {head}: slot {_VALID_SLOT} W_o column nonzero"
        )


def _build_l13_residual_with_mem_step(num_heads: int = 8, dim: int = 512):
    """Return a fresh residual stream representing one MEM-op step.

    Layout (one VM step, capped at PureAttention.NUM_POSITIONS=8):
        0: MEM marker (L1H0/L1H1/L1H2/H0 all set at MEM_I)
        1: addr byte 0 (CLEAN_EMBED nibbles for 0x42)
        2: addr byte 1 (CLEAN_EMBED nibbles for 0x00)
        3: addr byte 2 (CLEAN_EMBED nibbles for 0x00)
        4..7: val bytes 0..3 (MEM_VAL_B{i} flags)
    """
    S_len = 8
    x = torch.zeros(1, S_len, dim)
    # MEM marker at position 0.
    x[0, 0, _SetDim.L1H0 + _MEM_I] = 1.0
    x[0, 0, _SetDim.L1H1 + _MEM_I] = 1.0
    x[0, 0, _SetDim.L1H2 + _MEM_I] = 1.0
    x[0, 0, _SetDim.H0 + _MEM_I] = 1.0
    # Addr byte K positions.
    for k, byte_val in enumerate((0x42, 0x00, 0x00)):
        pos = 1 + k
        if k == 0:
            x[0, pos, _SetDim.L1H1 + _MEM_I] = 1.0
            x[0, pos, _SetDim.L1H2 + _MEM_I] = 1.0
            x[0, pos, _SetDim.H0 + _MEM_I] = 1.0
        elif k == 1:
            x[0, pos, _SetDim.L1H2 + _MEM_I] = 1.0
            x[0, pos, _SetDim.H0 + _MEM_I] = 1.0
        else:
            x[0, pos, _SetDim.H0 + _MEM_I] = 1.0
        x[0, pos, _SetDim.CLEAN_EMBED_LO + (byte_val & 0xF)] = 1.0
        x[0, pos, _SetDim.CLEAN_EMBED_HI + ((byte_val >> 4) & 0xF)] = 1.0
    # MEM val byte positions fire Q.
    for i in range(4):
        x[0, 4 + i, _MEM_VAL_QUERY_DIMS[i]] = 1.0
    # CONST channel = 1.0 everywhere (always-on positional flag).
    x[0, :, _SetDim.CONST] = 1.0
    return x


def test_l13_addr_b0_valid_fires_at_mem_val_byte_positions():
    """ADDR_B0_VALID lands ~1.0 at MEM val byte positions when L13 produces.

    Pins the per-op fresh-write contract: at every MEM val byte position
    (positions 4..7 in the synthetic step layout), the residual delta on
    ADDR_B0_VALID after L13 attention should be ~1.0. The downstream tail
    rules (L10 addr0 family per B4-H §3.2) read ADDR_B0_VALID at the MEM
    val byte 0 row -- so we additionally check that specific position.
    """
    attn = _build_baked_attn()
    x = _build_l13_residual_with_mem_step()
    valid_before = x[0, :, _ADDR_B0_VALID].clone()
    with torch.no_grad():
        y = attn(x)
    valid_after = y[0, :, _ADDR_B0_VALID]
    delta = valid_after - valid_before
    # MEM val byte 0..3 should each receive ~+1.0 (the L13 attention writes
    # a residual delta on top of whatever was already there).
    for pos in (4, 5, 6, 7):
        assert delta[pos].item() > 0.5, (
            f"ADDR_B0_VALID delta at MEM val byte position {pos} should be "
            f"~+1.0 (L13 produced ADDR_B0); got {delta[pos].item()}"
        )


def test_l13_addr_b0_valid_zero_on_non_mem_sequence():
    """ADDR_B0_VALID stays ~0 across a sequence with no MEM context at all.

    Models the "stale frame" / post-STEP_END condition mentioned in the
    B7-4 brief: if the current step contains no MEM operation, L13 has
    nothing to gather and ADDR_B0_VALID should remain 0 on the entire
    residual stream. With V reading ``L1H1+MEM_I`` (which is 0 when no
    MEM marker is present anywhere), softmax cannot synthesize a value
    out of zero V entries.
    """
    attn = _build_baked_attn()
    S_len = 6
    dim = 512
    x = torch.zeros(1, S_len, dim)
    # Pure non-MEM background: only CONST is set, no MEM markers/threshold
    # heads or MEM_VAL_B* flags.
    x[0, :, _SetDim.CONST] = 1.0
    with torch.no_grad():
        y = attn(x)
    delta = y[0, :, _ADDR_B0_VALID] - x[0, :, _ADDR_B0_VALID]
    for pos in range(S_len):
        assert abs(delta[pos].item()) < 1e-4, (
            f"non-MEM sequence position {pos}: ADDR_B0_VALID delta should "
            f"be ~0 (L13 has nothing to gather); got {delta[pos].item()}"
        )


def test_l13_addr_b0_valid_not_asserted_on_non_query_rows():
    """ADDR_B0_VALID is only written at MEM val byte positions, not elsewhere.

    Models the "consumer reads only at val byte positions" half of the
    B7-4 contract. Even within a MEM step, the residual delta on
    ADDR_B0_VALID at the MEM marker (pos 0) and addr byte rows (pos 1-3)
    should be far below the fresh ~1.0 written at MEM val byte positions:
    those rows have no MEM_VAL_B* flag so slot 34 Q does not fire, and
    softmax mass spreads diluting the V contribution well below the
    locked-attention ~1.0.
    """
    attn = _build_baked_attn()
    x = _build_l13_residual_with_mem_step()
    with torch.no_grad():
        y = attn(x)
    delta = y[0, :, _ADDR_B0_VALID] - x[0, :, _ADDR_B0_VALID]
    # MEM val byte positions (4-7) should be ~1.0 (fresh write).
    for pos in (4, 5, 6, 7):
        assert delta[pos].item() > 0.5, (
            f"MEM val byte position {pos}: expected ~1.0 fresh-VALID; "
            f"got {delta[pos].item()}"
        )
    # Non-query positions (0-3) should be visibly weaker: with V[L1H1+MEM]
    # only nonzero at MEM-band rows and softmax uniform here, the
    # contribution is at most |MEM-band-rows| / S_len ≈ 2/8 = 0.25.
    fresh = delta[4].item()
    for pos in (0, 1, 2, 3):
        assert delta[pos].item() < fresh - 0.4, (
            f"non-query position {pos}: ADDR_B0_VALID delta ({delta[pos].item()}) "
            f"should be at least 0.4 less than fresh ({fresh}); slot 34 "
            f"may be missing its Q gate on MEM_VAL_B*"
        )


# ---------------------------------------------------------------------------
# 3. Per-opcode behaviour (LI/LC/SI/SC all share the gather)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opcode_dim", [
    _SetDim.OP_LI,
    _SetDim.OP_LC,
    _SetDim.OP_SI,
    _SetDim.OP_SC,
])
def test_l13_attention_does_not_directly_gate_on_opcode(baked_attn, opcode_dim):
    """The L13 mem-addr gather is opcode-AGNOSTIC at the attention level.

    Opcode gating happens upstream (L7 memory heads + L9 stack/addr relays
    pick STACK0 vs AX_CARRY into the addr byte K positions; the MEM marker
    is only emitted on the bytecode rows that actually do a memory op).
    By the time L13 fires, the addr byte K positions already carry the
    right nibble values regardless of which of LI/LC/SI/SC drove the row.

    This test pins that contract: every Q / K row for heads 0-2 must have
    zero weight on the four memory-opcode flags (so future "let's gate L13
    on OP_LI" changes are caught immediately).
    """
    HD = baked_attn.W_q.shape[0] // baked_attn.num_heads
    for head in range(3):
        base = head * HD
        for row in range(HD):
            assert baked_attn.W_q[base + row, opcode_dim].item() == 0.0, (
                f"head {head} row {row}: unexpected Q weight on opcode dim "
                f"{opcode_dim}"
            )
            assert baked_attn.W_k[base + row, opcode_dim].item() == 0.0, (
                f"head {head} row {row}: unexpected K weight on opcode dim "
                f"{opcode_dim}"
            )


# ---------------------------------------------------------------------------
# 4. Symbolic forward: gathered nibbles land at MEM val byte positions.
# ---------------------------------------------------------------------------


def _gather_via_forward(
    addr_bytes: tuple[int, int, int],
    val_byte_idx: int,
    num_heads: int = 8,
    dim: int = 512,
) -> torch.Tensor:
    """Run L13-only attention on a synthetic sequence and return the residual
    ADDR_B*_LO/HI band at the val_byte_idx-th MEM val byte position.

    ``addr_bytes`` are the three 8-bit MEM address bytes; their LO/HI
    nibbles are placed into the CLEAN_EMBED_LO/HI lanes at each MEM addr
    byte position. Sequence layout (one VM step):

        positions:        meaning
        0:                MEM marker (L1H0[MEM]=1, L1H1[MEM]=1, ...)
        1:                addr byte 0   (L1H1[MEM]=1, NOT L1H0[MEM])
        2:                addr byte 1   (L1H2[MEM]=1, NOT L1H1[MEM])
        3:                addr byte 2   (H0[MEM]=1,   NOT L1H2[MEM])
        4..7:             val bytes 0..3 (MEM_VAL_B0..B3 = 1)
    """
    attn = _build_baked_attn(num_heads=num_heads, dim=dim)
    S_len = 8
    x = torch.zeros(1, S_len, dim)

    # MEM marker at position 0: nothing to set for the gather contract
    # (Q doesn't fire here; K thresholds happen to evaluate "in MEM band").
    # We still set L1H0[MEM]=1 for completeness so the K subtraction at
    # j=0 (-L1H0[MEM]) blocks marker self-attention.
    x[0, 0, _SetDim.L1H0 + _MEM_I] = 1.0
    x[0, 0, _SetDim.L1H1 + _MEM_I] = 1.0
    x[0, 0, _SetDim.L1H2 + _MEM_I] = 1.0
    x[0, 0, _SetDim.H0 + _MEM_I] = 1.0

    # Addr byte K positions -- each gets the threshold set that the bake
    # picks out, and CLEAN_EMBED_{LO,HI} carrying that byte's nibble value.
    for k, byte_val in enumerate(addr_bytes):
        pos = 1 + k
        # Threshold lattice (matches K's W_k linear combo per j).
        # At pos=1: L1H1[MEM]=1, L1H2[MEM]=1, H0[MEM]=1, L1H0[MEM]=0.
        # At pos=2: L1H2[MEM]=1, H0[MEM]=1, L1H1[MEM]=0.
        # At pos=3: H0[MEM]=1; all lower thresholds zero.
        if k == 0:
            x[0, pos, _SetDim.L1H1 + _MEM_I] = 1.0
            x[0, pos, _SetDim.L1H2 + _MEM_I] = 1.0
            x[0, pos, _SetDim.H0 + _MEM_I] = 1.0
        elif k == 1:
            x[0, pos, _SetDim.L1H2 + _MEM_I] = 1.0
            x[0, pos, _SetDim.H0 + _MEM_I] = 1.0
        else:  # k == 2
            x[0, pos, _SetDim.H0 + _MEM_I] = 1.0
        lo = byte_val & 0xF
        hi = (byte_val >> 4) & 0xF
        x[0, pos, _SetDim.CLEAN_EMBED_LO + lo] = 1.0
        x[0, pos, _SetDim.CLEAN_EMBED_HI + hi] = 1.0

    # MEM val byte positions: each carries its MEM_VAL_B{i} = 1 flag so
    # Q fires there. Index 0 in MEM_VAL_DIMS corresponds to val byte 0 at
    # position 4 (d=4 from MEM marker).
    for i in range(4):
        pos = 4 + i
        x[0, pos, _MEM_VAL_QUERY_DIMS[i]] = 1.0

    # CONST channel needed by the anti-leakage gate path.
    x[0, :, _SetDim.CONST] = 1.0

    with torch.no_grad():
        y = attn(x)

    # Read the ADDR_B*_LO/HI residual band at the val byte under test.
    target_pos = 4 + val_byte_idx
    out: dict[str, torch.Tensor] = {}
    for j in range(3):
        out[f"ADDR_B{j}_LO"] = y[0, target_pos,
                                 _ADDR_LO_BY_J[j]:_ADDR_LO_BY_J[j] + 16].clone()
        out[f"ADDR_B{j}_HI"] = y[0, target_pos,
                                 _ADDR_HI_BY_J[j]:_ADDR_HI_BY_J[j] + 16].clone()
    return out


@pytest.mark.parametrize(
    "addr_bytes,val_byte_idx",
    [
        # ((b0, b1, b2), val_byte_idx)
        ((0x00, 0x00, 0x00), 0),
        ((0x42, 0x00, 0x00), 0),
        ((0x00, 0x02, 0x00), 1),
        # STACK0-like multi-byte address (SI/SC source from stack push).
        ((0xAB, 0xCD, 0x00), 2),
        # AX_CARRY-like address with high nibble carry.
        ((0xFF, 0x05, 0x00), 3),
        ((0x10, 0xFF, 0xFF), 0),
    ],
)
def test_l13_symbolic_forward_gathers_addr_nibbles_to_val_byte(
    addr_bytes, val_byte_idx,
):
    """Driving CLEAN_EMBED nibbles at addr byte K positions produces matching
    ADDR_B{j}_LO/HI argmaxes at every MEM val byte position.

    This is the load-bearing semantic check: irrespective of which
    upstream source (STACK0 vs AX_CARRY) wrote the addr bytes, L13's
    attention has to copy them, one head per byte, to the val byte rows
    where L15 reads them. We verify per-head argmax matches the nibble
    we injected.
    """
    out = _gather_via_forward(addr_bytes, val_byte_idx)
    for j, byte_val in enumerate(addr_bytes):
        expected_lo = byte_val & 0xF
        expected_hi = (byte_val >> 4) & 0xF
        lo_argmax = int(out[f"ADDR_B{j}_LO"].argmax().item())
        hi_argmax = int(out[f"ADDR_B{j}_HI"].argmax().item())
        assert lo_argmax == expected_lo, (
            f"val byte {val_byte_idx} j={j}: ADDR_B{j}_LO argmax "
            f"{lo_argmax} != expected {expected_lo} (byte=0x{byte_val:02X}). "
            f"Per-slot residual: {out[f'ADDR_B{j}_LO'].tolist()}"
        )
        assert hi_argmax == expected_hi, (
            f"val byte {val_byte_idx} j={j}: ADDR_B{j}_HI argmax "
            f"{hi_argmax} != expected {expected_hi} (byte=0x{byte_val:02X}). "
            f"Per-slot residual: {out[f'ADDR_B{j}_HI'].tolist()}"
        )


# ---------------------------------------------------------------------------
# 5. Declarative claim verifier (Mode A static)
# ---------------------------------------------------------------------------
# Reuses the session-scoped ``static_claims_report`` fixture and shared
# ``_per_op_audit`` helpers so this harness shares the (~25-60s) verifier
# bake with sibling per-layer audit modules (test_l0_*, test_l2_*, ...).


from ._per_op_audit import assert_no_drift, assert_op_fires  # noqa: E402


@pytest.mark.lowering
def test_l13_mem_addr_gather_has_no_declaration_drift(static_claims_report):
    assert_no_drift(static_claims_report, "L13", "layer13_mem_addr_gather")


@pytest.mark.lowering
def test_l13_mem_addr_gather_fires_during_bake(static_claims_report):
    assert_op_fires(static_claims_report, "L13", "layer13_mem_addr_gather")


# ---------------------------------------------------------------------------
# 6. End-to-end smoke: tiny C-style program exercising SI/LI.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Pre-existing Phase-7 SI/LI breakage upstream of L13 mem_addr_gather: "
        "the gather itself (heads 0-2 on L13) is verified by the symbolic "
        "forward test above, but end-to-end value roundtrip depends on "
        "_set_layer14_mem_generation / _set_layer15_memory_lookup gaps "
        "(see TestSmokeMemory::test_si_li_roundtrip in test_smoke_pure_neural.py). "
        "This xfail tracks recovery: when the upstream fixes land the test "
        "should flip to xpass and we can drop the marker."
    ),
    strict=False,
)
def test_l13_e2e_int_assign_and_read_roundtrip(pure_neural_runner, make_bytecode):
    """Synthetic ``int x; x = 42; printf(x);`` equivalent through SI/LI.

    Mirrors TestSmokeMemory::test_si_li_roundtrip from test_smoke_pure_neural.py
    so the parent-agent E2E recipe still has a per-op tail to check that the
    L13 gather is not the bottleneck. Kept ``xfail(strict=False)`` for the
    reason noted in the decorator.
    """
    from neural_vm.embedding import Opcode

    bytecode = make_bytecode([
        (Opcode.IMM, 0x200),
        Opcode.PSH,
        (Opcode.IMM, 42),
        Opcode.SI,
        (Opcode.IMM, 0x200),
        Opcode.LI,
        Opcode.EXIT,
    ])
    _, result = pure_neural_runner.run(bytecode, b"", max_steps=30)
    assert result == 42
