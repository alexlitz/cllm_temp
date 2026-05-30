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
    }


def test_layer13_mem_addr_gather_claims_cover_three_heads():
    """The declared claim set covers all 3 heads x 32 V slots x CLEAN_EMBED."""
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops

    op = next(o for o in all_core_ops() if o.name == "layer13_mem_addr_gather")
    # Each head h in {0,1,2}: 16 LO + 16 HI V-row claims = 32 cells.
    # Plus 3 heads = 96 cells total.
    assert len(op.claims) == 3 * 32, (
        f"Expected 96 claims; got {len(op.claims)}"
    )
    # All claims live on layer 13 attn_W_v.
    layers_scopes = {(layer, scope) for layer, scope, _, _ in op.claims}
    assert layers_scopes == {(13, "attn_W_v")}
    # Heads referenced exactly cover 0, 1, 2.
    heads_seen = {ident.split("_")[0] for _, _, ident, _ in op.claims}
    assert heads_seen == {"0", "1", "2"}


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
