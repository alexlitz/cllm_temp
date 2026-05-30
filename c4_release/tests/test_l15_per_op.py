"""Per-op audit harness for Layer 15 (PSH SP/BP byte producer + STACK0 lookup).

Layer 15 owns the load-side memory pipeline plus the PSH-specific stack
byte producers that overlay the generic nibble-copy FFN:

* ``layer15_memory_lookup`` -- attention heads 0-3 implementing LI/LC
  and ``STACK0`` pop-group loads from the historical MEM section.
  Head 0 dual-roles as the STACK0 pop-group lookup; heads 1-3 handle
  the subsequent byte positions via BYTE_INDEX_{0,1,2} gating. The op
  declares a 124-cell ``W_v`` claim map (4 heads x (16 LO + 15 HI)
  slots reading ``CLEAN_EMBED_LO/HI``) -- the only L15 op with claims.
* ``layer15_nibble_copy`` (block-pinned) -- the FFN bake that lowers
  ``make_l15_nibble_copy_ir`` into L15's FFN. Includes 32 generic
  nibble-copy units, 8 PSH SP/BP byte producer units (PSH SP byte0 ->
  byte1=0xff, PSH SP byte1 -> byte2=0x00, PSH SP byte2 -> byte3=0x00,
  PSH BP byte1 -> byte2=0x01 preserve), and 2 first-step LEA AX byte2
  units. Ships without claims today (claim authoring not yet done for
  FFN block ops); tracked as absent so a future claim addition forces
  the audit list to be updated.
* ``nibble_copy_ffn`` (topology anchor) -- a no-op Operation that
  reserves the FFN slot for the bake above. No claims by design.
* ``layer15_store_stack0_sp_byte0_addr`` -- attention head 12 that
  copies the current post-pop SP byte0 into ``ADDR_B0`` at store
  STACK0 markers (powers the SI/SC top-store address). No claims today.
* ``layer15_si_mem_addr0_from_stack0`` -- attention head 13 that
  overrides the SI/SC MEM addr byte0 from the pre-store STACK0 byte0
  (clean CLEAN_EMBED path, sidestepping the OUTPUT residue L14 uses
  for PSH). No claims today.
* ``l15_attention_resize`` -- structural reshape (no W_q/W_k/W_v rows
  changed, just head count); ``writes=set()`` and no claims by design.

This module complements ``test_declarative_ffn_bakes_l15.py`` (which is
already comprehensive on the per-rule weight layout) with three things
that file does NOT cover:

1. ``static_claims_report``-backed declaration-drift gates so a future
   regression on the ``layer15_memory_lookup`` ``W_v`` claim map fails
   here with a layer-labelled error rather than as a slot index inside
   a 1096 backtrace.
2. ``assert_op_absent`` watchdogs on the empty-claims ops -- if anyone
   adds claims to ``layer15_nibble_copy`` etc. (likely for the PSH SP
   byte producer subset), this module's failure tells the next author
   to migrate the op into the drift-checked list.
3. Two focused symbolic-forward tests: PSH SP byte producer (drive a
   minimal PureFFN through ``lower_l15_psh_stack_ir`` and verify the
   byte-1 -> 0xff / byte-2 -> 0x00 / byte-3 -> 0x00 outputs fire at
   the PSH-at-SP signature and stay silent at non-PSH signatures), and
   STACK0 lookup (drive a minimal PureAttention through
   ``_set_layer15_memory_lookup`` and verify head 0 attends to the
   correct MEM val byte 0 at a STACK0 marker query while the other
   heads remain on their byte-index gates).

The ``static_claims_report`` fixture lives in ``conftest.py`` at
session scope so the ~25-60s verifier bake runs once per pytest
session regardless of how many per-layer harnesses are present.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.unified_compiler.ops.l15_ops import (  # noqa: E402
    lower_l15_psh_stack_ir,
    make_l15_psh_stack_ir,
)
from neural_vm.vm_step import _SetDim, _set_layer15_memory_lookup  # noqa: E402

from ._per_op_audit import (  # noqa: E402
    assert_no_drift,
    assert_op_absent,
    assert_op_fires,
)


# ---------------------------------------------------------------------------
# Op inventory: which L15 ops carry claims today, which do not.
# ---------------------------------------------------------------------------
#
# verify_claims_static only inspects ops with non-empty ``claims``. As of
# the audit baseline only ``layer15_memory_lookup`` declares claims (the
# 124-cell W_v map across heads 0-3). The other L15 ops are valid bakes
# but author-empty on claims; the absent-list keeps them watchdogged so
# a future claim addition is loud.

L15_OPS_WITH_CLAIMS = (
    "layer15_memory_lookup",
)

L15_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "nibble_copy_ffn",
    "layer15_nibble_copy",
    "layer15_store_stack0_sp_byte0_addr",
    "layer15_si_mem_addr0_from_stack0",
    "l15_attention_resize",
)


# ---------------------------------------------------------------------------
# 1. Drift checks (only meaningful for ops with non-empty claims).
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L15_OPS_WITH_CLAIMS)
def test_l15_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L15", op_name)


# ---------------------------------------------------------------------------
# 2. Fires-during-bake checks (op dispatched and emitted at least one cell).
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L15_OPS_WITH_CLAIMS)
def test_l15_op_fires_during_bake(
    static_claims_report, op_name: str
) -> None:
    assert_op_fires(static_claims_report, "L15", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L15_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD)
def test_l15_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L15", op_name)


# ---------------------------------------------------------------------------
# 3. Symbolic forward: PSH SP/BP byte producer.
# ---------------------------------------------------------------------------
#
# The PSH SP byte producer rules live in ``make_l15_psh_stack_ir`` and
# fire only at the (PSH_AT_SP + H1[sp_i] + IS_BYTE + BYTE_INDEX_X)
# signature emitted by upstream layers for PSH-at-SP byte positions.
# We drive the IR through both the symbolic interpreter (``symbolic_ffn``)
# and a real ``PureFFN`` bake (via ``lower_l15_psh_stack_ir``) so any
# drift between IR and lowering surfaces here before it shows up in the
# 1096 backtrace.


_SP_I = 2
_BP_I = 3


def _psh_sp_state(byte_index_name: str) -> dict[str, float]:
    """Build the residual-state map for PSH at SP byte position N."""
    return {
        "PSH_AT_SP": 1.0,
        f"H1+{_SP_I}": 1.0,
        f"H1+{_BP_I}": 0.0,  # MUST be zero so we don't accidentally trigger BP
        f"H4+{_BP_I}": 0.0,
        "H1+10": 0.0,
        "IS_BYTE": 1.0,
        byte_index_name: 1.0,
    }


def _psh_bp_state(byte_index_name: str) -> dict[str, float]:
    """Build the residual-state map for PSH at BP byte position N.

    Note: the BP preserve rule uses ``byte_index_name`` BYTE_INDEX_1 to
    predict byte 2 = 0x01. The condition asks for H1+bp_i, IS_BYTE,
    BYTE_INDEX_X.
    """
    return {
        "PSH_AT_SP": 1.0,
        f"H1+{_BP_I}": 1.0,
        f"H1+{_SP_I}": 0.0,
        f"H4+{_BP_I}": 0.0,
        "H1+10": 0.0,
        "IS_BYTE": 1.0,
        byte_index_name: 1.0,
    }


def test_l15_psh_sp_byte0_position_predicts_byte1_ff_symbolic():
    """PSH SP byte 0 position -> byte 1 = 0xff (low nibble 15, high nibble 15).

    Output is in semantic write-units (the lowering scales by 1/S).
    Each LO/HI write is +4 to slot 15 and -4 to slot 0 so the argmax
    across all 16 lo/hi slots lands at 15.
    """
    ir = make_l15_psh_stack_ir()
    state = _psh_sp_state("BYTE_INDEX_0")
    out = ir.symbolic_ffn(state)
    # Byte-1 == 0xff means low nibble 15 AND high nibble 15.
    assert out.get("OUTPUT_LO+15", 0.0) == 4.0, (
        f"Expected OUTPUT_LO+15 = +4.0 (lo nibble 15 for 0xff), "
        f"got {out.get('OUTPUT_LO+15')}; full state: {out}"
    )
    assert out.get("OUTPUT_LO+0", 0.0) == -4.0, (
        f"Expected OUTPUT_LO+0 = -4.0 (cancel competing 0x0 bias), "
        f"got {out.get('OUTPUT_LO+0')}"
    )
    assert out.get("OUTPUT_HI+15", 0.0) == 4.0
    assert out.get("OUTPUT_HI+0", 0.0) == -4.0


@pytest.mark.parametrize(
    "byte_index_name",
    ["BYTE_INDEX_1", "BYTE_INDEX_2"],
    ids=["sp_byte1->byte2=0x00", "sp_byte2->byte3=0x00"],
)
def test_l15_psh_sp_byte_position_predicts_zero_symbolic(byte_index_name: str):
    """PSH SP byte 1 / byte 2 positions -> following byte = 0x00."""
    ir = make_l15_psh_stack_ir()
    state = _psh_sp_state(byte_index_name)
    out = ir.symbolic_ffn(state)
    # Byte == 0x00 means low nibble 0 AND high nibble 0.
    # Rule writes ("OUTPUT_LO+0", 2.0), ("OUTPUT_HI+0", 2.0).
    assert out.get("OUTPUT_LO+0", 0.0) == 2.0, (
        f"Expected OUTPUT_LO+0 = +2.0 for PSH SP {byte_index_name}, "
        f"got {out.get('OUTPUT_LO+0')}"
    )
    assert out.get("OUTPUT_HI+0", 0.0) == 2.0


def test_l15_psh_bp_byte1_position_preserves_stack_init_byte2_symbolic():
    """PSH BP byte 1 position -> byte 2 = 0x01 (STACK_INIT preserve).

    PSH leaves BP unchanged; STACK_INIT byte 2 is 0x01.
    """
    ir = make_l15_psh_stack_ir()
    state = _psh_bp_state("BYTE_INDEX_1")
    out = ir.symbolic_ffn(state)
    # 0x01 -> low nibble 1, high nibble 0. The rules write +4 to OUTPUT_LO+1
    # / -4 to OUTPUT_LO+0 for LO; +4 to OUTPUT_HI+0 for HI.
    assert out.get("OUTPUT_LO+1", 0.0) == 4.0, (
        f"Expected OUTPUT_LO+1 = +4.0 (lo nibble 1 for 0x01), got "
        f"{out.get('OUTPUT_LO+1')}"
    )
    assert out.get("OUTPUT_LO+0", 0.0) == -4.0
    assert out.get("OUTPUT_HI+0", 0.0) == 4.0


def test_l15_psh_sp_byte_rules_do_not_fire_on_non_psh_signatures_symbolic():
    """Sanity: without PSH_AT_SP the SP-byte producer rules stay silent.

    A nibble-copy-style byte position (IS_BYTE + BYTE_INDEX_0 only) should
    leave OUTPUT_LO/HI untouched by these PSH-specific rules.
    """
    ir = make_l15_psh_stack_ir()
    state = {
        "PSH_AT_SP": 0.0,
        f"H1+{_SP_I}": 1.0,
        "IS_BYTE": 1.0,
        "BYTE_INDEX_0": 1.0,
    }
    out = ir.symbolic_ffn(state)
    for k in range(16):
        assert out.get(f"OUTPUT_LO+{k}", 0.0) == 0.0, (
            f"PSH-specific rules should be silent without PSH_AT_SP; "
            f"OUTPUT_LO+{k} = {out.get(f'OUTPUT_LO+{k}')}"
        )
        assert out.get(f"OUTPUT_HI+{k}", 0.0) == 0.0


def test_l15_psh_sp_byte0_lowered_ffn_predicts_byte1_ff():
    """End-to-end: lower the PSH stack IR into a real PureFFN and run it.

    Drives a synthetic 512-d residual where every PSH-at-SP byte0 flag
    is set, then runs the SwiGLU forward and checks that the LO/HI byte
    argmax lands at 15 (==0xf) and that the byte-0 slot is suppressed.
    This guards against any drift between IR and lowering math.
    """
    from neural_vm.base_layers import PureFFN
    from neural_vm.unified_compiler.primitives import Primitives

    S = 100.0
    ffn = PureFFN(dim=512, hidden_dim=42)
    rules = make_l15_psh_stack_ir().layer(0).ffn.rules
    psh_dim_positions = Primitives.dim_positions_from_bd(
        _SetDim,
        Primitives.ffn_rule_dim_names(rules),
    )
    end = lower_l15_psh_stack_ir(ffn, psh_dim_positions, start_unit=0, S=S)
    assert end > 0, "lower_l15_psh_stack_ir should advance the unit pointer"

    # Build a one-position residual carrying the PSH SP byte0 signature.
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.PSH_AT_SP] = 1.0
    x[0, 0, _SetDim.H1 + _SP_I] = 1.0
    x[0, 0, _SetDim.IS_BYTE] = 1.0
    x[0, 0, _SetDim.BYTE_INDEX_0] = 1.0

    with torch.no_grad():
        y = ffn(x)
    # Subtract the residual identity so we see the FFN delta only.
    delta = y - x

    lo_delta = delta[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16]
    hi_delta = delta[0, 0, _SetDim.OUTPUT_HI:_SetDim.OUTPUT_HI + 16]
    # Argmax over the 16-slot LO/HI bands must be the 0xf nibble (==15).
    assert int(lo_delta.argmax().item()) == 15, (
        f"PSH SP byte0 -> byte1 LO argmax should be 15 (0xf), got "
        f"{int(lo_delta.argmax().item())}; per-slot delta: {lo_delta.tolist()}"
    )
    assert int(hi_delta.argmax().item()) == 15, (
        f"PSH SP byte0 -> byte1 HI argmax should be 15 (0xf), got "
        f"{int(hi_delta.argmax().item())}; per-slot delta: {hi_delta.tolist()}"
    )
    # Slot 0 must be the most negative (suppressed) on each band.
    assert int(lo_delta.argmin().item()) == 0
    assert int(hi_delta.argmin().item()) == 0


# ---------------------------------------------------------------------------
# 4. Symbolic forward: STACK0 lookup (which heads, which slots).
# ---------------------------------------------------------------------------
#
# ``_set_layer15_memory_lookup`` wires heads 0-3 to dual-role between
# LI/LC at AX and pop-group loads at STACK0. Head 0 owns the STACK0
# byte-0 lookup (CMP[3]=POP relay activates it at the STACK0 marker),
# heads 1-3 own byte 1/2/3 lookups gated by BYTE_INDEX_{0,1,2}.
# V slots 32..47 read CLEAN_EMBED_LO[k]; V slots 48..62 read
# CLEAN_EMBED_HI[k]; O writes OUTPUT_LO/HI[k] from those V slots.
#
# These tests pin the slot routing AND drive a tiny attention forward
# that demonstrates head 0 selects the MEM val byte 0 K position over a
# non-store baseline, which is the load-bearing behaviour for STACK0
# pop-group reads.


_MEM_I = 4
_HD = 64


@pytest.fixture(scope="module")
def l15_lookup_attn():
    """PureAttention with ``_set_layer15_memory_lookup`` baked (module-scoped).

    Note: ``_set_layer15_memory_lookup`` internally calls
    ``_suppress_l15_lookup_during_current_store_generation`` which (a)
    re-scales the per-head W_o to ``value_scale = 40.0`` so the selected
    memory byte is authoritative against the pre-L15 OUTPUT residue, and
    (b) shifts heads 1-3 K[3] byte selection by one index relative to
    the bare bake (head 1 -> MEM_VAL_B2, head 2 -> MEM_VAL_B3, head 3
    -> H2/H3[MEM] thresholds for byte 3 source position). Tests below
    pin the post-suppress shape since that is what production runs.

    Module-scoped because the bake is read-only here -- no test mutates
    the returned PureAttention -- so sharing avoids ~6x duplicate bake
    work for the K[3]/V/O assertions below.
    """
    from neural_vm.base_layers import PureAttention

    attn = PureAttention(dim=512, num_heads=8)
    with torch.no_grad():
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        _set_layer15_memory_lookup(attn, 100.0, _SetDim, _HD)
    return attn


def test_l15_lookup_v_slots_route_clean_embed_to_output(l15_lookup_attn):
    """Per-head V/O routing: V[h*HD + 32 + k] reads CLEAN_EMBED_LO[k]
    and O[OUTPUT_LO+k] reads back from that same V slot. Same for HI.

    This mirrors the ``layer15_memory_lookup`` claim grid declared at
    Operation construction (heads 0-3 only), so a drift in the V/O
    routing should also surface here as a per-slot assertion failure
    (before the static report rolls everything into a single declared/
    observed bag).

    ``W_o`` is the post-suppress ``value_scale = 40.0`` from
    ``_suppress_l15_lookup_during_current_store_generation``, not the
    raw ``1.0`` from ``_set_layer15_memory_lookup`` itself. The suppress
    helper exists to make the selected memory byte authoritative against
    the pre-L15 OUTPUT residue; if the value scale drifts, the entire
    L15 load pipeline silently underperforms.
    """
    VALUE_SCALE = 40.0
    attn = l15_lookup_attn
    for h in range(4):
        base = h * _HD
        for k in range(16):
            assert attn.W_v.data[base + 32 + k, _SetDim.CLEAN_EMBED_LO + k] == 1.0, (
                f"V[h={h} slot=32+{k}] should read CLEAN_EMBED_LO+{k}"
            )
            assert attn.W_o.data[_SetDim.OUTPUT_LO + k, base + 32 + k] == VALUE_SCALE, (
                f"O[OUTPUT_LO+{k}] should read back from V slot "
                f"{base + 32 + k} with value_scale={VALUE_SCALE}; got "
                f"{float(attn.W_o.data[_SetDim.OUTPUT_LO + k, base + 32 + k])}"
            )
            if k < 15:
                assert attn.W_v.data[base + 48 + k, _SetDim.CLEAN_EMBED_HI + k] == 1.0
                assert (
                    attn.W_o.data[_SetDim.OUTPUT_HI + k, base + 48 + k]
                    == VALUE_SCALE
                )


def test_l15_lookup_head0_stack0_marker_attends_mem_val_byte0_over_baseline(
    l15_lookup_attn,
):
    """Head 0 STACK0 query prefers a stored MEM-val-byte0 K over a plain key.

    Drives a 2-position attention forward: position 0 is the STACK0
    marker query with CMP[3]=POP relay set, position 1 is a stored MEM
    val byte 0 (MARK_MEM=1 a few hops behind, MEM_STORE=1, MEM_VAL_B0
    target is the v0 token at d=5 from MEM, marked here by L2H0[MEM]=1
    AND H1[MEM]=0). The Q-K score budget defined in the helper
    docstring (head 0, target+store, perfect address match) should make
    the per-head softmax prefer K position 1 over a competing K
    position 2 that lacks the store/byte-selection K features.
    """
    attn = l15_lookup_attn
    base = 0  # head 0

    # Construct Q for the STACK0 marker (head 0's pop-group target).
    # Dim 0 (target gate): zero non-target penalty by setting STACK0+POP relay.
    # Dim 1 (store anchor): require LI relay or LC relay or CMP[3].
    # Dim 28 (per-head position gate): provide MARK_STACK0 to neutralize CONST*-500.
    q_stack0 = torch.zeros(512)
    q_stack0[_SetDim.CONST] = 1.0
    q_stack0[_SetDim.MARK_STACK0] = 1.0
    q_stack0[_SetDim.CMP + 3] = 1.0  # POP relay -> activates head 0
    q_stack0[_SetDim.HAS_SE] = 1.0

    # Build a "store + correct byte" key: K position carries MEM_STORE,
    # L2H0[MEM]=1, H1[MEM]=0 (byte 0 from MEM marker).
    k_store_b0 = torch.zeros(512)
    k_store_b0[_SetDim.CONST] = 1.0
    k_store_b0[_SetDim.MEM_STORE] = 1.0
    k_store_b0[_SetDim.L2H0 + _MEM_I] = 1.0

    # Build a "no-store, no byte-select" key as the baseline competitor.
    k_baseline = torch.zeros(512)
    k_baseline[_SetDim.CONST] = 1.0

    # Per-head Q·K score using the head 0 row band.
    q_proj = attn.W_q.data[base:base + _HD] @ q_stack0
    k_store_proj = attn.W_k.data[base:base + _HD] @ k_store_b0
    k_baseline_proj = attn.W_k.data[base:base + _HD] @ k_baseline

    score_store = float((q_proj * k_store_proj).sum())
    score_baseline = float((q_proj * k_baseline_proj).sum())
    assert score_store > score_baseline, (
        f"Head 0 STACK0 query should score the stored MEM val byte 0 K "
        f"higher than a baseline non-store K: store={score_store:.1f} "
        f"baseline={score_baseline:.1f}"
    )


def test_l15_lookup_heads_byte_index_gating_after_suppress(l15_lookup_attn):
    """Heads 1-3 Q[3] byte-selection gate stays on BYTE_INDEX_{h-1}.

    The Q-side gate is set by ``_set_layer15_memory_lookup`` and not
    rewritten by the suppress helper; head h fires only when its
    designated byte-index flag is on (head 1 -> BYTE_INDEX_0 query
    predicting byte 1; head 2 -> BYTE_INDEX_1; head 3 -> BYTE_INDEX_2).
    """
    attn = l15_lookup_attn
    for h in (1, 2, 3):
        base = h * _HD
        byte_index_dim = (
            _SetDim.BYTE_INDEX_0,
            _SetDim.BYTE_INDEX_1,
            _SetDim.BYTE_INDEX_2,
        )[h - 1]
        assert attn.W_q.data[base + 3, byte_index_dim] == 60.0, (
            f"Head {h} Q[3, BYTE_INDEX_{h-1}] should be 60.0"
        )


def test_l15_lookup_heads_post_suppress_k3_byte_source_shift(l15_lookup_attn):
    """K[3] byte-source selection after suppress: head 1 -> B2, head 2 -> B3.

    ``_set_layer15_memory_lookup`` sets K[3] to MEM_VAL_B{h} for heads
    h in {1,2,3}. ``_suppress_l15_lookup_during_current_store_generation``
    then zeroes the old assignments and shifts heads 1/2 by one index
    (head 1 -> MEM_VAL_B2, head 2 -> MEM_VAL_B3) and switches head 3
    to threshold flags (H2+MEM_I, H3+MEM_I) rather than a dedicated dim.
    This shift exists because the source row at the load-target byte
    position no longer reliably carries the MEM_VAL flag the bare bake
    expects -- the suppress helper compensates by looking at the
    following-byte flag instead. Pin it so a drift here can't silently
    misroute the per-byte load.
    """
    attn = l15_lookup_attn
    MEM_VAL_BY_B = {
        1: _SetDim.MEM_VAL_B1,
        2: _SetDim.MEM_VAL_B2,
        3: _SetDim.MEM_VAL_B3,
    }
    # Per-head expected hot MEM_VAL_B*: head 3 has none (uses thresholds).
    HEAD_HOT_B = {1: 2, 2: 3}

    # Head 1: MEM_VAL_B2 only.
    assert attn.W_k.data[1 * _HD + 3, _SetDim.MEM_VAL_B2] == 60.0
    # Head 2: MEM_VAL_B3 only.
    assert attn.W_k.data[2 * _HD + 3, _SetDim.MEM_VAL_B3] == 60.0
    # Head 3: threshold flags, no MEM_VAL_B*.
    assert attn.W_k.data[3 * _HD + 3, _SetDim.H3 + _MEM_I] == 60.0
    assert attn.W_k.data[3 * _HD + 3, _SetDim.H2 + _MEM_I] == -60.0

    # All other MEM_VAL_B* slots zeroed per head.
    for h in (1, 2, 3):
        hot_b = HEAD_HOT_B.get(h)
        for b, dim in MEM_VAL_BY_B.items():
            if b == hot_b:
                continue
            assert attn.W_k.data[h * _HD + 3, dim] == 0.0, (
                f"Head {h} K[3, MEM_VAL_B{b}] should be 0 after suppress; "
                f"got {float(attn.W_k.data[h * _HD + 3, dim])}"
            )


def test_l15_lookup_head0_byte_selection_targets_mem_val_byte0(l15_lookup_attn):
    """Head 0 byte selection targets MEM val byte 0 (d=5 from MEM marker).

    The byte-0 K position differs from heads 1-3: rather than keying on
    a dedicated MEM_VAL_B0 dim (which exists but is reserved for the
    legacy LEV-specific head 4), it keys on the threshold pair
    (L2H0[MEM]=1, H1[MEM]=0) that fires only at d=5 from MEM. This test
    pins that pair.
    """
    attn = l15_lookup_attn
    base = 0
    assert attn.W_k.data[base + 3, _SetDim.L2H0 + _MEM_I] == 60.0, (
        "Head 0 K[3, L2H0+MEM_I] should be +60 (val byte 0 threshold)"
    )
    assert attn.W_k.data[base + 3, _SetDim.H1 + _MEM_I] == -60.0, (
        "Head 0 K[3, H1+MEM_I] should be -60 (suppress d<=4 from MEM)"
    )
    # Head 0 also gates Q[3] on MARK_STACK0 (the STACK0 marker query).
    assert attn.W_q.data[base + 3, _SetDim.MARK_STACK0] == 60.0
    # And on MARK_AX for the LI/LC byte-0 query.
    assert attn.W_q.data[base + 3, _SetDim.MARK_AX] == 60.0
