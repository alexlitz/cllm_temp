"""Per-op audit harness for ``l10_post_ops_combined`` (the L17 FFN).

Despite the name, ``l10_post_ops_combined`` is dependency-anchored at L17 as
``block.ffn`` (B6-L inventory § 1a/1b). The factory at
``c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1289`` bakes 4
post-op modules sequentially into a single PureFFN at phase=10.5:

  1. ``BinaryOpByteZeroingPostOp`` (~8 units) — zeros AX bytes 1-3 for
     byte-0-only opcodes (EQ/NE/LT/GT/LE/GE/SHL/SHR/MUL/DIV/MOD).
  2. 3x ``CarryPropagationPostOp`` (~1536 units) — ZEROED post-bake
     (``l10_ops.py:1334-1338``). Their slot is reserved for stable
     indexing but produces no contribution.
  3. ``ComparisonCombine`` (~302 units) — combines CMP[0..3] into truth
     bytes for EQ/NE/LT/GT/LE/GE.

Post-bake guards then write strong ``-S * 10**k`` blockers on most opcodes
(ADD/SUB/MUL/SHL/SHR/IMM/JMP/LI_RELAY/LC_RELAY) and on TEMP+8/9/10,
CARRY+1/2/3, CMP+7, H1+0 — so the only opcodes the combined FFN actually
emits at all are the comparison family (EQ/NE/LT/GT/LE/GE).

This harness covers what ``test_l10_post_op_attach.py`` does not:

1. **Static drift check**: pin ``l10_post_ops_combined`` against
   ``static_claims_report``. The op carries no per-cell claims today, but
   if a future PR adds them the verifier upgrades this test from
   ``assert_op_absent`` to ``assert_no_drift`` automatically (via the
   ``L10_POST_OPS_COMBINED_OPS_WITH_CLAIMS`` list).
2. **Carry-slice zero invariant**: confirm the 3x ``CarryPropagationPostOp``
   slice is fully zeroed after bake. The bake hinges on ``if carry_end >
   carry_start:`` and is silently broken if anyone reorders the sequence
   so the carry slice is empty or misaligned.
3. **Comparison forward**: drive a synthetic CMP-flag residual through the
   baked FFN and confirm the comparison-byte OUTPUT writes match the
   spec table (EQ true, NE false, etc.).
4. **Suppressed-opcode guard**: confirm OP_ADD/OP_MUL/OP_SHL inputs are
   strongly blocked — the FFN must not emit OUTPUT bytes at those
   opcodes (those bytes are owned by the upstream structural ALUs).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.base_layers import PureFFN  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler import (  # noqa: E402
    declare_setdim_compat_dims,
)
from neural_vm.unified_compiler.layer_compiler import LayerCompiler  # noqa: E402
from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops  # noqa: E402
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    make_l10_post_ops_combined,
)

from ._per_op_audit import assert_no_drift, assert_op_absent, assert_op_fires  # noqa: E402


# ---------------------------------------------------------------------------
# Op inventory: l10_post_ops_combined ships without per-cell claims today.
# ---------------------------------------------------------------------------

L10_POST_OPS_COMBINED_OPS_WITH_CLAIMS: tuple[str, ...] = ()

L10_POST_OPS_COMBINED_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD = (
    "l10_post_ops_combined",
)


# ---------------------------------------------------------------------------
# Shared layout / bake fixtures (module-scoped — the bake fills ~1846 units).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compact_layout():
    """Build the production compact layout once per test module."""
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        compiler.add_op(op)
    return compiler.compile()


@pytest.fixture(scope="module")
def baked_combined_ffn(compact_layout):
    """PureFFN baked via ``make_l10_post_ops_combined`` only (module-scoped)."""
    layout = compact_layout
    with torch.no_grad():
        ffn = PureFFN(dim=layout.d_model, hidden_dim=2048)
        # Zero baseline so post-bake invariants are easy to test.
        ffn.W_up.data.zero_()
        ffn.b_up.data.zero_()
        ffn.W_gate.data.zero_()
        ffn.b_gate.data.zero_()
        ffn.W_down.data.zero_()
        make_l10_post_ops_combined().bake_fn(ffn, layout.dim_positions, 100.0)
    return ffn, layout


# ---------------------------------------------------------------------------
# 1. Static drift checks via ``static_claims_report``.
# ---------------------------------------------------------------------------


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L10_POST_OPS_COMBINED_OPS_WITH_CLAIMS)
def test_l10_post_ops_combined_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    assert_no_drift(static_claims_report, "L17", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L10_POST_OPS_COMBINED_OPS_WITH_CLAIMS)
def test_l10_post_ops_combined_op_fires_during_bake(
    static_claims_report, op_name: str
) -> None:
    assert_op_fires(static_claims_report, "L17", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize(
    "op_name", L10_POST_OPS_COMBINED_OPS_WITHOUT_CLAIMS_DEFAULT_BUILD
)
def test_l10_post_ops_combined_unclaimed_op_remains_absent_from_default_report(
    static_claims_report, op_name: str
) -> None:
    assert_op_absent(static_claims_report, "L17", op_name)


# ---------------------------------------------------------------------------
# 2. Carry-slice zero invariant.
# ---------------------------------------------------------------------------


def test_l10_post_ops_combined_zeroes_carry_slice(baked_combined_ffn):
    """The 3x ``CarryPropagationPostOp`` slice (1536 units) is zeroed.

    ``make_l10_post_ops_combined`` deliberately overwrites the carry slice
    after baking it (``l10_ops.py:1334-1338``) because the structural L10
    post-op pipeline already owns ADD/SUB carry propagation. If the
    ``if carry_end > carry_start:`` guard or the slice indexing drifts,
    the zeroing silently no-ops and the FFN double-applies carry
    propagation at L17 — exactly the runaway-amplification class B4-B
    chased.

    Test strategy: identify the carry slice by re-running the same
    sub-bakes against a probe FFN to compute (carry_start, carry_end),
    then assert the W_up/b_up/W_gate/b_gate/W_down rows in that slice are
    all zero in the production-baked FFN.
    """
    from neural_vm.unified_compiler.ops.shared import _bake_post_op_into
    from neural_vm.vm_step import (
        BinaryOpByteZeroingPostOp,
        CarryPropagationPostOp,
    )

    ffn, layout = baked_combined_ffn
    d_model = layout.d_model
    dim_positions = layout.dim_positions
    S = 100.0

    # Replicate the layout of the bake to compute the carry slice bounds.
    probe = PureFFN(dim=d_model, hidden_dim=2048)
    with torch.no_grad():
        probe.W_up.data.zero_()
        offset = 0
        offset = _bake_post_op_into(
            probe,
            BinaryOpByteZeroingPostOp(d_model, S, dim_positions=dim_positions),
            offset,
        )
        carry_start = offset
        for byte_idx, cascade in ((0, False), (1, True), (2, True)):
            offset = _bake_post_op_into(
                probe,
                CarryPropagationPostOp(
                    d_model, S, byte_idx=byte_idx, cascade=cascade,
                    dim_positions=dim_positions,
                ),
                offset,
            )
        carry_end = offset

    assert carry_end > carry_start, (
        "Carry slice has zero width — bake layout has drifted and the "
        "zero-out guard would silently skip; the L17 FFN may now be "
        "double-applying carry propagation."
    )

    # The load-bearing zero invariants are W_down (the output writes) and
    # the SwiGLU gate-side params (b_up, b_gate, W_gate). The W_up rows of
    # the carry slice get re-written downstream by the post-bake guards
    # (l10_ops.py:1347-1396 add ``-S * 10**k`` blockers across W_up for
    # ALL units up to ``offset``), so the carry-slice W_up rows are
    # intentionally non-zero in the final FFN. As long as the SwiGLU gate
    # params are zero, the units cannot produce a positive activation,
    # and W_down being zero means even a leaked activation cannot reach
    # any output cell.
    for tensor_name in ("b_up", "W_gate", "b_gate"):
        slice_data = getattr(ffn, tensor_name).data[carry_start:carry_end]
        assert slice_data.abs().max().item() == 0.0, (
            f"Carry slice [{carry_start}:{carry_end}] in {tensor_name} is "
            f"non-zero (max={slice_data.abs().max().item()}); the "
            f"post-bake zero-out at l10_ops.py:1334-1338 has regressed."
        )
    w_down_slice = ffn.W_down.data[:, carry_start:carry_end]
    assert w_down_slice.abs().max().item() == 0.0, (
        f"Carry slice [{carry_start}:{carry_end}] in W_down columns is "
        f"non-zero (max={w_down_slice.abs().max().item()}); the post-bake "
        f"zero-out has regressed. Even a strongly-firing W_up row can no "
        f"longer rewrite OUTPUT, because the gate (W_gate/b_gate/b_up) is "
        f"zero and W_down is zero."
    )


# ---------------------------------------------------------------------------
# 3. Comparison forward — ComparisonCombine slice fires for EQ/NE/etc.
# ---------------------------------------------------------------------------


def _cmp_input(
    *,
    op_name: str,
    cmp_flags: dict[int, float] | None,
    dim_positions: dict[str, int],
    d_model: int,
) -> torch.Tensor:
    """One-step residual at MARK_AX with the requested CMP flags lit."""
    x = torch.zeros(1, 1, d_model)
    x[0, 0, dim_positions["MARK_AX"]] = 1.0
    x[0, 0, dim_positions["CONST"]] = 1.0
    x[0, 0, dim_positions[op_name]] = 1.0
    cmp_base = dim_positions["CMP"]
    if cmp_flags is not None:
        for slot, value in cmp_flags.items():
            x[0, 0, cmp_base + slot] = value
    return x


@pytest.mark.parametrize(
    ("op_name", "cmp_flags", "expected_byte"),
    [
        # EQ default = 0; override fires only when CMP[1]=hi_eq AND CMP[2]=lo_eq.
        ("OP_EQ", None, 0),
        ("OP_EQ", {1: 1.0, 2: 1.0}, 1),
        # NE default = 1; override flips to 0 on hi_eq AND lo_eq.
        ("OP_NE", None, 1),
        ("OP_NE", {1: 1.0, 2: 1.0}, 0),
    ],
)
def test_l10_post_ops_combined_comparison_writes_byte(
    baked_combined_ffn, op_name, cmp_flags, expected_byte
):
    """ComparisonCombine slice baked into l10_post_ops_combined produces
    the canonical truth byte for EQ / NE at the AX marker.

    This regression-guards the only opcode family the combined FFN is
    expected to write OUTPUT for: the comparison ops. ADD/SUB/MUL/SHL
    rows are strongly blocked by the post-bake guards and are exercised
    by the suppression test below.
    """
    ffn, layout = baked_combined_ffn
    x = _cmp_input(
        op_name=op_name,
        cmp_flags=cmp_flags,
        dim_positions=layout.dim_positions,
        d_model=layout.d_model,
    )
    with torch.no_grad():
        y = ffn(x)
    delta = (y - x)[0, 0]

    out_lo_base = layout.dim_positions["OUTPUT_LO"]
    out_hi_base = layout.dim_positions["OUTPUT_HI"]
    lo_slice = delta[out_lo_base:out_lo_base + 16]
    hi_slice = delta[out_hi_base:out_hi_base + 16]
    lo_argmax = int(lo_slice.argmax().item())
    hi_argmax = int(hi_slice.argmax().item())

    assert lo_argmax == expected_byte, (
        f"{op_name} (cmp={cmp_flags}): OUTPUT_LO argmax should be "
        f"{expected_byte}, got {lo_argmax}. delta_LO={lo_slice.tolist()}"
    )
    assert hi_argmax == 0, (
        f"{op_name} OUTPUT_HI argmax should be 0 (comparison result is "
        f"single-byte), got {hi_argmax}. delta_HI={hi_slice.tolist()}"
    )


# ---------------------------------------------------------------------------
# 4. Suppression guards — ADD/MUL/SHL inputs must not produce OUTPUT writes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ["OP_ADD", "OP_MUL", "OP_SHL", "OP_IMM"])
def test_l10_post_ops_combined_suppresses_non_comparison_opcodes(
    baked_combined_ffn, op_name
):
    """The post-bake guards write ``-S * 1000`` blockers on every unit for
    ADD/SUB/MUL/SHL/SHR/IMM/JMP/LI_RELAY/LC_RELAY opcodes. Even with the
    full operand inputs lit, those opcodes must produce essentially zero
    OUTPUT writes (the comparison slice's default unit might fire weakly
    on the MARK_AX baseline but should never write more than negligible
    magnitude under the suppressed opcodes).

    Regression sentinel for the B6-L Section 4b finding: this FFN runs
    on ~1846 units; a single dropped blocker fans out across all of them.
    """
    ffn, layout = baked_combined_ffn
    # Provide non-trivial operand state so the suppression check is real
    # — if guards drop, the units fire on this state.
    x = torch.zeros(1, 1, layout.d_model)
    x[0, 0, layout.dim_positions["MARK_AX"]] = 1.0
    x[0, 0, layout.dim_positions["CONST"]] = 1.0
    x[0, 0, layout.dim_positions[op_name]] = 1.0
    x[0, 0, layout.dim_positions["ALU_LO"] + 5] = 1.0
    x[0, 0, layout.dim_positions["AX_CARRY_LO"] + 7] = 1.0
    x[0, 0, layout.dim_positions["CMP"] + 1] = 1.0
    x[0, 0, layout.dim_positions["CMP"] + 2] = 1.0

    with torch.no_grad():
        y = ffn(x)
    delta = (y - x)[0, 0]
    out_lo_base = layout.dim_positions["OUTPUT_LO"]
    out_hi_base = layout.dim_positions["OUTPUT_HI"]
    lo_slice = delta[out_lo_base:out_lo_base + 16]
    hi_slice = delta[out_hi_base:out_hi_base + 16]

    # Energy budget: with the opcode-level blocker the SwiGLU gate ought to
    # crush every unit to near-zero. Anything above ~1.0 means a guard
    # dropped.
    energy = float(lo_slice.abs().sum() + hi_slice.abs().sum())
    assert energy < 1.0, (
        f"{op_name}: post-bake suppression guard dropped — "
        f"OUTPUT band energy={energy:.4f} should be <1.0. The combined "
        f"L17 FFN may now be double-applying carry/binary logic on top "
        f"of the structural ALU's authoritative byte."
    )


# ---------------------------------------------------------------------------
# 5. Op-level metadata pin.
# ---------------------------------------------------------------------------


def test_l10_post_ops_combined_op_metadata_pinned():
    """Pin the load-bearing Operation metadata for ``l10_post_ops_combined``.

    The phase=10.5 + kind="ffn" + ffn_units_used=1846 combination is what
    routes this op to L17 in the layout (the dep-graph anchor that ends
    up materialised as block 30 after Phase 0 expansion). Any drift in
    these fields silently relocates the op.
    """
    op = make_l10_post_ops_combined()
    assert op.name == "l10_post_ops_combined"
    assert op.phase == 10.5
    assert op.kind == "ffn"
    assert op.ffn_units_used == 1846
    assert op.migrated is True
    assert op.declarative_authority == "declarative"
    assert "OUTPUT_LO" in op.writes
    assert "OUTPUT_HI_THIS_STEP" in op.writes
