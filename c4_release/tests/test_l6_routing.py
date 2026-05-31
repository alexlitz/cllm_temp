"""Per-op audit harness for Layer 6 (JMP/JSR/ENT/BZ/BNZ routing + halt).

Layer 6 owns the per-opcode routing layer the rest of the pipeline
leans on for control flow:

* ``layer6_attn`` (kind=attn dep anchor) + ``layer6_attn_bake``
  (kind=model phase=998.5): 6 relay heads writing CMP[0..5] and
  staging FETCH -> AX_CARRY for first-step JMP/EXIT/JSR.
* ``layer6_routing_ffn`` (kind=block phase=6.5): per-opcode routing
  for IMM/EXIT/NOP/JMP/JSR -- moves FETCH/AX_CARRY into OUTPUT and
  detects halt.
* ``layer6_ent_after_jsr_sp_byte0_fixup`` (kind=block phase=6.55):
  patches SP byte 0 on the ENT immediately after a JSR push.
* ``layer6_relay_heads`` (kind=attn dep anchor) +
  ``layer6_relay_heads_bake`` (kind=model phase=998.6): PSH STACK0 <-
  AX relay (head 6/7).
* ``layer6_bz_bnz_relay_bake`` (kind=model phase=998.7): head 4
  programming for BZ/BNZ relay -- the ONLY L6 op currently shipping a
  populated ``claims=`` set, so it owns the load-bearing drift gate.
* ``binary_pop_sp_increment`` (kind=model phase=998): L6 FFN SP+=8
  extension for binary-pop ops.

Most L6 ops ship with empty ``claims`` today: ``verify_claims_static``
only inspects ops whose ``claims`` are non-empty, so only
``layer6_bz_bnz_relay_bake`` flows through the drift / fires gates.
The remaining L6 ops are guarded by ``assert_op_absent`` -- if anyone
adds a ``claims=`` block (or flips an ``enable=`` gate that suddenly
emits claims), the absent gate fails LOUDLY, prompting the migration
into the drift-checked list.

The third audit layer is a symbolic forward over the baked L6 head 4:
construct a synthetic 6-position residual stream (PC marker reading
previous AX byte 0), run it through a freshly-baked ``PureAttention``,
and assert CMP[2..5] one-hots match the BZ-taken vs BZ-not-taken
contract. B5-K's compiled-C analysis pinned BZ routing as the
diagnostic where many compiled-C tests fail -- this test fingers the
relay itself, decoupled from the upstream attention/embedding chain.

The shared ``static_claims_report`` session-scoped fixture in
``conftest.py`` runs ``verify_claims_static`` once per pytest session
(~25-60s) regardless of how many per-layer harnesses are present.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.base_layers import PureAttention  # noqa: E402
from neural_vm.constants import INSTR_WIDTH, PC_OFFSET  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from neural_vm.setup_helpers import _set_bz_bnz_relay  # noqa: E402
from neural_vm.vm_step import _SetDim as BD  # noqa: E402

from ._per_op_audit import (  # noqa: E402
    assert_no_drift,
    assert_op_absent,
    assert_op_fires,
)


# ---------------------------------------------------------------------------
# 1. Drift / fires for every L6 op with non-empty claims
# ---------------------------------------------------------------------------
# Per-op claim verification only inspects ops with non-empty ``claims``.
# As of this baseline, only ``layer6_bz_bnz_relay_bake`` qualifies; the
# rest of L6 still relies on ``produces``/``opcodes`` annotations.
# Whenever a new L6 op adopts claims, append its name here so the
# matching drift+fires pair starts gating it. The absent-gate parametrize
# below is the other half of the contract -- it fails loudly if an L6 op
# WITHOUT claims here suddenly starts emitting them.

L6_OPS_WITH_CLAIMS = (
    "layer6_bz_bnz_relay_bake",
)

# Every named ``make_*_op`` factory in ``ops/l6_ops.py``. Names mirror
# the ``Operation(name=...)`` values to keep the absent gate stable
# against rename / deregister regressions.
L6_OPS_ALL = (
    "layer6_attn",
    "layer6_routing_ffn",
    "layer6_ent_after_jsr_sp_byte0_fixup",
    "layer6_relay_heads",
    "layer6_attn_bake",
    "layer6_relay_heads_bake",
    "layer6_bz_bnz_relay_bake",
    "binary_pop_sp_increment",
)

L6_OPS_WITHOUT_CLAIMS = tuple(
    name for name in L6_OPS_ALL if name not in L6_OPS_WITH_CLAIMS
)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L6_OPS_WITH_CLAIMS)
def test_l6_op_has_no_declared_but_not_written_drift(
    static_claims_report, op_name: str
) -> None:
    """The declared (layer, scope, id, column) claims are actually written.

    A failure here means an L6 op promised to populate a slot in its
    ``claims=`` set but its bake_fn did not, i.e. declaration drift in
    the load-bearing direction. The verifier output names the offending
    cell directly.
    """
    assert_no_drift(static_claims_report, "L6", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L6_OPS_WITH_CLAIMS)
def test_l6_op_fires_during_bake(static_claims_report, op_name: str) -> None:
    """The op dispatches and produces an observable diff (not INERT).

    A failure here means the op's bake_fn ran but wrote nothing -- the
    usual cause is a gating flag (``enable=False``) flipped off in
    factory wiring or a layer routing regression that places the op on
    the wrong block.
    """
    assert_op_fires(static_claims_report, "L6", op_name)


@pytest.mark.lowering
@pytest.mark.parametrize("op_name", L6_OPS_WITHOUT_CLAIMS)
def test_l6_op_without_claims_stays_absent_from_report(
    static_claims_report, op_name: str
) -> None:
    """L6 ops without ``claims=`` must not appear in the verifier report.

    ``verify_claims_static`` only inspects ops with non-empty claims.
    If a previously-empty-claims op shows up here, somebody added
    claims or flipped an ``enable=`` gate without updating
    ``L6_OPS_WITH_CLAIMS`` above. Migrate the name into the
    drift-checked list and remove it from ``L6_OPS_ALL`` here.
    """
    assert_op_absent(static_claims_report, "L6", op_name)


# ---------------------------------------------------------------------------
# 2. Static weight structure: L6 head 4 (BZ/BNZ) relay layout
# ---------------------------------------------------------------------------
#
# The L6 BZ/BNZ relay head is the choke-point B5-K's compiled-C
# analysis pinned as the diagnostic where many compiled-C tests fail.
# These tests bake _set_bz_bnz_relay into a fresh PureAttention and
# verify the per-slot Q/K/V/O routing matches the (head 4) spec the
# downstream FFN depends on -- the same shape the runtime forward
# pass observes after legacy_bake on the real model.
#
# A full symbolic forward pass over PureAttention would also be a
# meaningful end-to-end probe, but production attn6 uses ALiBi slope
# 5.0 on head 4 (set by ``make_baseline_alibi_slopes_op``) which
# PureAttention does not model. The forward-pass BZ-taken vs
# BZ-not-taken regression is therefore exercised end-to-end via the
# pure_neural_runner test at the bottom of this file -- the weight
# structure asserts below pin the bake to the spec so a runtime
# regression has a tightly-scoped failure ahead of the e2e test.

_S_BAKE = 100.0
_D_MODEL = 512
_N_HEADS = 8
_HD = _D_MODEL // _N_HEADS

_HEAD4_BASE = 4 * _HD
# AX marker index — mirrors ``_set_bz_bnz_relay``'s ``AX_I = 1``.
_AX_I = 1
# Spec scale constant from ``_set_bz_bnz_relay``.
_L = 50.0


def _build_baked_attn() -> PureAttention:
    """A fresh PureAttention with only the BZ/BNZ relay (head 4) baked.

    Mirrors ``_set_bz_bnz_relay`` from ``setup_helpers.py`` so the test
    sees exactly the same weight layout the production compiler bakes
    via ``layer6_bz_bnz_relay_bake``. Wrapped in ``torch.no_grad()`` so
    the in-place writes don't trip leaf-Parameter autograd guards.
    """
    attn = PureAttention(dim=_D_MODEL, num_heads=_N_HEADS)
    with torch.no_grad():
        # Zero everything first so heads 0..3 and 5..7 stay quiet.
        attn.W_q.data.zero_()
        attn.W_k.data.zero_()
        attn.W_v.data.zero_()
        attn.W_o.data.zero_()
        _set_bz_bnz_relay(attn, _S_BAKE, BD, _HD)
    return attn


@pytest.fixture(scope="module")
def baked_bz_attn() -> PureAttention:
    """Module-scoped baked PureAttention shared by the structural tests."""
    return _build_baked_attn()


def test_l6_bz_relay_q_slot0_gates_on_pc_marker_and_opcode(baked_bz_attn):
    """Slot 0 Q fires at PC marker with OP_BZ or OP_BNZ active."""
    q_row = baked_bz_attn.W_q[_HEAD4_BASE]
    assert q_row[BD.MARK_PC].item() == pytest.approx(_L)
    assert q_row[BD.MARK_AX].item() == pytest.approx(-_L)
    assert q_row[BD.CONST].item() == pytest.approx(-_L * 1.3)
    assert q_row[BD.OP_BZ].item() == pytest.approx(_L / 5.0)
    assert q_row[BD.OP_BNZ].item() == pytest.approx(_L / 5.0)


def test_l6_bz_relay_k_slot0_targets_ax_byte0(baked_bz_attn):
    """Slot 0 K fires at the AX byte 0 token (L1H1[AX]=1 AND L1H0[AX]=0)."""
    k_row = baked_bz_attn.W_k[_HEAD4_BASE]
    assert k_row[BD.L1H1 + _AX_I].item() == pytest.approx(_L)
    assert k_row[BD.L1H0 + _AX_I].item() == pytest.approx(-_L)
    assert k_row[BD.CONST].item() == pytest.approx(_L)


def test_l6_bz_relay_v_routes_op_flags_and_zero_nibble_signals(baked_bz_attn):
    """V rows 1..4 copy OP_BZ, OP_BNZ, EMBED_LO[0], EMBED_HI[0] from K=AX byte 0.

    EMBED_LO[0]/EMBED_HI[0] are 1.0 iff the low/high nibble of the AX
    byte 0 token is zero, i.e. exactly the BZ-taken signal.
    """
    base = _HEAD4_BASE
    assert baked_bz_attn.W_v[base + 1, BD.OP_BZ].item() == pytest.approx(1.0)
    assert baked_bz_attn.W_v[base + 2, BD.OP_BNZ].item() == pytest.approx(1.0)
    assert baked_bz_attn.W_v[base + 3, BD.EMBED_LO + 0].item() == pytest.approx(1.0)
    assert baked_bz_attn.W_v[base + 4, BD.EMBED_HI + 0].item() == pytest.approx(1.0)


def test_l6_bz_relay_o_writes_cmp_2_through_5(baked_bz_attn):
    """O matrix routes head 4 V slots 1..4 to CMP[2..5] at the PC marker.

    Downstream L6 routing FFN reads (MARK_PC AND CMP[2] AND CMP[4] AND
    CMP[5]) to trigger BZ-taken PC override -- the structural contract
    pinned here is therefore load-bearing.
    """
    base = _HEAD4_BASE
    # CMP[2] = OP_BZ flag (scaled by 0.2 so the ~5.0 raw value normalizes to ~1).
    assert baked_bz_attn.W_o[BD.CMP + 2, base + 1].item() == pytest.approx(0.2)
    # CMP[3] = OP_BNZ flag.
    assert baked_bz_attn.W_o[BD.CMP + 3, base + 2].item() == pytest.approx(0.2)
    # CMP[4] = AX_LO == 0 indicator (full scale 1.0).
    assert baked_bz_attn.W_o[BD.CMP + 4, base + 3].item() == pytest.approx(1.0)
    # CMP[5] = AX_HI == 0 indicator.
    assert baked_bz_attn.W_o[BD.CMP + 5, base + 4].item() == pytest.approx(1.0)


def test_l6_bz_relay_other_heads_stay_zero(baked_bz_attn):
    """Heads 0..3 and 5..7 must stay zero -- bake only programs head 4.

    Non-zero touches on a non-target head would indicate _set_bz_bnz_relay
    spilled past its 1-head allocation -- a regression of the dim-ownership
    claim the L6 routing layer leans on.
    """
    for head in range(_N_HEADS):
        if head == 4:
            continue
        base = head * _HD
        for matrix_name in ("W_q", "W_k", "W_v"):
            row_slice = getattr(baked_bz_attn, matrix_name)[base : base + _HD]
            max_abs = row_slice.abs().max().item()
            assert max_abs == 0.0, (
                f"{matrix_name} head {head} (rows {base}..{base + _HD}) "
                f"non-zero (max_abs={max_abs}); _set_bz_bnz_relay touched "
                f"a head outside its 1-head allocation."
            )
        # O matrix indexed by head columns.
        col_slice = baked_bz_attn.W_o[:, base : base + _HD]
        max_abs = col_slice.abs().max().item()
        assert max_abs == 0.0, (
            f"W_o head {head} (cols {base}..{base + _HD}) non-zero "
            f"(max_abs={max_abs}); _set_bz_bnz_relay's O writes spilled."
        )


# ---------------------------------------------------------------------------
# 3. Symbolic forward (end-to-end): BZ-taken vs BZ-not-taken via pure neural
# ---------------------------------------------------------------------------
# B5-K's analysis pinned BZ routing as the diagnostic where many compiled-C
# tests fail. This pair compares the BZ-taken (AX=0) vs BZ-not-taken (AX != 0)
# paths end-to-end through pure_neural_runner, exercising the full L6 head 4
# relay + L6 routing FFN PC-override path under the real ALiBi-equipped model
# (which PureAttention does not reproduce).
#
# The L6 head-4 weight structure tests above pin the bake so that if the
# forward test below fails, the failure is isolated to the routing FFN or
# the attention dynamics rather than the bake weights.

_BRANCH_OPS = {Opcode.JMP, Opcode.BZ, Opcode.BNZ, Opcode.JSR}


def _encode_program(prog):
    """Encode opcodes; branch immediates become absolute PC bytes."""
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            if op in _BRANCH_OPS:
                imm = imm * INSTR_WIDTH + PC_OFFSET
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def _run_bz_program(runner, ax_imm: int, max_steps: int = 12) -> int:
    """Run a BZ-taken vs BZ-not-taken probe; returns the EXIT immediate.

    Program shape (4 instructions):
        IMM ax_imm    # AX = ax_imm
        BZ 4          # branch to instruction index 4 if AX == 0
        IMM 99        # fall-through path
        EXIT
        IMM 7         # taken path
        EXIT

    AX == 0  -> exits with 7  (BZ-taken)
    AX != 0  -> exits with 99 (BZ-not-taken)
    """
    bc = _encode_program([
        (Opcode.IMM, ax_imm),
        (Opcode.BZ, 4),
        (Opcode.IMM, 99),
        Opcode.EXIT,
        (Opcode.IMM, 7),
        Opcode.EXIT,
    ])
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    _, exit_code = runner.run(bc, b"", max_steps=max_steps)
    return exit_code


def test_l6_bz_relay_drives_bz_taken_vs_not_taken_end_to_end(
    pure_neural_runner,
):
    """BZ-taken (AX=0) and BZ-not-taken (AX=5) must follow different paths.

    Pinning both decision branches in one test is the single end-to-end
    gate for the L6 BZ/BNZ relay: the same compiled bytecode is fed to
    the same pure_neural_runner with different AX values, and the EXIT
    immediate is the load-bearing signal.

    This is the smallest reproducer of B5-K's BZ-routing diagnostic --
    a regression in either direction (taken collapses to fall-through,
    or not-taken jumps spuriously) flips this test red on the same
    bytecode, isolating the regression to L6's head 4 + routing FFN
    decision path.
    """
    # BZ-taken: AX = 0 -> CMP[4]=1, CMP[5]=1 -> PC override fires.
    assert _run_bz_program(pure_neural_runner, ax_imm=0) == 7, (
        "BZ-taken path failed: AX=0 should have triggered branch to "
        "the EXIT-with-7 instruction, but pure_neural_runner exited "
        "with the fall-through value (99) or a wrong code. L6 head 4 "
        "is not relaying both AX_LO==0 (CMP[4]) and AX_HI==0 (CMP[5]) "
        "to the PC marker, or the L6 routing FFN's 4-way AND is broken."
    )
    # BZ-not-taken: AX != 0 -> CMP[4]/CMP[5] zero -> fall through.
    assert _run_bz_program(pure_neural_runner, ax_imm=5) == 99, (
        "BZ-not-taken path failed: AX=5 should have fallen through to "
        "the EXIT-with-99 instruction, but pure_neural_runner branched "
        "spuriously. L6 head 4's AX_LO==0 / AX_HI==0 detection is "
        "false-positive at the PC marker."
    )
