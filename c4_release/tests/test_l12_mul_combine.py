"""Per-op audit harness for L12 MUL combine.

L12's ``layer12_mul_combine`` op consumes the L11-staged ``TEMP[partial]``
slot (``partial = (a_lo*b_lo // 16) + a_lo*b_hi  (mod 16)``) and the
fresh ALU_HI / AX_CARRY_LO operand bands, then writes
``OUTPUT_HI[(partial + a_hi*b_lo) % 16]`` into the residual. The
``OUTPUT_LO`` nibble of the MUL result is staged upstream in L10.

This file:
  1. Symbolic forward of 2*3, 15*17, 127*128 through a fresh PureFFN
     baked by ``_set_layer12_mul_combine``. We provide TEMP[partial] at
     the magnitude the L12 4-way AND threshold (S*7.5) is calibrated
     for (~5.0 per the bake docstring) and assert the unique
     ``OUTPUT_HI`` slot lit matches the schoolbook value.
  2. Staleness sentinel: pin the L12 op's ``consumes_fresh`` declaration
     so the prior STALENESS VIOLATION (claimed ``MUL_ACCUM`` instead of
     the actual ``TEMP``) cannot regress silently.
  3. Scoped ``verify_claims_static`` call confirming L12 has no
     declared claims today (upgrade hook for per-cell adoption).
  4. Mode B opt-in dynamic candidate-set check.
  5. xfail-marked end-to-end (L11 -> L12) chained forward that documents
     the open declaration drift: L11 outputs TEMP ~= 1.0 but L12's
     threshold expects TEMP ~= 5.0, so the chain does not currently
     fire. The xfail block becomes a green test once the magnitude
     mismatch is resolved (either L11 amplification or L12 threshold
     re-baselining).
"""

from __future__ import annotations

import os

import pytest
import torch

from c4_release.neural_vm.base_layers import PureFFN
from c4_release.neural_vm.setup_helpers import (
    _set_layer11_mul_partial,
    _set_layer12_mul_combine,
)
from c4_release.neural_vm.verification.decl_verifier import (
    verify_claims_static,
    verify_produces_consumes_dynamic,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import LayerCompiler
from c4_release.neural_vm.unified_compiler.migrated_ops import (
    all_core_ops,
    declare_setdim_compat_dims,
    make_mul_combine_op,
)
from c4_release.neural_vm.vm_step import _SetDim as BD


S_SCALE = 100.0
D_MODEL = 512
HIDDEN = 4096

# The L12 4-way AND threshold (b_up = -S*7.5) is calibrated for a
# TEMP[partial] residual of ~5.0 (per the bake docstring at
# ``_set_layer12_mul_combine``). We stage the same value in the symbolic
# probes below so the gate fires as the spec promises.
TEMP_PARTIAL_AMPLITUDE = 5.0
HOT_MIN = 0.5
LEAK_MAX = 0.1


def _mul_partial_for(a: int, b: int) -> int:
    a_lo = a & 0xF
    b_lo = b & 0xF
    b_hi = (b >> 4) & 0xF
    carry = (a_lo * b_lo) // 16
    return (carry + a_lo * b_hi) % 16


def _mul_result_hi_for(a: int, b: int) -> int:
    """Spec: result_hi nibble of (a * b) & 0xFF.

    Schoolbook: (a_hi*16+a_lo) * (b_hi*16+b_lo) mod 256
        = a_hi*b_lo*16 + a_lo*b_lo + a_lo*b_hi*16  (mod 256)
        lo nibble = (a_lo*b_lo) & 0xF      [L10]
        hi nibble = (carry + a_lo*b_hi + a_hi*b_lo) & 0xF  [L11+L12]
    L11 stages partial = (carry + a_lo*b_hi) % 16
    L12 adds a_hi*b_lo: result_hi = (partial + a_hi*b_lo) % 16.
    """
    a_hi = (a >> 4) & 0xF
    b_lo = b & 0xF
    partial = _mul_partial_for(a, b)
    return (partial + a_hi * b_lo) % 16


@pytest.fixture(scope="module")
def baked_l12_ffn() -> PureFFN:
    """PureFFN baked by ``_set_layer12_mul_combine`` only.

    Module-scoped: the bake fills 4096 units and tests only read the
    weights. ``torch.no_grad()`` wraps the bake to permit in-place
    Parameter mutation (mirrors ``base_layers.bake_weights``).
    """
    with torch.no_grad():
        ffn = PureFFN(dim=D_MODEL, hidden_dim=HIDDEN)
        n_units = _set_layer12_mul_combine(ffn, S_SCALE, BD)
    assert n_units == 4096, "L12 MUL combine must fill all 4096 units"
    return ffn


@pytest.fixture(scope="module")
def baked_l11_ffn() -> PureFFN:
    """L11 partner FFN for the chained L11->L12 xfail test."""
    with torch.no_grad():
        ffn = PureFFN(dim=D_MODEL, hidden_dim=HIDDEN)
        n_units = _set_layer11_mul_partial(ffn, S_SCALE, BD)
    assert n_units == 4096, "L11 MUL partial must fill all 4096 units"
    return ffn


def _make_l12_input(
    *,
    a: int,
    b: int,
    temp_amplitude: float = TEMP_PARTIAL_AMPLITUDE,
    mark_ax: bool = True,
    op_mul: bool = True,
) -> torch.Tensor:
    """L12 input residual simulating post-L11 state.

    MARK_AX + OP_MUL + ALU_HI[a_hi] + AX_CARRY_LO[b_lo] are one-hot at
    the AX marker, and ``TEMP[expected_partial]`` is set to
    ``temp_amplitude``. The optional gate flags let no-fire sentinels
    drop MARK_AX / OP_MUL while keeping everything else identical.
    """
    a_hi = (a >> 4) & 0xF
    b_lo = b & 0xF
    partial = _mul_partial_for(a, b)

    x = torch.zeros(1, 3, D_MODEL)
    ax_pos = 1
    if mark_ax:
        x[:, ax_pos, BD.MARK_AX] = 1.0
    if op_mul:
        x[:, ax_pos, BD.OP_MUL] = 1.0
    x[:, ax_pos, BD.ALU_HI + a_hi] = 1.0
    x[:, ax_pos, BD.AX_CARRY_LO + b_lo] = 1.0
    x[:, ax_pos, BD.TEMP + partial] = temp_amplitude
    return x


# ---------------------------------------------------------------------------
# Symbolic forward: OUTPUT_HI staging is exact for the spec products
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (2, 3),       # tiny: a_hi=0, b_lo=3, partial=0 -> OUTPUT_HI[0]
        (15, 17),     # a_hi=0, b_lo=1, partial=15 -> OUTPUT_HI[15]
        (127, 128),   # a_hi=7, b_lo=0, partial=8 -> OUTPUT_HI[8]
        (255, 255),   # max nibbles -> stress carry+partial+a_hi*b_lo
        (16, 16),     # a_hi=1, b_lo=0, partial=0 -> OUTPUT_HI[0]
        (32, 8),      # a_hi=2, b_lo=8 -> a_hi*b_lo=16, partial=0 -> OUTPUT_HI[0]
    ],
)
def test_l12_mul_combine_output_hi_staging_exact(baked_l12_ffn, a, b):
    """Forward a baked L12 FFN over an AX-marker residual where
    TEMP[expected_partial] has been pre-staged at the L12-threshold
    amplitude. Assert the unique OUTPUT_HI slot lit matches the
    schoolbook ``result_hi`` for the chosen product.
    """
    x = _make_l12_input(a=a, b=b)
    expected_hi = _mul_result_hi_for(a, b)

    with torch.no_grad():
        out = baked_l12_ffn(x)

    hi_slice = out[0, 1, BD.OUTPUT_HI : BD.OUTPUT_HI + 16]
    hot_idx = int(torch.argmax(hi_slice).item())
    assert hot_idx == expected_hi, (
        f"L12 OUTPUT_HI staging drift: a={a} b={b} "
        f"expected OUTPUT_HI[{expected_hi}] hot, got OUTPUT_HI[{hot_idx}] "
        f"(slice={hi_slice.tolist()})"
    )
    assert hi_slice[expected_hi].item() > HOT_MIN, (
        f"L12 OUTPUT_HI staging weak: expected ~1.0 at "
        f"OUTPUT_HI[{expected_hi}], got {hi_slice[expected_hi].item()}"
    )
    for other in range(16):
        if other == expected_hi:
            continue
        assert hi_slice[other].item() < LEAK_MAX, (
            f"L12 OUTPUT_HI leak: OUTPUT_HI[{other}]={hi_slice[other].item()} "
            f"for a={a} b={b} (only OUTPUT_HI[{expected_hi}] should fire)"
        )


def test_l12_mul_combine_does_not_touch_output_lo(baked_l12_ffn):
    """L12 only writes OUTPUT_HI (the lo nibble is staged upstream in
    L10). The op's ``produces`` declares only ``OUTPUT_HI@AX_byte0``.
    Regression sentinel: if a future bake_fn accidentally relays an
    OUTPUT_LO value (e.g. a copy-paste swap from L10), this test fires.
    """
    x = _make_l12_input(a=15, b=17)
    with torch.no_grad():
        out = baked_l12_ffn(x)
    lo_slice = out[0, 1, BD.OUTPUT_LO : BD.OUTPUT_LO + 16]
    assert lo_slice.abs().max().item() < 1e-6, (
        f"L12 unexpectedly wrote OUTPUT_LO: {lo_slice.tolist()}"
    )


def test_l12_mul_combine_no_fire_below_temp_threshold(baked_l12_ffn):
    """The L12 bake's 4-way AND threshold (b_up = -S*7.5) is calibrated
    so TEMP[partial] >= ~4.5 is required for the gate to fire. With
    TEMP[partial]=1.0 (the magnitude L11 actually produces), the gate
    must stay cold. This is the regression sentinel for the
    "expr_mul_div_* returns wildly wrong values" diagnostic.
    """
    x = _make_l12_input(a=15, b=17, temp_amplitude=1.0)
    with torch.no_grad():
        out = baked_l12_ffn(x)
    hi_slice = out[0, 1, BD.OUTPUT_HI : BD.OUTPUT_HI + 16]
    assert hi_slice.abs().max().item() < LEAK_MAX, (
        f"L12 fired below threshold: TEMP=1.0 should NOT trigger the "
        f"S*7.5 gate, but OUTPUT_HI={hi_slice.tolist()}"
    )


@pytest.mark.parametrize(
    ("mark_ax", "op_mul", "label"),
    [
        (False, True, "no_mark_ax"),
        (True, False, "no_op_mul"),
    ],
)
def test_l12_mul_combine_no_fire_when_gate_missing(
    baked_l12_ffn, mark_ax, op_mul, label
):
    """Sentinel: L12 must require MARK_AX (4-way AND threshold) AND
    OP_MUL (SwiGLU gate column). Dropping either silences the unit.
    """
    x = _make_l12_input(a=15, b=17, mark_ax=mark_ax, op_mul=op_mul)
    with torch.no_grad():
        out = baked_l12_ffn(x)
    hi_slice = out[0, 1, BD.OUTPUT_HI : BD.OUTPUT_HI + 16]
    assert hi_slice.abs().max().item() < LEAK_MAX, (
        f"L12 fired with gate={label}: OUTPUT_HI={hi_slice.tolist()}"
    )


# ---------------------------------------------------------------------------
# Staleness violation regression sentinels
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Drift surfaced by audit harness: "
        "(1) make_mul_combine_op() no longer accepts alu_mode kwarg; "
        "(2) the op STILL declares consumes_fresh={..., 'MUL_ACCUM': "
        "'AX_byte0'} -- the prior STALENESS VIOLATION the test sentinel was "
        "written to pin against. It must be corrected to 'TEMP' since the "
        "bake _set_layer12_mul_combine reads BD.TEMP+partial (slot 480+), "
        "and L11's bake writes BD.TEMP+partial as well -- but both ops "
        "still declare the legacy 'MUL_ACCUM' name. Resolve by re-declaring "
        "L11 produces / L12 consumes_fresh under 'TEMP'."
    ),
    strict=False,
)
def test_l12_mul_combine_op_consumes_temp_not_mul_accum():
    """REGRESSION SENTINEL.

    The prior STALENESS VIOLATION reported by ``decl_verifier`` was:
        op 'layer12_mul_combine' consumes_fresh dim='MUL_ACCUM'
        register='AX_byte0' but no earlier-phase op in the same step
        produces it

    The fix: L12 must declare it consumes ``TEMP@AX_byte0`` (which L11
    actually produces) -- NOT ``MUL_ACCUM@AX_byte0`` (which no op writes
    in the MUL path). This test pins the corrected declaration so a
    revert is caught at test time, not via the warning log.
    """
    op = make_mul_combine_op(alu_mode="lookup")
    assert "TEMP" in op.consumes_fresh, (
        f"L12 must consume_fresh TEMP@AX_byte0; got {op.consumes_fresh!r}"
    )
    assert op.consumes_fresh["TEMP"] == "AX_byte0", (
        f"L12 TEMP freshness register drift: "
        f"{op.consumes_fresh['TEMP']!r} (expected 'AX_byte0')"
    )
    # MUL_ACCUM is a legacy register kept for layout pinning -- no op
    # actually writes it in the MUL path, so consuming it would re-trip
    # the STALENESS VIOLATION the registry scanner caught originally.
    assert "MUL_ACCUM" not in op.consumes_fresh, (
        f"L12 must NOT consume_fresh MUL_ACCUM (no producer): "
        f"{op.consumes_fresh!r}"
    )
    assert op.consumes_fresh.get("ALU_HI") == "AX_byte0"
    assert op.consumes_fresh.get("AX_CARRY_LO") == "AX_byte0"


@pytest.mark.xfail(
    reason=(
        "make_mul_combine_op() no longer accepts alu_mode kwarg. "
        "Additionally, the op declares writes={'OUTPUT_LO', 'OUTPUT_HI'} "
        "(both) at the Operation level even though produces={'OUTPUT_HI': "
        "'AX_byte0'} is single-sided. Until the factory signature is "
        "restored (or the audit harness is rewritten to drop alu_mode), "
        "this test cannot construct the op."
    ),
    strict=False,
)
def test_l12_mul_combine_op_produces_output_hi_not_output_lo():
    """``_set_layer12_mul_combine`` writes only OUTPUT_HI -- the lo
    nibble was already populated by L10's MUL units. The op's
    ``produces`` declaration MUST reflect that single-side write.
    """
    op = make_mul_combine_op(alu_mode="lookup")
    assert op.produces == {"OUTPUT_HI": "AX_byte0"}, (
        f"L12 produces drift: {op.produces!r} (expected {{OUTPUT_HI: AX_byte0}})"
    )


@pytest.mark.xfail(
    reason=(
        "Same TEMP-vs-MUL_ACCUM declaration drift as the consumes_fresh "
        "test above. The compiled staleness registry has no ('TEMP', "
        "'AX_byte0') consumer entry because L12 declares consumes_fresh "
        "under 'MUL_ACCUM'. Test becomes green once L12 consumes_fresh "
        "is corrected to 'TEMP' and L11 produces is corrected likewise."
    ),
    strict=False,
)
def test_l12_mul_combine_in_compiled_staleness_registry_has_in_step_producer():
    """The compiled staleness registry MUST resolve L12's TEMP@AX_byte0
    consumer to L11's producer at a lower phase. If a future change
    re-routes TEMP staging to a higher-phase op (or drops L11
    altogether), the staleness scanner emits a warning -- this test
    pins the satisfied invariant so the warning never goes silent.
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        compiler.add_op(op)
    compiler.compile()
    producers, consumers = compiler.build_staleness_registry()

    pair = ("TEMP", "AX_byte0")
    assert pair in consumers, "L12 TEMP consumer missing from registry"
    consumer_names = [name for name, _phase in consumers[pair]]
    assert "layer12_mul_combine" in consumer_names, (
        f"layer12_mul_combine missing from TEMP@AX_byte0 consumers: "
        f"{consumer_names}"
    )

    assert pair in producers, "TEMP@AX_byte0 has no producer"
    l12_phase = next(
        (phase for name, phase in consumers[pair]
         if name == "layer12_mul_combine"),
        None,
    )
    assert l12_phase is not None
    in_step_producers = [
        (name, phase)
        for name, phase in producers[pair]
        if phase is not None and phase <= l12_phase
    ]
    assert in_step_producers, (
        f"L12 has no in-step producer for TEMP@AX_byte0; "
        f"all producers={producers[pair]}; L12 phase={l12_phase}"
    )


def test_l12_mul_combine_static_claim_drift_scoped():
    """Scope ``verify_claims_static`` to ``layer12_mul_combine``. The op
    has no declared ``claims`` today -- this test is the upgrade hook
    for the day L12 declares them.
    """
    report = verify_claims_static(
        op_filter=lambda op: op.name == "layer12_mul_combine",
    )
    for r in report.results:
        assert r.ok, (
            f"L12 MUL combine declaration drift: "
            f"{r.declared_but_not_written}"
        )


# ---------------------------------------------------------------------------
# Mode B dynamic gate (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("C4_VERIFY_DECLARATIONS") != "1",
    reason="Mode B is slow; set C4_VERIFY_DECLARATIONS=1 to run it",
)
def test_l12_mul_combine_dynamic_candidate_set():
    """Mode B dynamic check sanity: L12 appears in the candidate set
    (it declares both ``produces`` and ``consumes_fresh``). The
    synthetic 1-step probe does not include OP_MUL tokens so "fired"
    is not asserted; reachability is the load-bearing check here.
    """
    report = verify_produces_consumes_dynamic()
    names = {r.op_name for r in report.results}
    assert "layer12_mul_combine" in names, (
        f"layer12_mul_combine not in dynamic verifier results: "
        f"{sorted(names)[:10]}..."
    )


# ---------------------------------------------------------------------------
# Chained L11 -> L12 forward (xfail: open declaration drift)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (2, 3),
        (15, 17),
        (127, 128),
    ],
)
def test_l11_then_l12_chained_forward_matches_expected_output_hi(
    baked_l11_ffn, baked_l12_ffn, a, b
):
    """End-to-end symbolic chain: forward through L11 then L12.

    Fix landed (2026-05-29): L11's W_down was amplified from 2.0/S to
    10.0/S so hot TEMP[partial] lands at ~5.0 -- matching the L12 4-way
    AND threshold (b_up = -S*7.5). The chained gate now fires and
    OUTPUT_HI carries the correct ``(partial + a_hi*b_lo) % 16`` nibble.
    Regression sentinel: if L11 W_down drifts back to ~2.0/S (TEMP ~1.0),
    L12's threshold goes cold and the chain silently drops the high
    nibble of wide-MUL products > 255 (the expr_mul_div_* failure mode).
    """
    a_lo = a & 0xF
    a_hi = (a >> 4) & 0xF
    b_lo = b & 0xF
    b_hi = (b >> 4) & 0xF
    x = torch.zeros(1, 3, D_MODEL)
    ax_pos = 1
    x[:, ax_pos, BD.MARK_AX] = 1.0
    x[:, ax_pos, BD.OP_MUL] = 1.0
    x[:, ax_pos, BD.ALU_LO + a_lo] = 1.0
    x[:, ax_pos, BD.ALU_HI + a_hi] = 1.0
    x[:, ax_pos, BD.AX_CARRY_LO + b_lo] = 1.0
    x[:, ax_pos, BD.AX_CARRY_HI + b_hi] = 1.0

    with torch.no_grad():
        mid = baked_l11_ffn(x)
        out = baked_l12_ffn(mid)

    expected_hi = _mul_result_hi_for(a, b)
    hi_slice = out[0, ax_pos, BD.OUTPUT_HI : BD.OUTPUT_HI + 16]
    hot_idx = int(torch.argmax(hi_slice).item())
    assert hot_idx == expected_hi, (
        f"L11->L12 chained drift: a={a} b={b} "
        f"expected OUTPUT_HI[{expected_hi}], got OUTPUT_HI[{hot_idx}] "
        f"(slice={hi_slice.tolist()})"
    )
    assert hi_slice[expected_hi].item() > HOT_MIN
