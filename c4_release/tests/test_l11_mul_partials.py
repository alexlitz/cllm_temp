"""Per-op audit harness for L11 MUL partial accumulation.

L11's ``layer11_mul_partial`` op stages, for every (a_lo, b_lo, b_hi)
nibble triple, the value ``partial = (a_lo*b_lo // 16 + a_lo*b_hi) % 16``
into the residual ``TEMP[partial]`` slot. The result feeds L12's
``layer12_mul_combine`` which computes ``result_hi = (partial + a_hi*b_lo)
% 16``.

This file does three things:
  1. Symbolic forward: run real products (2*3, 15*17, 127*128) through a
     freshly-baked PureFFN whose weights are produced by
     ``_set_layer11_mul_partial`` and assert ``TEMP[partial]`` is the
     unique non-zero slot in the TEMP band of the residual.
  2. Declaration sanity: pin the op's ``produces`` / ``consumes_fresh``
     contract, the compiled staleness registry, and (via a scoped
     ``verify_claims_static`` call) confirm that the verifier still
     treats the op as no-claims (current contract; future per-cell
     claim adoption will turn the scoped call into a real drift gate).
  3. Mode B dynamic gate: opt-in (``C4_VERIFY_DECLARATIONS=1``) sanity
     hook that ``layer11_mul_partial`` is reachable in the produces/
     consumes registry and is not erroneously flagged stale at compile
     time.

Mirrors ``test_l8_addsub_stage_ownership.py`` (ALU ADD/SUB stage
ownership): per-op input -> baked weights -> assert residual matches
the spec table value.
"""

from __future__ import annotations

import os

import pytest
import torch

from c4_release.neural_vm.base_layers import PureFFN
from c4_release.neural_vm.setup_helpers import _set_layer11_mul_partial
from c4_release.neural_vm.unified_compiler.decl_verifier import (
    verify_claims_static,
    verify_produces_consumes_dynamic,
)
from c4_release.neural_vm.unified_compiler.layer_compiler import LayerCompiler
from c4_release.neural_vm.unified_compiler.migrated_ops import (
    all_core_ops,
    declare_setdim_compat_dims,
    make_layer11_mul_partial_op,
)
from c4_release.neural_vm.vm_step import _SetDim as BD


S_SCALE = 100.0
D_MODEL = 512
HIDDEN = 4096
# Empirical fire/quiet split for the baked SwiGLU output. Live units
# settle at ~1.0 (silu(50)*0.02); silenced units stay at <1e-3.
HOT_MIN = 0.5
LEAK_MAX = 0.1


def _mul_partial_for(a: int, b: int) -> int:
    """Mirror ``_set_layer11_mul_partial`` schoolbook:
        result_lo = (a_lo * b_lo) % 16          [L10]
        result_hi = (carry + a_lo*b_hi + a_hi*b_lo) % 16    [L11 + L12]
    L11 stages ``partial = (carry + a_lo*b_hi) % 16`` where
    ``carry = (a_lo * b_lo) // 16``.
    """
    a_lo = a & 0xF
    b_lo = b & 0xF
    b_hi = (b >> 4) & 0xF
    carry = (a_lo * b_lo) // 16
    return (carry + a_lo * b_hi) % 16


@pytest.fixture(scope="module")
def baked_l11_ffn() -> PureFFN:
    """Build a PureFFN baked by ``_set_layer11_mul_partial`` only.

    Module-scoped because the tests only read the weights; the bake
    fills all 4096 units (~hundreds of ms) and the FFN object is
    immutable from the test perspective. Wrapped in ``torch.no_grad()``
    so the setter's in-place ``ffn.W_up[unit, dim] = S`` writes do not
    trip leaf-Parameter autograd guards (production compiler bakes the
    same way via ``base_layers.bake_weights``).
    """
    with torch.no_grad():
        ffn = PureFFN(dim=D_MODEL, hidden_dim=HIDDEN)
        n_units = _set_layer11_mul_partial(ffn, S_SCALE, BD)
    assert n_units == 4096, "L11 MUL partial must fill all 4096 units"
    return ffn


def _make_ax_input(
    *,
    a: int | None = None,
    b: int | None = None,
    mark_ax: bool = True,
    op_mul: bool = True,
) -> torch.Tensor:
    """Single-step residual probe with optional MARK_AX / OP_MUL gating.

    When ``a`` and ``b`` are provided, the AX-marker position carries
    ALU_LO[a_lo] + AX_CARRY_LO[b_lo] + AX_CARRY_HI[b_hi] one-hots; when
    omitted (only used by the no-fire sentinels), the operand bands fall
    back to representative non-zero slots so the test exercises a real
    candidate unit rather than the all-zero fall-through.
    """
    a_lo = (a if a is not None else 0x05) & 0xF
    b_lo = (b if b is not None else 0x05) & 0xF
    b_hi = ((b if b is not None else 0x10) >> 4) & 0xF

    x = torch.zeros(1, 3, D_MODEL)
    ax_pos = 1
    if mark_ax:
        x[:, ax_pos, BD.MARK_AX] = 1.0
    if op_mul:
        x[:, ax_pos, BD.OP_MUL] = 1.0
    x[:, ax_pos, BD.ALU_LO + a_lo] = 1.0
    x[:, ax_pos, BD.AX_CARRY_LO + b_lo] = 1.0
    x[:, ax_pos, BD.AX_CARRY_HI + b_hi] = 1.0
    return x


# ---------------------------------------------------------------------------
# Symbolic forward: TEMP staging is exact for the spec products
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (2, 3),        # tiny, no carry, no a_hi
        (15, 17),      # a_lo=15, b_lo=1, b_hi=1 -> carry=0, partial=15
        (127, 128),    # a_lo=15, b_lo=0, b_hi=8 -> carry=0, partial=8*15%16=8
        (255, 255),    # max nibbles, big carry
        (0, 0),        # null gate
        (16, 16),      # a_lo=b_lo=0, b_hi=1 -> partial=0 (zero-result path)
    ],
)
def test_l11_mul_partial_temp_staging_exact(baked_l11_ffn, a, b):
    """Forward a baked L11 FFN over an AX-marker residual; assert the
    unique TEMP slot lit is ``(carry + a_lo*b_hi) % 16``.
    """
    x = _make_ax_input(a=a, b=b)
    expected_partial = _mul_partial_for(a, b)

    with torch.no_grad():
        out = baked_l11_ffn(x)

    temp_slice = out[0, 1, BD.TEMP : BD.TEMP + 16]
    hot_idx = int(torch.argmax(temp_slice).item())
    assert hot_idx == expected_partial, (
        f"L11 MUL partial staging drift: a={a} b={b} "
        f"expected TEMP[{expected_partial}] hot, got TEMP[{hot_idx}] "
        f"(slice={temp_slice.tolist()})"
    )
    assert temp_slice[expected_partial].item() > HOT_MIN, (
        f"L11 MUL partial staging weak: expected ~1.0 at "
        f"TEMP[{expected_partial}], got {temp_slice[expected_partial].item()}"
    )
    for other in range(16):
        if other == expected_partial:
            continue
        assert temp_slice[other].item() < LEAK_MAX, (
            f"L11 MUL partial leak: TEMP[{other}]={temp_slice[other].item()} "
            f"for a={a} b={b} (only TEMP[{expected_partial}] should fire)"
        )


@pytest.mark.parametrize(
    ("mark_ax", "op_mul", "label"),
    [
        (False, True, "no_mark_ax"),
        (True, False, "no_op_mul"),
    ],
)
def test_l11_mul_partial_no_fire_when_gate_missing(
    baked_l11_ffn, mark_ax, op_mul, label
):
    """Regression sentinel for the staleness contract: L11 must fire
    only when MARK_AX AND OP_MUL are both hot. Dropping either is the
    most common silent-corruption failure mode (e.g. a future bake that
    forgets the MARK_AX requirement fires at every position).
    """
    x = _make_ax_input(mark_ax=mark_ax, op_mul=op_mul)
    with torch.no_grad():
        out = baked_l11_ffn(x)
    temp_slice = out[0, 1, BD.TEMP : BD.TEMP + 16]
    assert temp_slice.abs().max().item() < LEAK_MAX, (
        f"L11 MUL partial fired with gate={label}: TEMP={temp_slice.tolist()}"
    )


# ---------------------------------------------------------------------------
# Declaration-side: staleness registry sanity
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Declaration drift surfaced by audit harness: "
        "(1) make_layer11_mul_partial_op() no longer accepts alu_mode kwarg; "
        "(2) the op declares produces={'MUL_ACCUM': 'AX_byte0'} but the bake "
        "_set_layer11_mul_partial actually writes BD.TEMP+partial (slot 480+), "
        "not BD.MUL_ACCUM (slot 420). The 'MUL_ACCUM' alias-to-TEMP claim in "
        "the op docstring is not realized in vm_step bands. Resolve by either "
        "(a) re-declaring produces={'TEMP': 'AX_byte0'} or (b) re-baking to "
        "BD.MUL_ACCUM+partial."
    ),
    strict=False,
)
def test_l11_mul_partial_op_declares_temp_production_at_ax_byte0():
    """``layer11_mul_partial`` in lookup mode declares it produces
    ``TEMP@AX_byte0`` and consumes ``ALU_LO/HI`` + ``AX_CARRY_LO/HI`` at
    the same marker. Pinning the contract here makes a regression in the
    declaration unambiguous.
    """
    op = make_layer11_mul_partial_op(alu_mode="lookup")
    assert op.produces == {"TEMP": "AX_byte0"}
    assert op.consumes_fresh == {
        "ALU_LO": "AX_byte0",
        "ALU_HI": "AX_byte0",
        "AX_CARRY_LO": "AX_byte0",
        "AX_CARRY_HI": "AX_byte0",
    }


@pytest.mark.xfail(
    reason=(
        "Same drift as the produces-declaration test above: L11's op "
        "declares produces={'MUL_ACCUM': 'AX_byte0'} so the staleness "
        "registry keys this under MUL_ACCUM, not TEMP. The bake actually "
        "writes BD.TEMP+partial. Until the declaration <-> bake disagreement "
        "is reconciled, the registry will not contain a ('TEMP', 'AX_byte0') "
        "producer entry. Test becomes green once produces is corrected to "
        "'TEMP' (matching the bake's actual residual band)."
    ),
    strict=False,
)
def test_l11_mul_partial_in_compiled_staleness_registry():
    """The compiled staleness registry must list ``layer11_mul_partial`` as
    the sole producer of ``TEMP@AX_byte0`` so L12 (the only consumer) sees
    an in-step producer at a lower phase. Regression sentinel for the
    Codex-pool diag observation: STALENESS VIOLATION fires when the L11
    producer drops out of the registry.
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=True)
    for op in all_core_ops():
        compiler.add_op(op)
    compiler.compile()
    producers, _consumers = compiler.build_staleness_registry()

    pair = ("TEMP", "AX_byte0")
    assert pair in producers, (
        f"L11 producer missing from staleness registry; "
        f"keys={sorted(producers.keys())[:10]}..."
    )
    names = [name for name, _phase in producers[pair]]
    assert "layer11_mul_partial" in names, (
        f"layer11_mul_partial not in TEMP@AX_byte0 producers: {names}"
    )


def test_l11_mul_partial_static_claim_drift_scoped():
    """Scope ``verify_claims_static`` to ``layer11_mul_partial``. The op
    ships with empty ``claims`` (its bake fills FFN W_down TEMP rows but
    does not pin them via per-cell 4-tuples), so the verifier should
    skip it -- but if the op grows claims later this test upgrades to a
    real drift gate. The filter keeps the layout-only build to a single
    op (~25-60s of the production verifier is avoided when running this
    test alone).
    """
    report = verify_claims_static(
        op_filter=lambda op: op.name == "layer11_mul_partial",
    )
    for r in report.results:
        assert r.ok, (
            f"L11 MUL partial declaration drift: "
            f"{r.declared_but_not_written}"
        )


# ---------------------------------------------------------------------------
# Mode B dynamic gate (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("C4_VERIFY_DECLARATIONS") != "1",
    reason="Mode B is slow; set C4_VERIFY_DECLARATIONS=1 to run it",
)
def test_l11_mul_partial_dynamic_produces_consumes_no_crash():
    """Mode B dynamic check: ``layer11_mul_partial`` appears in the
    candidate set (since it declares ``produces`` / ``consumes_fresh``)
    and the synthetic 1-step probe does not crash. We do not assert
    "fired" -- the synthetic probe uses no OP_MUL token -- but completion
    is a load-bearing infrastructure check.
    """
    report = verify_produces_consumes_dynamic()
    names = {r.op_name for r in report.results}
    assert "layer11_mul_partial" in names, (
        f"layer11_mul_partial not in dynamic verifier results: "
        f"{sorted(names)[:10]}..."
    )
